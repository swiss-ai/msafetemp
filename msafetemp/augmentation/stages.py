"""Language-agnostic stage calls of the template pipeline.

Each `run_*` function makes one model call and returns (parsed_result_or_None, raw_text).
Judges answer independent per-criterion booleans; the pass/reject verdict and the list of
failed criteria are computed here in code, never self-reported by the judge (a
safety-tuned judge sometimes described a variant as harmful in its reasoning and still
returned the opposite overall verdict). JUDGE1 is the exception and returns a verdict
directly.

Sampling settings (configs/augmentation/models_and_settings.yaml): creative stages
use nonzero temperature, judges temperature 0 with large token budgets.
"""

from __future__ import annotations

from typing import Any

from msafetemp.common.llm import chat, parse_json_object, parse_string_list, track_prompt_usage
from msafetemp.augmentation.prompts.core import (
    EXTRACTOR_SYSTEM, EXTRACTOR_USER, FEEDBACK_SECTION,
    JUDGE1_SYSTEM, JUDGE1_USER,
    JUDGE2_SYSTEM, JUDGE2_USER,
    PARAPHRASER_SYSTEM, PARAPHRASER_USER,
    PARA_JUDGE_SYSTEM, PARA_JUDGE_USER,
    PREFIX_GENERATOR_SYSTEM, PREFIX_GENERATOR_USER,
    SUFFIX_GENERATOR_SYSTEM, SUFFIX_GENERATOR_USER,
    JAILBREAK_EXTRACTOR_SYSTEM, JAILBREAK_EXTRACTOR_USER,
)


def build_slots_block(slots: list[dict]) -> str:
    return "\n".join(
        f"  <{s.get('name','?').strip('<>')}> — {s.get('description','')}" for s in slots
    ) or "  (none)"


def _verdict_from_booleans(parsed: dict[str, Any] | None,
                           criteria: list[tuple[str, str, str]]) -> dict[str, Any] | None:
    """verdict / failed_criteria from independent booleans. A missing boolean means the
    reply is treated as unparsed (None) so the caller retries."""
    if parsed is None:
        return None
    failed = []
    reasoning_parts = []
    for bool_key, label, reasoning_key in criteria:
        if bool_key not in parsed:
            return None
        reasoning_parts.append(f"{label}: {parsed.get(reasoning_key, '')}")
        if not parsed[bool_key]:
            failed.append(label)
    return {
        "reasoning": " ".join(reasoning_parts),
        "verdict": "reject" if failed else "pass",
        "failed_criteria": failed,
        "feedback": parsed.get("feedback", ""),
    }


# ---------------------------------------------------------------------------
# Stage 1-2: template extraction + validity judge (English)
# ---------------------------------------------------------------------------

def run_extractor(client, model: str, prompt: str,
                  feedback: str | None = None) -> tuple[dict[str, Any] | None, str]:
    feedback_section = FEEDBACK_SECTION.format(feedback=feedback) if feedback else ""
    user_msg = EXTRACTOR_USER.format(prompt=prompt, feedback_section=feedback_section)
    raw = chat(client, model, [
        {"role": "system", "content": EXTRACTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=8192, temperature=0.2)
    return parse_json_object(raw), raw


def run_judge1(client, model: str, original_prompt: str, extraction: dict[str, Any],
               system: str | None = None) -> tuple[dict[str, Any] | None, str]:
    track_prompt_usage("judge1", system is not None)
    user_msg = JUDGE1_USER.format(
        original_prompt=original_prompt,
        template=extraction.get("template", ""),
        slots_formatted=build_slots_block(extraction.get("slots", [])),
        toxic_vector=extraction.get("toxic_vector", ""),
    )
    raw = chat(client, model, [
        {"role": "system", "content": system or JUDGE1_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=32768, temperature=0.0)
    return parse_json_object(raw), raw


# ---------------------------------------------------------------------------
# Stage 5: four-axis variant judge (shared by TR and PH)
# ---------------------------------------------------------------------------

JUDGE2_CRITERIA = [
    ("is_genuinely_harmful", "HARMFUL", "reasoning_harmful"),
    ("is_language_consistent", "LANGUAGE_CONSISTENCY", "reasoning_language"),
    ("is_coherent", "COHERENT", "reasoning_coherent"),
    ("is_framing_intact", "FRAMING_INTACT", "reasoning_framing"),
]


def run_judge2(client, model: str, template: str, variant: str,
               system: str | None = None) -> tuple[dict[str, Any] | None, str]:
    track_prompt_usage("judge2", system is not None)
    user_msg = JUDGE2_USER.format(template=template, variant=variant)
    raw = chat(client, model, [
        {"role": "system", "content": system or JUDGE2_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=32768, temperature=0.0)
    return _verdict_from_booleans(parse_json_object(raw), JUDGE2_CRITERIA), raw


# ---------------------------------------------------------------------------
# Paraphrases
# ---------------------------------------------------------------------------

def run_paraphraser(client, model: str, extraction: dict[str, Any],
                    num_paraphrases: int) -> tuple[list[str] | None, str]:
    slots = extraction.get("slots", [])
    user_msg = PARAPHRASER_USER.format(
        n=num_paraphrases,
        template=extraction.get("template", ""),
        slots_block=build_slots_block(slots),
    )
    # N paraphrases in one array; long jailbreak scaffolds need the token headroom.
    raw = chat(client, model, [
        {"role": "system", "content": PARAPHRASER_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=32768, temperature=0.9)
    return parse_string_list(raw), raw


PARA_JUDGE_CRITERIA = [
    ("is_self_contained", "SELF_CONTAINED", "reasoning_self_contained"),
    ("is_slot_integrity_preserved", "SLOT_INTEGRITY", "reasoning_slot_integrity"),
    ("is_harm_preserved", "HARM_PRESERVED", "reasoning_harm"),
    ("is_language_consistent", "LANGUAGE_CONSISTENCY", "reasoning_language"),
]


def run_para_judge(client, model: str, original_template: str, para_template: str,
                   system: str | None = None) -> tuple[dict[str, Any] | None, str]:
    track_prompt_usage("para_judge", system is not None)
    user_msg = PARA_JUDGE_USER.format(original_template=original_template,
                                      para_template=para_template)
    raw = chat(client, model, [
        {"role": "system", "content": system or PARA_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=32768, temperature=0.0)
    return _verdict_from_booleans(parse_json_object(raw), PARA_JUDGE_CRITERIA), raw


# ---------------------------------------------------------------------------
# Unjudged attack prefixes / suffixes (template-level list columns in the release)
# ---------------------------------------------------------------------------

def _run_list_generator(client, model: str, sys_prompt: str,
                        user_prompt: str) -> tuple[list[str] | None, str]:
    raw = chat(client, model, [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ], max_tokens=8192, temperature=0.9)
    return parse_string_list(raw), raw


def run_prefix_generator(client, model: str, template: str, n: int):
    return _run_list_generator(client, model, PREFIX_GENERATOR_SYSTEM,
                               PREFIX_GENERATOR_USER.format(n=n, template=template))


def run_suffix_generator(client, model: str, template: str, n: int):
    return _run_list_generator(client, model, SUFFIX_GENERATOR_SYSTEM,
                               SUFFIX_GENERATOR_USER.format(n=n, template=template))


def run_affix_generator(client, model: str, template: str, n: int, gen_fn) -> list[str]:
    """Two attempts; an empty list if both fail to parse."""
    result = None
    for _ in range(2):
        try:
            result, _ = gen_fn(client, model, template, n)
        except Exception:
            pass
        if result is not None:
            break
    return result or []


# ---------------------------------------------------------------------------
# Jailbreak-wrapper extractor
# ---------------------------------------------------------------------------

def run_jailbreak_extractor(client, model: str, wrapper: str,
                            feedback: str | None = None) -> tuple[dict[str, Any] | None, str]:
    """Abstract a jailbreak wrapper's swappable scaffold (persona, framing, format devices)
    into typed slots while keeping the fixed payload slot <harmful_request> verbatim."""
    feedback_section = FEEDBACK_SECTION.format(feedback=feedback) if feedback else ""
    user_msg = JAILBREAK_EXTRACTOR_USER.format(prompt=wrapper, feedback_section=feedback_section)
    # Echoes the whole scaffold; long in-the-wild wrappers reach ~7k tokens.
    raw = chat(client, model, [
        {"role": "system", "content": JAILBREAK_EXTRACTOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=32768, temperature=0.2)
    return parse_json_object(raw), raw


# ---------------------------------------------------------------------------
# Debug printing
# ---------------------------------------------------------------------------

def print_extraction(e: dict) -> None:
    if e.get("language"):
        print(f"  language:     {e['language']}")
    print(f"  toxic_vector: {e.get('toxic_vector', '-')}")
    print(f"  template:     {e.get('template', '-')}")
    for s in e.get("slots", []):
        print(f"    <{s.get('name','?').strip('<>')}> - {s.get('description','')}")
    r = e.get("reasoning", "")
    if r:
        print(f"  reasoning:    {r}")


def print_judge1(j: dict) -> None:
    verdict = j.get("verdict", "?")
    failed = j.get("failed_criteria", [])
    print(f"  judge1: {verdict.upper()}" + (f"  failed={failed}" if failed else ""))
    if j.get("reasoning"):
        print(f"  reasoning: {j['reasoning']}")
    if j.get("feedback"):
        print(f"  feedback:  {j['feedback']}")


def print_variants(variants: list, label: str = "variants", judge_name: str = "j2") -> None:
    passing = sum(1 for _, j in variants if j is None or j.get("verdict") == "pass")
    print(f"  {label}: {len(variants)} generated, {passing} passed")
    for i, (v, j) in enumerate(variants, 1):
        suffix = ""
        if j is not None:
            failed = j.get("failed_criteria", [])
            suffix = f"  [{j.get('verdict')}] {judge_name}" + (f" failed={failed}" if failed else "")
        print(f"  [{i}]{suffix}: {v}")
        if j and j.get("verdict") != "pass" and j.get("feedback"):
            print(f"    -> {j['feedback']}")
