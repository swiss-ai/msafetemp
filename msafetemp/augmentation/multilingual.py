"""The multilingual template pipeline for one English seed prompt and one target language.

Stage order:
  1  English extractor      harm-bearing spans -> typed <slot> labels
  2  English judge1         template validity; a rejected extraction is retried with the
                            judge's feedback (max_retries extra attempts)
  3  template translator    surrounding text translated, slot labels kept in English
                            (skipped for English)
  3b translation judge      harm / slot integrity / language; on reject a patcher edits
                            only the flagged spans; after max_retries feedback rounds the
                            best translation is kept with its validation record
  4  harmful generator      TR fills of the native template, plus fills of each validated
                            paraphrase (paraphraser + paraphrase judge)
  5  judge2                 four-axis variant judge; a rejected variant with feedback is
                            regenerated once and re-judged
  4b/5b benign branch       exact-fill benign minimal pairs + class judge (OR)
  affixes                   optional unjudged prefix/suffix lists

The TR, paraphrase and benign branches run concurrently off the same translated template.

`track` selects the per-track settings that differed between the two released tracks:
  wildguard  literal translator prompt; patcher and generator max_tokens 32768 (long
             adversarial scaffolds need the headroom)
  harmbench  localization translator prompt; patcher and generator max_tokens 8192
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from msafetemp.common.llm import chat, parse_json_object, parse_string_list, track_prompt_usage
from msafetemp.augmentation.prompts.core import (
    GENERATOR_SYSTEM, GENERATOR_USER, VARIANT_FEEDBACK_SECTION,
    JUDGE1_SYSTEM_CLINICAL, JUDGE2_SYSTEM_CLINICAL, PARA_JUDGE_SYSTEM_CLINICAL,
)
from msafetemp.augmentation.prompts.translation import (
    TRANSLATOR_PROMPTS, CULTURAL_CONTEXT_SECTION, TRANSLATOR_FEEDBACK_SECTION,
    TRANSLATION_PATCHER_SYSTEM, TRANSLATION_PATCHER_USER,
    TRANSLATION_JUDGE_SYSTEM, TRANSLATION_JUDGE_SYSTEM_CLINICAL, TRANSLATION_JUDGE_USER,
)
from msafetemp.augmentation.prompts.benign import (
    BENIGN_EXACTFILL_GENERATOR_SYSTEM, BENIGN_EXACTFILL_GENERATOR_USER,
    BENIGN_EXACTFILL_JUDGE_SYSTEM, BENIGN_EXACTFILL_JUDGE_SYSTEM_CLINICAL,
    BENIGN_EXACTFILL_JUDGE_USER,
)
from msafetemp.augmentation.stages import (
    build_slots_block, run_extractor, run_judge1, run_judge2, run_paraphraser,
    run_para_judge, run_prefix_generator, run_suffix_generator, run_affix_generator,
    _verdict_from_booleans,
)

TRACKS = ("wildguard", "harmbench")

# Per-track token budgets (see module docstring).
_PATCHER_MAX_TOKENS = {"wildguard": 32768, "harmbench": 8192}
_GENERATOR_MAX_TOKENS = {"wildguard": 32768, "harmbench": 8192}


# ---------------------------------------------------------------------------
# Stage 3 / 3b: translation, translation judge, targeted patcher
# ---------------------------------------------------------------------------

def run_template_translator(client, model: str, extraction: dict[str, Any], target_language: str,
                            cultural_guidance: str | None = None, feedback_section: str = "",
                            track: str = "wildguard") -> tuple[dict[str, Any] | None, str]:
    """Translate an English template; slot labels stay in English. Optional cultural
    guidance (context_source other than `none`) steers idiomatic phrasing; the released
    data were generated with context_source=none."""
    system, user_tmpl = TRANSLATOR_PROMPTS[track]
    ctx_section = ""
    if cultural_guidance:
        ctx_section = CULTURAL_CONTEXT_SECTION.format(
            target_language=target_language, cultural_guidance=cultural_guidance)
    user_msg = user_tmpl.format(
        target_language=target_language,
        template=extraction.get("template", ""),
        slots_block=build_slots_block(extraction.get("slots", [])),
        cultural_context_section=ctx_section,
        feedback_section=feedback_section,
    )
    raw = chat(client, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ], max_tokens=8192, temperature=0.3)
    return parse_json_object(raw), raw


def run_translation_patcher(client, model: str, english_template: str, translated_template: str,
                            feedback: str, target_language: str,
                            track: str = "wildguard") -> tuple[dict[str, Any] | None, str]:
    """Minimally edit an existing translation to fix the errors the judge named, instead
    of retranslating from scratch (which reintroduced or reverted fixes in spans the judge
    had not flagged)."""
    user_msg = TRANSLATION_PATCHER_USER.format(
        target_language=target_language,
        target_language_upper=target_language.upper(),
        english_template=english_template,
        translated_template=translated_template,
        feedback=feedback,
    )
    raw = chat(client, model, [
        {"role": "system", "content": TRANSLATION_PATCHER_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=_PATCHER_MAX_TOKENS[track], temperature=0.2)
    return parse_json_object(raw), raw


TRANSLATION_JUDGE_CRITERIA = [
    ("is_harm_preserved", "HARM_PRESERVED", "reasoning_harm"),
    ("is_slot_integrity_preserved", "SLOT_INTEGRITY", "reasoning_slot_integrity"),
    ("is_language_consistent", "LANGUAGE_CONSISTENCY", "reasoning_language"),
]


def run_translation_judge(client, model: str, english_template: str, translated_template: str,
                          target_language: str,
                          system: str | None = None) -> tuple[dict[str, Any] | None, str]:
    track_prompt_usage("translation_judge", system is not None)
    user_msg = TRANSLATION_JUDGE_USER.format(
        target_language=target_language,
        target_language_upper=target_language.upper(),
        english_template=english_template,
        translated_template=translated_template,
    )
    raw = chat(client, model, [
        {"role": "system", "content": system or TRANSLATION_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=32768, temperature=0.0)
    return _verdict_from_booleans(parse_json_object(raw), TRANSLATION_JUDGE_CRITERIA), raw


# ---------------------------------------------------------------------------
# Benign (OR) branch: exact-fill minimal pairs
# ---------------------------------------------------------------------------
# The generator fills only the <slot> markers; the fixed template text stays verbatim, so
# the resolved prompt still SOUNDS suspicious while being safe to answer. Exact
# substitution is re-verified in code; the judge only checks answerability, coherence and
# language, never punishing the suspicious surface.

def _benign_slot_marker(raw_name: Any) -> str | None:
    if not isinstance(raw_name, str):
        return None
    name = raw_name.strip()
    if name.startswith("<") and name.endswith(">"):
        name = name[1:-1]
    if not name or "<" in name or ">" in name:
        return None
    return f"<{name}>"


def _benign_declared_markers(slots: list) -> list[str]:
    markers: list[str] = []
    for slot in slots or []:
        raw = slot.get("name") if isinstance(slot, dict) else slot
        marker = _benign_slot_marker(raw)
        if marker and marker not in markers:
            markers.append(marker)
    return markers


def materialize_benign_candidate(template: str, slots: list,
                                 item: dict) -> tuple[str | None, str | None]:
    """Literal slot replacement of one generator candidate; (text, None) or (None, error)."""
    markers = _benign_declared_markers(slots)
    if not markers:
        return None, "source has no valid declared slots"
    required = set(markers)
    if any(marker not in template for marker in required):
        return None, "a declared slot is missing from the template"
    fills = item.get("fills")
    if not isinstance(fills, dict):
        return None, "fills must be an object"
    normalized: dict[str, str] = {}
    for raw_key, raw_value in fills.items():
        key = _benign_slot_marker(raw_key)
        if key is None or key not in required:
            return None, "fills contain an unknown slot"
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None, "a fill is empty"
        value = raw_value.strip()
        if any(marker in value for marker in required):
            return None, "a fill contains a slot marker"
        if key in normalized:
            return None, "a slot is filled twice"
        normalized[key] = value
    if set(normalized) != required:
        return None, "fills must match the template slots exactly"
    text = template
    for marker in sorted(required, key=len, reverse=True):
        text = text.replace(marker, normalized[marker])
    if any(marker in text for marker in required):
        return None, "an unresolved slot remains"
    filled = item.get("filled_prompt")
    if isinstance(filled, str) and filled.strip() and filled != text:
        return None, "filled_prompt is not the exact literal slot replacement"
    return text, None


def run_benign_exactfill_generator(client, model: str, extraction: dict[str, Any],
                                   num_benign: int) -> tuple[list[dict] | None, str]:
    slots = extraction.get("slots", [])
    user_msg = BENIGN_EXACTFILL_GENERATOR_USER.format(
        n=num_benign,
        template=extraction.get("template", ""),
        slots_block=build_slots_block(slots),
        language=extraction.get("language", "English"),
    )
    raw = chat(client, model, [
        {"role": "system", "content": BENIGN_EXACTFILL_GENERATOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=8192, temperature=0.8)
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("candidates"), list):
        return None, raw
    return [c for c in parsed["candidates"] if isinstance(c, dict)], raw


# Independent booleans the judge must return (verdict and class computed in code).
_BENIGN_EXACTFILL_BOOLS = (
    "is_safe_to_answer",
    "is_fully_answerable_without_transferable_harm",
    "is_coherent",
    "is_language_consistent",
)


def benign_class_from_booleans(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    """Quality gate first, then toxicity, then the direct / bounded (transferability) split."""
    if parsed is None:
        return None
    for key in _BENIGN_EXACTFILL_BOOLS:
        if not isinstance(parsed.get(key), bool):
            return None

    reasoning = " ".join([
        f"SAFE: {parsed.get('reasoning_safety', '')}",
        f"TRANSFERABLE: {parsed.get('reasoning_transferable', '')}",
        f"COHERENT: {parsed.get('reasoning_coherent', '')}",
        f"LANGUAGE: {parsed.get('reasoning_language', '')}",
    ])

    if not parsed["is_coherent"]:
        cls, failed = "reject_quality", ["INCOHERENT"]
    elif not parsed["is_language_consistent"]:
        cls, failed = "reject_quality", ["LANGUAGE_INCONSISTENT"]
    elif not parsed["is_safe_to_answer"]:
        cls, failed = "toxic", ["TOXIC"]
    elif parsed["is_fully_answerable_without_transferable_harm"]:
        cls, failed = "direct", []
    else:
        cls, failed = "bounded", []

    return {
        "reasoning": reasoning,
        "class": cls,
        "verdict": "pass" if cls in ("direct", "bounded") else "reject",
        "failed_criteria": failed,
        "has_suspicious_surface": bool(parsed.get("has_suspicious_surface")),
        "feedback": parsed.get("feedback", ""),
    }


def run_benign_exactfill_judge(client, model: str, template: str, variant: str, language: str,
                               system: str | None = None) -> tuple[dict[str, Any] | None, str]:
    track_prompt_usage("benign_judge", system is not None)
    user_msg = BENIGN_EXACTFILL_JUDGE_USER.format(template=template, variant=variant,
                                                  language=language)
    raw = chat(client, model, [
        {"role": "system", "content": system or BENIGN_EXACTFILL_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=32768, temperature=0.0)
    return benign_class_from_booleans(parse_json_object(raw)), raw


def process_benign_exactfill(client, model: str, extraction: dict, num_benign: int,
                             skip_generator: bool = False, skip_judge: bool = False,
                             verbose: bool = False, judge_client=None,
                             judge_model: str | None = None):
    """Exact-fill benign branch. Returns [(variant_text, judge_dict)]; judge_dict carries a
    `class` (direct | bounded | toxic | reject_quality) and a pass/reject `verdict`
    (pass = direct or bounded)."""
    if skip_generator or num_benign <= 0:
        return None

    template = extraction.get("template", "")
    slots = extraction.get("slots", [])
    language = extraction.get("language", "English")

    candidates, gen_raw = None, ""
    for attempt in range(2):
        try:
            candidates, gen_raw = run_benign_exactfill_generator(client, model, extraction, num_benign)
        except Exception as e:
            if verbose:
                print(f"  [benign generator error] {e}")
        if candidates is not None:
            break
        if verbose:
            print(f"  [benign generator parse fail attempt {attempt+1}] {gen_raw[:80]}")
    if candidates is None:
        return None

    # Materialize (exact fill verified in code) + dedup, serially.
    materialized: list[tuple[str, str | None]] = []  # (text, schema_error)
    seen = set()
    for item in candidates[:num_benign]:
        text, err = materialize_benign_candidate(template, slots, item)
        if text is None:
            fallback = str(item.get("filled_prompt") or "").strip() or "<unparseable candidate>"
            materialized.append((fallback, err or "invalid candidate"))
            continue
        if text in seen:
            continue
        seen.add(text)
        materialized.append((text, None))

    def _judge_with_retries(text):
        for system in (None, None, BENIGN_EXACTFILL_JUDGE_SYSTEM_CLINICAL):
            try:
                result, _ = run_benign_exactfill_judge(
                    judge_client or client, judge_model or model, template, text, language,
                    system=system)
            except Exception as e:
                if verbose:
                    print(f"  [benign judge error] {e}")
                result = None
            if result is not None:
                return result
        return None

    def _judge_one(entry):
        text, schema_error = entry
        if schema_error is not None:
            return text, {
                "reasoning": schema_error,
                "class": "reject_quality",
                "verdict": "reject",
                "failed_criteria": ["FRAME_CHANGED"],
                "feedback": schema_error,
            }
        if skip_judge:
            return text, None
        judge = _judge_with_retries(text)
        if judge is None:
            return text, {
                "reasoning": "benign judge did not return parseable JSON",
                "class": "reject_quality",
                "verdict": "reject",
                "failed_criteria": ["UNJUDGED"],
                "feedback": "",
            }
        return text, judge

    with ThreadPoolExecutor(max_workers=max(1, len(materialized))) as ex:
        return list(ex.map(_judge_one, materialized))


# ---------------------------------------------------------------------------
# Stage 4 / 5: harmful fills (TR) + judge2, paraphrases
# ---------------------------------------------------------------------------

def run_generator_multilingual(client, model: str, extraction: dict[str, Any],
                               num_variants: int = 3, taxonomy_supplements: dict | None = None,
                               cultural_guidance: str | None = None, feedback_section: str = "",
                               track: str = "wildguard") -> tuple[list[str] | None, str]:
    """Fill the (translated) template's slots. Optional cultural context blocks are
    inserted after the ENTITY SELECTION header; with context_source=none (the released
    runs) the prompt is the plain GENERATOR_USER."""
    slots = extraction.get("slots", [])
    user_msg = GENERATOR_USER.format(
        n=num_variants,
        template=extraction.get("template", ""),
        language=extraction.get("language", ""),
        slots_block=build_slots_block(slots),
        feedback_section=feedback_section,
    )

    inserts = []
    if taxonomy_supplements:
        lines = [
            "─── CULTURAL TAXONOMY ────────────────────────────────────────────────────────",
            "Prefer entities from these culturally specific lists when filling slots:",
        ]
        for category, entities in taxonomy_supplements.items():
            def _fmt(e):
                if isinstance(e, dict):
                    ent = e.get("entity", "")
                    ctx = e.get("societal_context", "")
                    return f"{ent} — {ctx}" if ctx else ent
                return str(e)
            entity_str = "; ".join(_fmt(e) for e in entities[:10])
            lines.append(f"  {category}: {entity_str}")
        lines.append("─────────────────────────────────────────────────────────────────────────────")
        inserts.append("\n".join(lines))
    if cultural_guidance:
        inserts.append(
            "─── CULTURAL CONTEXT ────────────────────────────────────────────────────────\n"
            + cultural_guidance
            + "\n─────────────────────────────────────────────────────────────────────────────"
        )
    if inserts:
        marker = "─── ENTITY SELECTION ───"
        idx = user_msg.find(marker)
        if idx != -1:
            after_header = user_msg.find("\n", idx) + 1
            injection = "\n\n".join(inserts) + "\n\n"
            user_msg = user_msg[:after_header] + injection + user_msg[after_header:]

    raw = chat(client, model, [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=_GENERATOR_MAX_TOKENS[track], temperature=0.9)
    return parse_string_list(raw), raw


def judge2_with_retries(judge_client, judge_model, template, variant, verbose=False):
    """Main system prompt twice, then the clinical framing."""
    j2 = None
    for system in (None, None, JUDGE2_SYSTEM_CLINICAL):
        try:
            j2, _ = run_judge2(judge_client, judge_model, template, variant, system=system)
        except Exception as e:
            if verbose:
                print(f"  [judge2 error] {e}")
        if j2 is not None:
            break
    return j2


def process_variants(client, model, extraction, num_variants, skip_generator, skip_judge2,
                     verbose, judge_client, judge_model, taxonomy_supplements=None,
                     cultural_guidance=None, track="wildguard"):
    if skip_generator:
        return None

    jc = judge_client or client
    jm = judge_model or model
    variants_raw = None
    for attempt in range(2):
        gen_raw = ""
        try:
            variants_raw, gen_raw = run_generator_multilingual(
                client, model, extraction, num_variants,
                taxonomy_supplements=taxonomy_supplements,
                cultural_guidance=cultural_guidance, track=track)
        except Exception as e:
            if verbose:
                print(f"  [generator error attempt {attempt+1}] {e}")
        if variants_raw is not None:
            break
        if verbose:
            print(f"  [generator parse fail attempt {attempt+1}] ...{gen_raw[-200:]!r}")
    if variants_raw is None:
        return None

    template = extraction.get("template", "")

    def _judge_one(variant):
        if skip_judge2:
            return variant, None
        j2 = judge2_with_retries(jc, jm, template, variant, verbose)
        # Rejected with feedback -> regenerate this one variant and re-judge once.
        if j2 is not None and j2.get("verdict") != "pass" and j2.get("feedback"):
            feedback_section = VARIANT_FEEDBACK_SECTION.format(
                rejected_variant=variant, feedback=j2["feedback"])
            try:
                new_variants, _ = run_generator_multilingual(
                    client, model, extraction, 1,
                    taxonomy_supplements=taxonomy_supplements,
                    cultural_guidance=cultural_guidance,
                    feedback_section=feedback_section, track=track)
            except Exception as e:
                if verbose:
                    print(f"  [judge2 feedback-retry generator error] {e}")
                new_variants = None
            if new_variants:
                new_variant = new_variants[0]
                new_j2 = judge2_with_retries(jc, jm, template, new_variant, verbose)
                if new_j2 is not None:
                    return new_variant, new_j2
        return variant, j2

    with ThreadPoolExecutor(max_workers=max(1, len(variants_raw))) as ex:
        return list(ex.map(_judge_one, variants_raw))


def process_paraphrases(client, model, extraction, num_paraphrases, num_variants,
                        skip_generator, skip_judge2, verbose, judge_client, judge_model,
                        taxonomy_supplements=None, cultural_guidance=None, track="wildguard"):
    original_template = extraction.get("template", "")
    para_templates = None
    for _ in range(2):
        try:
            para_templates, _ = run_paraphraser(client, model, extraction, num_paraphrases)
        except Exception as e:
            if verbose:
                print(f"  [paraphraser error] {e}")
        if para_templates is not None:
            break
    if para_templates is None:
        return []

    jc = judge_client or client
    jm = judge_model or model

    def _process_one(tmpl):
        j1, para_raw = None, ""
        for system in (None, None, PARA_JUDGE_SYSTEM_CLINICAL):
            try:
                j1, para_raw = run_para_judge(jc, jm, original_template, tmpl, system=system)
            except Exception as e:
                if verbose:
                    print(f"  [para judge error] {e}")
                para_raw = str(e)
            if j1 is not None:
                break
        if not j1:
            return {"template": tmpl, "judge1": None, "judge_failure_raw": para_raw, "variants": None}
        para_extraction = {**extraction, "template": tmpl}
        para_variants = None
        if j1.get("verdict") == "pass":
            para_variants = process_variants(
                client, model, para_extraction, num_variants, skip_generator, skip_judge2,
                verbose, judge_client, judge_model, taxonomy_supplements, cultural_guidance,
                track)
        return {"template": tmpl, "judge1": j1, "variants": para_variants}

    with ThreadPoolExecutor(max_workers=max(1, len(para_templates))) as ex:
        return list(ex.map(_process_one, para_templates))


# ---------------------------------------------------------------------------
# Full pipeline for one seed and one language
# ---------------------------------------------------------------------------

def run_multilingual_pipeline(client, model: str, english_prompt: str, target_language: str,
                              cultural_context: dict | None = None, max_retries: int = 3,
                              num_variants: int = 5, skip_generator: bool = False,
                              skip_judge2: bool = False, num_paraphrases: int = 2,
                              num_prefixes: int = 0, num_suffixes: int = 0,
                              num_benign: int = 5, verbose: bool = False, judge_client=None,
                              judge_model: str | None = None,
                              track: str = "wildguard") -> dict[str, Any]:
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")
    ctx = cultural_context or {}
    cultural_guidance = ctx.get("cultural_guidance") or None
    taxonomy_supplements = ctx.get("taxonomy_supplements") or None
    jc = judge_client or client
    jm = judge_model or model

    def _build_result(status, attempt, extraction=None, judge1=None, translation=None,
                      translated_template=None, translation_raw=None, translation_judge=None,
                      translation_attempts=None, variants=None, paraphrases=None,
                      prefixes=None, suffixes=None, benign_variants=None, judge_raw=None):
        return {
            "status": status,
            "original_prompt": english_prompt,
            "translation": translation,
            "translated_template": translated_template,
            "translation_raw": translation_raw,
            "translation_judge": translation_judge,
            "translation_attempts": translation_attempts or [],
            "extraction": extraction,
            "judge1": judge1,
            "variants": variants,
            "paraphrases": paraphrases,
            "prefixes": prefixes or [],
            "suffixes": suffixes or [],
            "benign_variants": benign_variants,
            "attempts": attempt,
            "judge_failure_raw": judge_raw,
        }

    def _vlog(msg: str):
        if verbose:
            print(f"  {msg}")

    # Stages 1-2: English extraction + judge1 loop.
    feedback = None
    last_extraction = None
    last_judge1 = None
    for attempt in range(1, max_retries + 2):
        try:
            extraction, ext_raw = run_extractor(client, model, english_prompt, feedback)
        except Exception as e:
            _vlog(f"[extractor error] {e}")
            return _build_result("extractor_failed", attempt)
        if not extraction:
            _vlog(f"[extractor parse fail] ...{ext_raw[-200:]!r}")
            return _build_result("extractor_failed", attempt)
        last_extraction = extraction
        if not extraction.get("templateable", False):
            return _build_result("benign", attempt, extraction=extraction)

        judge1, judge_raw = None, ""
        for system in (None, None, JUDGE1_SYSTEM_CLINICAL):
            try:
                judge1, judge_raw = run_judge1(jc, jm, english_prompt, extraction, system=system)
            except Exception as e:
                _vlog(f"[judge1 error] {e}")
                judge_raw = str(e)
            if judge1:
                break
        if not judge1:
            return _build_result("judge_failed", attempt, extraction=last_extraction,
                                 judge_raw=judge_raw)
        last_judge1 = judge1
        if judge1.get("verdict") == "pass":
            break
        feedback = judge1.get("feedback", "")
        _vlog(f"[judge1 reject #{attempt}] {judge1.get('failed_criteria', [])} -> retrying")
    else:
        return _build_result("reject", max_retries + 1, extraction=last_extraction,
                             judge1=last_judge1)

    # Stage 3: translate the validated English template. Skipped for English: the same
    # generator / judge2 / paraphrase / benign stages then run on the English template,
    # giving a directly comparable English baseline.
    is_english = target_language.lower() == "english"
    translation, trans_raw = None, ""
    last_trans_judge = None
    translation_attempts = []

    if is_english:
        translated_template = extraction.get("template", "")
    else:
        try:
            translation, trans_raw = run_template_translator(
                client, model, extraction, target_language, cultural_guidance, track=track)
        except Exception as e:
            _vlog(f"[translator error] {e}")
            return _build_result("translator_failed", attempt, extraction=extraction,
                                 judge1=judge1, translation_raw=str(e))
        if not translation or not translation.get("translated_template"):
            return _build_result("translator_failed", attempt, extraction=extraction,
                                 judge1=judge1, translation=translation, translation_raw=trans_raw)
        translated_template = translation["translated_template"]

    # Stage 3b: translation judge; patch named errors on reject, full retranslation only
    # when the judge itself never parsed.
    if not is_english:
        for t_attempt in range(1, max_retries + 2):
            trans_judge, trans_judge_raw = None, ""
            for system in (None, None, TRANSLATION_JUDGE_SYSTEM_CLINICAL):
                try:
                    trans_judge, trans_judge_raw = run_translation_judge(
                        jc, jm, extraction.get("template", ""), translated_template,
                        target_language, system=system)
                except Exception as e:
                    _vlog(f"[translation judge error] {e}")
                    trans_judge_raw = str(e)
                if trans_judge is not None:
                    break

            translation_attempts.append({
                "attempt": t_attempt,
                "translated_template": translated_template,
                "verdict": trans_judge.get("verdict") if trans_judge is not None else "unparseable",
                "failed_criteria": trans_judge.get("failed_criteria") if trans_judge is not None else None,
                "reasoning": trans_judge.get("reasoning") if trans_judge is not None else None,
                "feedback": trans_judge.get("feedback") if trans_judge is not None else None,
                "judge_raw_tail": None if trans_judge is not None else trans_judge_raw[-500:],
            })

            if trans_judge is not None:
                last_trans_judge = trans_judge
                if trans_judge.get("verdict") == "pass":
                    break
                feedback_text = trans_judge.get("feedback", "")
            else:
                feedback_text = None

            if t_attempt <= max_retries:
                if feedback_text is not None:
                    # Concrete, named feedback: minimally patch the existing translation so
                    # unflagged spans that were already correct cannot regress.
                    try:
                        retry_translation, _ = run_translation_patcher(
                            client, model, extraction.get("template", ""), translated_template,
                            feedback_text, target_language, track=track)
                    except Exception as e:
                        _vlog(f"[translation patcher error] {e}")
                        retry_translation = None
                else:
                    # The judge never parsed: nothing concrete to patch, retranslate.
                    feedback_section = TRANSLATOR_FEEDBACK_SECTION.format(
                        rejected_translation=translated_template,
                        feedback="The previous translation could not be validated by the quality "
                                 "judge — regenerate a clean, correct translation from scratch.",
                    )
                    try:
                        retry_translation, _ = run_template_translator(
                            client, model, extraction, target_language, cultural_guidance,
                            feedback_section=feedback_section, track=track)
                    except Exception as e:
                        _vlog(f"[translator retry error] {e}")
                        retry_translation = None
                if retry_translation and retry_translation.get("translated_template"):
                    translation = retry_translation
                    translated_template = translation["translated_template"]
        # If max_retries feedback rounds pass without a clean verdict, the sample is kept
        # with the best translation so far (most remaining rejects were judge
        # hallucinations or minor issues); translation_judge / translation_attempts
        # record the final verdict for downstream auditing.

    translated_extraction = {**extraction, "template": translated_template,
                             "language": target_language}

    # Stages 4-5 and the benign branch: independent branches off the same template.
    with ThreadPoolExecutor(max_workers=3) as branch_ex:
        variants_future = branch_ex.submit(
            process_variants, client, model, translated_extraction, num_variants,
            skip_generator, skip_judge2, verbose, judge_client, judge_model,
            taxonomy_supplements, cultural_guidance, track)
        paraphrases_future = branch_ex.submit(
            process_paraphrases, client, model, translated_extraction, num_paraphrases,
            num_variants, skip_generator, skip_judge2, verbose, judge_client, judge_model,
            taxonomy_supplements, cultural_guidance, track) if num_paraphrases > 0 else None
        benign_future = branch_ex.submit(
            process_benign_exactfill, client, model, translated_extraction, num_benign,
            skip_generator, skip_judge2, verbose, judge_client, judge_model,
        ) if num_benign > 0 else None
        variants = variants_future.result()
        paraphrases = paraphrases_future.result() if paraphrases_future else None
        benign_variants = benign_future.result() if benign_future else None
    prefixes = run_affix_generator(client, model, translated_template, num_prefixes,
                                   run_prefix_generator) if num_prefixes > 0 else []
    suffixes = run_affix_generator(client, model, translated_template, num_suffixes,
                                   run_suffix_generator) if num_suffixes > 0 else []

    return _build_result(
        "pass", attempt, extraction=extraction, judge1=judge1, translation=translation,
        translated_template=translated_template, translation_judge=last_trans_judge,
        translation_attempts=translation_attempts, variants=variants, paraphrases=paraphrases,
        prefixes=prefixes, suffixes=suffixes, benign_variants=benign_variants)
