"""Translate the English wrapper bank into the target languages, with fail-closed checks.

For every (wrapper, language) form: the payload slot is replaced by the literal sentinel
<harmful_request>, the generator translates (or on a retry repairs) the scaffold under a
strict JSON schema, and a quality model checks it independently:

  language check   candidate only (never the source), with a narrow set of preserved
                   protocol labels masked; Swiss German gets dialect notes
  meaning check    candidate vs. English source (conditions, negations, examples,
                   requested output behaviour preserved)
  static checks    exact slot multiplicity / no new template markers; no mixed-script
                   words (Latin/Greek/Cyrillic/Armenian/Georgian inside one word)

A form passes only if all three booleans are true, `uncertain` is false and no static
flag fires; otherwise the judge's short feedback drives a repair round (max_retries,
default 2; the third attempt translates fresh from the reference). Rejected and errored
forms are terminal records and are never used for assembly. No payload is generated
and no target model is called here.

    python -m msafetemp.augmentation.wrappers.translate --corpus_dir $PYRIT_TEMPLATES \\
        --languages german ukrainian ... \\
        --gen_url $GEN_URL --gen_model google/gemma-4-31B-it \\
        --quality_url $QC_URL --quality_model <multilingual quality model> \\
        --output $D/wrappers/wrapper_bank.jsonl

Output rows: {wrapper_id, language, template (with the canonical {{ prompt }} slot),
source_template_sha256, translation_status (original | pass | reject | error),
translation_provenance}. English originals are written once with status `original`.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import threading
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import read_jsonl
from msafetemp.common.llm import make_client
from msafetemp.augmentation.prompts.wrappers import (
    LANGUAGE_SYSTEM, MEANING_SYSTEM, SWISS_GERMAN_LANGUAGE_NOTE, SWISS_GERMAN_MEANING_NOTE,
    TRANSFORM_SYSTEM,
)
from msafetemp.augmentation.wrappers.corpus import CANONICAL_SLOT, fill_slot, load_jailbreak_wrappers

SLOT = "<harmful_request>"
CHECK_FIELDS = ("language_correct", "natural", "meaning_preserved", "uncertain")


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ProtocolError(RuntimeError):
    """An unscored request that must be retried, never cached as a rejection."""

    def __init__(self, stage, error_type):
        self.stage, self.error_type = stage, error_type
        super().__init__(f"{stage}: {error_type}")


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def schema_valid(value, schema):
    if not isinstance(value, dict) or set(value) != set(schema["properties"]):
        return False
    types = {"boolean": bool, "string": str, "integer": int}
    return all(type(value[k]) is types[spec["type"]] for k, spec in schema["properties"].items())


def request_json(client, model, stage, system, data, schema, *, max_tokens=2048,
                 temperature=0.0, attempts=3, retry_temperature=None, grow_token_budget=True):
    """Strict-schema JSON call (server-side json_schema response format) with bounded
    protocol retries; the token budget grows linearly per attempt."""
    last_error = "no_attempt"
    contract = ("\nRequired output JSON schema. Include EVERY required field, including uncertain "
                "when specified. Do not omit fields whose values are false.\n" + json.dumps(schema))
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system + contract + (
                    f"\nPrevious attempt failed the output protocol ({last_error}); correct the format only."
                    if attempt else "")},
                          {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
                response_format={"type": "json_schema", "json_schema": {
                    "name": "component_result", "strict": True, "schema": schema}},
                max_tokens=max_tokens * (attempt + 1) if grow_token_budget else max_tokens,
                temperature=retry_temperature if attempt and retry_temperature is not None else temperature,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason != "stop":
                last_error = "unfinished_completion"
            else:
                try:
                    value = json.loads(content)
                except (ValueError, TypeError):
                    value = None
                if schema_valid(value, schema):
                    return value
                last_error = "invalid_schema" if isinstance(value, dict) else "invalid_json"
        except Exception as exc:
            last_error = type(exc).__name__
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 4))
    raise ProtocolError(stage, last_error)


def check_passed(result):
    return (all(result.get(k) is True for k in CHECK_FIELDS[:-1])
            and result.get("uncertain") is False)


def slots_preserved(source, candidate):
    """Exact slot count and no newly introduced template markers."""
    markers = lambda s: collections.Counter(re.findall(r"<[a-zA-Z][a-zA-Z0-9_]*>", s))
    return (source.count(SLOT) > 0 and source.count(SLOT) == candidate.count(SLOT)
            and markers(source) == markers(candidate)
            and "{{" not in candidate and "{%" not in candidate)


def mixed_word_spans(text):
    """Conservative confusable-word signal: words mixing alphabetic scripts that do not
    normally mix within a word. Japanese Han/kana and Indic combining marks are excluded."""
    spans = []
    for word in re.findall(r"[^\W\d_]+", text, re.UNICODE):
        scripts = {unicodedata.name(c, "").split(" ")[0] for c in word if c.isalpha()}
        scripts &= {"LATIN", "GREEK", "CYRILLIC", "ARMENIAN", "GEORGIAN"}
        if len(scripts) > 1:
            spans.append(word)
    return sorted(set(spans))


def language_view(source, candidate):
    """Hide only a narrow set of source-preserved protocol labels from the language check
    (e.g. preserved UserQuery / ResponseFormat labels). The unmodified candidate still goes
    to the meaning check; ordinary untranslated sentences are never masked."""
    patterns = [r'\[(?:start|end) (?:input|output)[\]}]',
                r'\b(?:UserQuery|ResponseFormat)\b', re.escape(SLOT),
                r'(?m)^[A-Z][A-Z0-9_]{1,40}: *[A-Z][A-Z0-9_]{0,40}(?=\s|$)']
    literals = {match.group() for pattern in patterns
                for match in re.finditer(pattern, source, re.I if pattern == patterns[0] else 0)}
    view, protected = candidate, []
    for literal in sorted(literals, key=lambda value: (-len(value), value)):
        if literal in view:
            protected.append(literal)
            view = view.replace(literal, f'⟦{len(protected)}⟧')
    return view, protected


def check_text(client, model, source, candidate, language, *, kind):
    view, protected = language_view(source, candidate)
    language_system, meaning_system = LANGUAGE_SYSTEM, MEANING_SYSTEM
    if language == 'swiss_german':
        language_system += SWISS_GERMAN_LANGUAGE_NOTE
        meaning_system += SWISS_GERMAN_MEANING_NOTE
    linguistic = request_json(client, model, kind + "_language", language_system,
        {"candidate_text": view, "target_language": language, "kind": kind},
        object_schema({**{key: {"type": "boolean"}
                          for key in ("language_correct", "natural", "uncertain")},
                       "feedback": {"type": "string"}}))
    semantic = {"meaning_preserved": source == candidate, "uncertain": False, "feedback": ""}
    if source != candidate:
        semantic = request_json(client, model, kind + "_meaning", meaning_system,
            {"reference_text": source, "candidate_text": candidate,
             "target_language": language, "kind": kind},
            object_schema({"meaning_preserved": {"type": "boolean"},
                           "uncertain": {"type": "boolean"}, "feedback": {"type": "string"}}))
    result = {"language_correct": linguistic["language_correct"], "natural": linguistic["natural"],
              "meaning_preserved": semantic["meaning_preserved"],
              "uncertain": linguistic["uncertain"] or semantic["uncertain"],
              "feedback": "; ".join(value for value in
                                    (linguistic["feedback"], semantic["feedback"]) if value)[:300]}
    flags = []
    if kind == "wrapper" and not slots_preserved(source, candidate):
        flags.append("slot_integrity")
    if mixed_word_spans(candidate):
        flags.append("mixed_script_word")
    return {**result, "static_flags": flags, "accepted": check_passed(result) and not flags,
            "language_check": linguistic, "meaning_check": semantic,
            "protected_literal_count": len(protected)}


def translate_checked(client, model, jclient, jmodel, source, language, max_retries):
    attempts = []
    translated = source if language == "english" else None
    feedback = ""
    check = None
    for attempt in range(max_retries + 1):
        if translated is None or attempt:
            response = request_json(
                client, model, "wrapper_translation", TRANSFORM_SYSTEM,
                {"source": source, "target_language": language,
                 # A fresh final translation avoids anchoring on an already garbled repair.
                 "candidate_to_repair": None if attempt >= 2 else translated,
                 "translation_mode": "fresh_from_reference" if attempt >= 2 else "translate_or_repair",
                 "feedback": feedback},
                object_schema({"text": {"type": "string"}}),
                max_tokens=8192, temperature=0.2)
            translated = response["text"].strip()
        check = check_text(jclient, jmodel, source, translated, language, kind="wrapper")
        attempts.append({"attempt": attempt + 1, "verdict": "pass" if check["accepted"] else "reject",
                         "checks": check, "candidate_hash": digest(translated)})
        if check["accepted"]:
            return translated, {"verdict": "pass", "checks": check}, attempts
        feedback = check["feedback"] or ", ".join(check["static_flags"])
    return translated, {"verdict": "reject", "checks": check}, attempts


def original_rows(corpus_dir: str) -> list[dict]:
    rows = []
    for w in load_jailbreak_wrappers(corpus_dir):
        template = fill_slot(w["value"], CANONICAL_SLOT)
        rows.append({"wrapper_id": w["rel_path"], "language": "english", "template": template,
                     "source_template_sha256": sha256_text(template),
                     "translation_status": "original", "source_name": w["name"]})
    return rows


def translate_form(original, language, args, gen, qc):
    source = fill_slot(original["template"], SLOT)
    base = {k: v for k, v in original.items() if k not in ("template", "language", "translation_status")}
    provenance = {"models": {"generation": args.gen_model, "quality": args.quality_model},
                  "source_sentinel_sha256": sha256_text(source), "checks": {}, "attempts": []}
    template, status = None, "error"
    try:
        candidate, judgment, attempts = translate_checked(
            gen, args.gen_model, qc, args.quality_model, source, language, args.max_retries)
        intact = slots_preserved(source, candidate)
        status = ("pass" if judgment.get("verdict") == "pass"
                  and judgment["checks"].get("accepted") is True and intact else "reject")
        template = candidate.replace(SLOT, CANONICAL_SLOT)
        provenance.update(checks=judgment["checks"], attempts=attempts, slot_integrity=intact,
                          template_sha256=sha256_text(template))
    except Exception as exc:
        provenance["error"] = {"type": type(exc).__name__, "stage": getattr(exc, "stage", None),
                               "protocol_error": getattr(exc, "error_type", None)}
    return {**base, "language": language, "template": template,
            "translation_status": status, "translation_provenance": provenance}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus_dir", required=True, help="directory of PyRIT-style wrapper YAML files")
    p.add_argument("--languages", nargs="+", required=True, help="target languages (english = originals)")
    p.add_argument("--gen_url", required=True)
    p.add_argument("--gen_model", required=True)
    p.add_argument("--quality_url", required=True)
    p.add_argument("--quality_model", required=True)
    p.add_argument("--api_key", default=None)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--workers", type=int, default=256)
    p.add_argument("--output", required=True, help="wrapper_bank.jsonl (appended, resumable)")
    args = p.parse_args()

    key = resolve_api_key(args.api_key)
    gen = make_client(args.gen_url, key, args.workers, max_retries=0)
    qc = make_client(args.quality_url, key, args.workers, max_retries=0)
    originals = {r["wrapper_id"]: r for r in original_rows(args.corpus_dir)}
    print(f"{len(originals)} eligible single-slot wrappers")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["wrapper_id"], r["language"]) for r in read_jsonl(out, skip_bad=True)}
    lock = threading.Lock()
    counts = collections.Counter()
    with open(out, "a", encoding="utf-8") as fh:
        for wid, row in originals.items():
            if (wid, "english") not in done:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        items = [(wid, lang) for wid in sorted(originals) for lang in args.languages
                 if lang != "english" and (wid, lang) not in done]
        print(f"{len(items)} translations pending")
        recent = collections.deque(maxlen=32)
        source = iter(items)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            pending = {}

            def replenish():
                while len(pending) < args.workers:
                    item = next(source, None)
                    if item is None:
                        break
                    pending[ex.submit(translate_form, originals[item[0]], item[1], args, gen, qc)] = item

            replenish()
            while pending:
                finished, _ = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for fut in finished:
                    pending.pop(fut)
                    row = fut.result()
                    with lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                    counts[row["translation_status"]] += 1
                    recent.append(row["translation_status"] == "error")
                    print(f"\r{sum(counts.values())}/{len(items)} {dict(counts)}", end="", flush=True)
                    # Operational circuit breaker: 16 errors among the last 32 forms.
                    if len(recent) == 32 and sum(recent) >= 16:
                        raise SystemExit("\noperational error circuit breaker tripped; rerun to resume")
                replenish()
    print(f"\ndone: {dict(counts)}")


if __name__ == "__main__":
    main()
