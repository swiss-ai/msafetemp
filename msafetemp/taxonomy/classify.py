"""LinguaSafe taxonomy classification of the English seed prompts.

    # WildGuardMix: the 2,500 canonical originals, keyed wg_NNNN (index in canonical_2500.json)
    python -m msafetemp.taxonomy.classify --track wildguard \\
        --canonical $D/wildguard/canonical_2500.json \\
        --judge_url $JUDGE_URL --judge_model openai/gpt-oss-safeguard-120b \\
        --output $D/wildguard/classification

    # HarmBench: the 200 standard behaviors, keyed hb_NNNN exactly as the dataset builder
    python -m msafetemp.taxonomy.classify --track harmbench \\
        --harmbench_raw $D/harmbench/raw --judge_url ... --output $D/harmbench/classification

Temperature 0, max_tokens 16384. Each prompt: main system prompt twice, then the clinical
fallback framing. Writes a resumable <output>/tags.jsonl and a consolidated
<output>/canonical_2500_tags.json (wildguard) or harmbench_200_tags.json (harmbench)
mapping original_id -> {labels, high_level, primary, primary_high_level, n_labels,
is_non_consensual_sexual, participants, reasoning}.

Two HarmBench behaviors were labelled manually in the released data (one unparseable
output, one abstention); this tool leaves such rows failed / unlabelled.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import format_duration, read_jsonl
from msafetemp.common.llm import (
    APICircuitBreaker, chat, get_prompt_usage_counts, make_client, parse_json_object,
    track_prompt_usage,
)
from msafetemp.taxonomy.prompts import CLASSIFY_SYSTEM, CLASSIFY_SYSTEM_CLINICAL, CLASSIFY_USER

TAG_FILES = {"wildguard": "canonical_2500_tags.json", "harmbench": "harmbench_200_tags.json"}

_MAX_TOKENS = 16384

# Subcategory -> high-level category, derived here rather than asked for, so the model
# cannot return a parent / child pair that disagrees.
HIGH_LEVEL_OF = {
    "controlled_items": "crimes_and_illegal_activities",
    "terrorism": "crimes_and_illegal_activities",
    "sex_related_crimes": "crimes_and_illegal_activities",
    "animal_related_crimes": "crimes_and_illegal_activities",
    "violent_crimes": "crimes_and_illegal_activities",
    "financial_crimes": "crimes_and_illegal_activities",
    "digital_crimes": "crimes_and_illegal_activities",
    "gender": "fairness_and_justice",
    "sexual_orientation": "fairness_and_justice",
    "age": "fairness_and_justice",
    "race": "fairness_and_justice",
    "religion": "fairness_and_justice",
    "disability": "fairness_and_justice",   # extension, not in LinguaSafe
    "erotic_content": "explicit_content",
    "violence": "explicit_content",
    "profanity": "explicit_content",
    "pii": "privacy_and_property",
    "intellectual_property": "privacy_and_property",
    "business_secrets": "privacy_and_property",
    "physical_harm": "harm_and_misuse",
    "mental_harm": "harm_and_misuse",
    "misinformation_misuse": "harm_and_misuse",
    "code_misuse": "harm_and_misuse",
    "bio_chemical_misuse": "harm_and_misuse",
}

_MAX_LABELS = 3


def load_canonical(path: str) -> list[dict]:
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))["original_prompts"]
    return [{"original_id": f"wg_{i:04d}", "original_index": i, "original_prompt": p}
            for i, p in enumerate(prompts)]


def load_harmbench(harmful_dir: str) -> list[dict]:
    """The HarmBench behaviors via the dataset builder's own load_behaviors, so the
    hb_NNNN ids and English wording are exactly the ones the dataset carries."""
    from msafetemp.augmentation.build_dataset_harmbench import find_languages, load_behaviors
    behaviors = load_behaviors(harmful_dir, find_languages(harmful_dir, None))
    return [{"original_id": f"hb_{i:04d}", "original_index": i,
             "original_prompt": behaviors[i]["behavior"]} for i in sorted(behaviors)]


def _normalize(value) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("&", "and")


def _as_text(value) -> str:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value or "")


def validate(parsed):
    """Order-preserving dedupe of the label list, unknown ids dropped. A reply is rejected
    only when NOTHING recognisable survives. An empty list is a valid abstention. GATE 0
    (non-consensual sexual) is enforced here, not trusted to the label list."""
    if not parsed:
        return None
    raw_labels = parsed.get("labels")
    if not isinstance(raw_labels, list):
        return None
    gate0 = parsed.get("is_non_consensual_sexual")
    if not isinstance(gate0, bool):
        return None
    labels = []
    for item in raw_labels:
        sub = _normalize(item)
        if sub in HIGH_LEVEL_OF and sub not in labels:
            labels.append(sub)
    if raw_labels and not labels:
        return None     # every id hallucinated: a garbled answer, not a deliberate "none"
    labels = labels[:_MAX_LABELS]
    if gate0:
        labels = ["sex_related_crimes"] + [l for l in labels
                                           if l not in ("sex_related_crimes", "erotic_content")]
        labels = labels[:_MAX_LABELS]
    high_level = []
    for sub in labels:
        if HIGH_LEVEL_OF[sub] not in high_level:
            high_level.append(HIGH_LEVEL_OF[sub])
    return {
        "labels": labels,
        "high_level": high_level,
        "primary": labels[0] if labels else "",
        "primary_high_level": high_level[0] if high_level else "",
        "n_labels": len(labels),
        "is_non_consensual_sexual": gate0,
        "participants": _as_text(parsed.get("participants")),
        "reasoning": parsed.get("reasoning", ""),
    }


def run_classifier(client, model, prompt, system=None):
    raw = chat(client, model, [
        {"role": "system", "content": system or CLASSIFY_SYSTEM},
        {"role": "user", "content": CLASSIFY_USER.format(prompt=prompt)},
    ], max_tokens=_MAX_TOKENS, temperature=0.0)
    return validate(parse_json_object(raw)), raw


def classify_with_retry(client, model, prompt):
    parsed, raw = None, ""
    for system in (None, None, CLASSIFY_SYSTEM_CLINICAL):
        try:
            parsed, raw = run_classifier(client, model, prompt, system=system)
        except APICircuitBreaker:
            raise
        except Exception as e:
            raw = str(e)
        if parsed:
            track_prompt_usage("classifier", system is CLASSIFY_SYSTEM_CLINICAL)
            return parsed, raw
    return None, raw


def process_sample(sample, client, model):
    parsed, raw = classify_with_retry(client, model, sample["original_prompt"])
    record = {**sample, "classification": parsed}
    if parsed is None:
        record["classification_failure_raw"] = raw[-2000:]
    return record


def _new_counts() -> dict:
    return {"processed": 0, "failed": 0, "gate0_fired": 0, "unclassified": 0,
            "primary": {}, "primary_high_level": {}, "label": {}, "n_labels": {}}


def _bump(d, key):
    d[key] = d.get(key, 0) + 1


def _tally(record, counts):
    counts["processed"] += 1
    c = record.get("classification")
    if not c:
        counts["failed"] += 1
        return
    if not c["labels"]:
        counts["unclassified"] += 1
    else:
        _bump(counts["primary"], c["primary"])
        _bump(counts["primary_high_level"], c["primary_high_level"])
    _bump(counts["n_labels"], str(c["n_labels"]))
    if c["is_non_consensual_sexual"]:
        counts["gate0_fired"] += 1
    for sub in c["labels"]:
        _bump(counts["label"], sub)


def _print_summary(counts, elapsed):
    total = max(counts["processed"] - counts["failed"], 1)
    print(f"\nsummary  elapsed={format_duration(elapsed)}")
    print(f"  processed {counts['processed']}  failed {counts['failed']}  "
          f"unclassified {counts['unclassified']}  gate0_fired {counts['gate0_fired']}")
    print("  labels per prompt: " + "  ".join(f"{k}:{v}" for k, v in sorted(counts["n_labels"].items())))
    for key in ("primary_high_level", "primary"):
        print(f"  {key}:")
        for name, n in sorted(counts[key].items(), key=lambda x: -x[1]):
            print(f"    {name:<32} {n:>5}  ({n / total:.1%})")


def consolidate(out_file: Path, dest: Path, source: str) -> int:
    tags = {}
    for r in read_jsonl(out_file):
        if r.get("classification"):
            tags[r["original_id"]] = {"original_index": r["original_index"], **r["classification"]}
    dest.write_text(json.dumps({"source": source, "n_tagged": len(tags),
                                "tags": dict(sorted(tags.items()))}, indent=2), encoding="utf-8")
    return len(tags)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=sorted(TAG_FILES), default="wildguard")
    ap.add_argument("--canonical", default=None, help="wildguard: canonical_2500.json")
    ap.add_argument("--harmbench_raw", default=None, help="harmbench: generate.py output (standard_<lang>/regen.jsonl)")
    ap.add_argument("--judge_url", required=True)
    ap.add_argument("--judge_model", required=True, help="released labels: openai/gpt-oss-safeguard-120b")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--n", type=int, default=None, help="first N originals only (debug)")
    ap.add_argument("--workers", type=int, default=200)
    ap.add_argument("--checkpoint_every", type=int, default=50)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.track == "wildguard":
        if not args.canonical:
            ap.error("--canonical is required for --track wildguard")
        source, samples = args.canonical, load_canonical(args.canonical)
    else:
        if not args.harmbench_raw:
            ap.error("--harmbench_raw is required for --track harmbench")
        source, samples = args.harmbench_raw, load_harmbench(args.harmbench_raw)
    if args.n:
        samples = samples[:args.n]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "tags.jsonl"
    done = {r.get("original_id") for r in read_jsonl(out_file, skip_bad=True)} if out_file.exists() else set()
    pending = [s for s in samples if s["original_id"] not in done]
    print(f"source: {source}  ({len(samples)} originals, track={args.track}); pending {len(pending)}")
    if pending:
        client = make_client(args.judge_url, resolve_api_key(args.api_key), args.workers,
                             timeout=900.0, max_retries=0)
        counts = _new_counts()
        start = time.time()
        with open(out_file, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_sample, s, client, args.judge_model) for s in pending]
            try:
                for i, fut in enumerate(as_completed(futures), 1):
                    record = fut.result()
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    _tally(record, counts)
                    if i % args.checkpoint_every == 0:
                        fh.flush()
                    print(f"\r  {i}/{len(pending)}", end="", flush=True)
            except APICircuitBreaker as e:
                print(f"\n\nABORTED: {e}")
                for f in futures:
                    f.cancel()
        counts["prompt_usage"] = get_prompt_usage_counts()
        (output_dir / "progress.json").write_text(json.dumps(counts))
        _print_summary(counts, time.time() - start)
    n = consolidate(out_file, output_dir / TAG_FILES[args.track], source)
    print(f"wrote {output_dir / TAG_FILES[args.track]} ({n} tags)")


if __name__ == "__main__":
    main()
