"""Exact-fill benign (OR) minimal pairs over already-generated WildGuard templates.

Re-runs only the exact-fill benign branch (multilingual.process_benign_exactfill) against
each language's validated translated templates from generate.py, restricted to the
canonical 2,500 seeds, so benign, post-hoc and evaluation cover the same source prompts.
Five candidates per template; the judge classes each as
  direct          fully helpful answer is safe and non-transferable (eval + train)
  bounded         legitimate reading, but a full answer transfers harm (train only)
  toxic           no safe answer exists (rejected)
  reject_quality  incoherent / wrong language / frame broken (rejected)

    python -m msafetemp.augmentation.benign \\
        --raw_dir $MSAFETEMP_DATA_DIR/wildguard/raw \\
        --canonical $MSAFETEMP_DATA_DIR/wildguard/canonical_2500.json \\
        --target_url $GEN_URL --target_model google/gemma-4-31B-it \\
        --judge_url $JUDGE_URL --judge_model openai/gpt-oss-120b \\
        --output $MSAFETEMP_DATA_DIR/wildguard/benign

Writes <output>/<language>/benign.jsonl (+ progress.json). The HarmBench track's benign
pairs are produced by generate.py itself (--num_variants 0 --paraphrasing 0 --benign 5).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import read_jsonl
from msafetemp.common.llm import make_client
from msafetemp.augmentation._pool import run_language_pool
from msafetemp.augmentation.multilingual import process_benign_exactfill
from msafetemp.augmentation.select_canonical import load_canonical

_KNOWN_BENIGN_CRITERIA = {"TOXIC", "INCOHERENT", "LANGUAGE_INCONSISTENT", "FRAME_CHANGED", "UNJUDGED"}


def load_samples(raw_dir: str, language: str, n: int | None = None, seed: int = 0,
                 canonical: set[str] | None = None) -> list[dict]:
    """Passing, templateable samples of one language, restricted to the canonical seeds.
    n=None returns all in file order; otherwise a deterministic reservoir sample."""
    path = os.path.join(raw_dir, language, "regen.jsonl")
    rng = random.Random(seed)
    samples, reservoir, seen = [], [], 0
    for r in read_jsonl(path):
        if r.get("status") != "pass":
            continue
        if canonical is not None and r.get("original_prompt") not in canonical:
            continue
        slots = (r.get("extraction") or {}).get("slots") or []
        translated_template = r.get("translated_template")
        if not slots or not translated_template:
            continue
        record = {"language": language, "original_prompt": r.get("original_prompt"),
                  "template": translated_template, "slots": slots,
                  "wg_subcategory": r.get("wg_subcategory")}
        if n is None:
            samples.append(record)
            continue
        seen += 1
        if len(reservoir) < n:
            reservoir.append(record)
        else:
            j = rng.randint(0, seen - 1)
            if j < n:
                reservoir[j] = record
    return samples if n is None else reservoir


def sample_key(rec: dict):
    return (rec.get("original_prompt"), rec.get("template"))


def _new_counts() -> dict:
    return {"processed": 0, "benign_variants_total": 0, "benign_variants_possible": 0,
            "benign_rejected": 0, "benign_direct": 0, "benign_bounded": 0,
            "benign_failed_criteria": {}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", required=True, help="generate.py --track wildguard output dir")
    p.add_argument("--canonical", default=None, help="canonical_2500.json (omit to process all passing)")
    p.add_argument("--languages", nargs="+", default=None,
                   help="default: every language dir with a regen.jsonl")
    p.add_argument("--n", type=int, default=None, help="cap samples per language (reservoir)")
    p.add_argument("--benign", type=int, default=5, help="benign candidates per template")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument("--target_url", required=True)
    p.add_argument("--target_model", required=True)
    p.add_argument("--judge_url", required=True)
    p.add_argument("--judge_model", required=True)
    p.add_argument("--api_key", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    languages = args.languages or sorted(
        d for d in os.listdir(args.raw_dir) if os.path.exists(os.path.join(args.raw_dir, d, "regen.jsonl")))
    canonical = set(load_canonical(args.canonical)) if args.canonical else None
    print(f"canonical filter: {len(canonical) if canonical else 'DISABLED'}")
    key = resolve_api_key(args.api_key)
    client = make_client(args.target_url, key, args.workers)
    judge_client = make_client(args.judge_url, key, args.workers)

    def process(language, sample):
        extraction = {"template": sample["template"], "slots": sample["slots"], "language": language}
        benign = process_benign_exactfill(client, args.target_model, extraction, args.benign,
                                          judge_client=judge_client, judge_model=args.judge_model)
        return {
            **sample,
            "benign_variants": [{"text": v, "benign_judge": bj} for v, bj in (benign or [])
                                if bj is None or bj.get("verdict") == "pass"],
            "rejected_benign_variants": [{"text": v, "benign_judge": bj} for v, bj in (benign or [])
                                         if bj is not None and bj.get("verdict") != "pass"],
        }

    def tally(rec, counts):
        counts["benign_variants_total"] += len(rec["benign_variants"])
        counts["benign_variants_possible"] += args.benign
        counts["benign_rejected"] += len(rec["rejected_benign_variants"])
        for v in rec["benign_variants"]:
            cls = (v["benign_judge"] or {}).get("class")
            if cls in ("direct", "bounded"):
                counts[f"benign_{cls}"] += 1
        for v in rec["rejected_benign_variants"]:
            for crit in (v["benign_judge"] or {}).get("failed_criteria", []):
                if crit in _KNOWN_BENIGN_CRITERIA:
                    fc = counts["benign_failed_criteria"]
                    fc[crit] = fc.get(crit, 0) + 1

    def summary(language, counts, elapsed):
        poss = counts.get("benign_variants_possible", 0)
        yld = f"{counts['benign_variants_total'] / poss:.1%}" if poss else "-"
        print(f"[{language}] processed={counts['processed']}  accepted={counts['benign_variants_total']} "
              f"({yld})  direct={counts['benign_direct']}  bounded={counts['benign_bounded']}  "
              f"rejected={counts['benign_rejected']}  {counts['benign_failed_criteria']}")

    run_language_pool(
        languages, lambda lang: load_samples(args.raw_dir, lang, args.n, args.seed, canonical),
        sample_key, process, Path(args.output), "benign.jsonl", args.workers,
        _new_counts, tally, summary, args.checkpoint_every)


if __name__ == "__main__":
    main()
