#!/usr/bin/env python3
"""Build the balanced MSafeTemp safety SFT set (`train_balanced.jsonl`) from generated responses.

Inputs:
  --responses        output dir of `msafetemp.augmentation.generate_responses` (refusal design v2)
                     over the response manifest (`select_response_rows.py`); one
                     `<lang>/responses.jsonl` per language. Only refusal records are read from it.
  benign answers     either --benign_rows, an SFT-format JSONL whose `target_behavior == "comply"`
                     rows are taken unchanged (the released run used the benign answers of the
                     first response round, in this format), or --benign_responses, a
                     generate_responses output dir whose accepted benign records are converted to
                     the same row format (may be the same dir as --responses).
  --eval_dir         optional reference-model evaluation of the train split (see
                     select_response_rows.py); flags rows it answered harmfully, which the
                     balancing step takes first.

Outputs (in --output):
  train.jsonl           every accepted row (refuse : comply about 1.8 : 1)
  train_balanced.jsonl  1 : 1 per language: all benign rows plus as many refusals, split between
                        harmful and posthoc by their share, reference-model failures first
  rejected.jsonl        refusals that failed the judges or were exact duplicates
  stats.json

Row schema: id, messages (user, assistant), arm (harmful / posthoc / benign_direct /
benign_bounded), target_behavior (refuse / comply), language, original_id, group_id, branch,
refusal_shape, refusal_length, refusal_opener, refusal_design, reference_unsafe_compliance
(the reference-model flag; field name kept as in the released file).

    python -m msafetemp.training.data.build_safety_sft --responses $OUT/responses \\
        --benign_responses $OUT/responses --output $OUT/safety_sft

Model sampling is not deterministic, so a re-run gives a different set of the same design; the
released `train_balanced.jsonl` is the reference file (see docs/DATA.md).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from msafetemp.training.data.select_response_rows import reference_verdicts


def latest_records(root):
    """Latest record per id over <root>/*/responses.jsonl (a gen_failed row may be followed by its retry)."""
    recs = {}
    for p in sorted(Path(root).glob("*/responses.jsonl")):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["id"]] = r
    return recs


def benign_from_responses(root, verdicts):
    """Accepted benign answers of a generate_responses run, in the SFT row format."""
    rows = []
    recs = latest_records(root)
    for rid in sorted(recs):
        r = recs[rid]
        if r.get("kind") != "benign" or not r.get("accepted"):
            continue
        rows.append({
            "id": rid,
            "messages": [{"role": "user", "content": r["prompt"]},
                         {"role": "assistant", "content": r["response"]}],
            "arm": f"benign_{r['benign_class'] or 'direct'}", "target_behavior": "comply",
            "language": r["language"], "original_id": r["original_id"], "group_id": r["group_id"],
            "branch": r["branch"], "refusal_shape": "", "refusal_length": "",
            "reference_unsafe_compliance": verdicts.get(rid) == "unsafe_compliance",
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True, help="generate_responses output dir (v2 refusals)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--benign_rows", help="SFT-format JSONL; its comply rows are the benign answers")
    src.add_argument("--benign_responses", help="generate_responses output dir with the benign answers")
    ap.add_argument("--eval_dir", default=None, help="optional reference-model evaluation of the train split")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=111)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    verdicts = reference_verdicts(args.eval_dir)

    recs = {rid: r for rid, r in latest_records(args.responses).items()
            if r.get("kind", "refusal") == "refusal"}

    refusals, rejected, seen = [], [], set()
    for rid in sorted(recs):
        r = recs[rid]
        row = {
            "id": rid,
            "messages": [{"role": "user", "content": r["prompt"]},
                         {"role": "assistant", "content": r["response"]}],
            "arm": r["category"], "target_behavior": "refuse", "language": r["language"],
            "original_id": r["original_id"], "group_id": r["group_id"], "branch": r["branch"],
            "refusal_shape": r["shape"], "refusal_length": r["length"],
            "refusal_opener": r["opener"], "refusal_design": "v2",
            "reference_unsafe_compliance": verdicts.get(rid) == "unsafe_compliance",
        }
        if not r["accepted"]:
            rejected.append({**row, "reject_reason": r["reject_reason"], "judge": r.get("judge"),
                             "eval_harm": r.get("eval_harm")})
            continue
        key = r["response"].strip()
        if key in seen:
            rejected.append({**row, "reject_reason": "duplicate_response"})
            continue
        seen.add(key)
        refusals.append(row)

    benign = []
    if args.benign_rows:
        for line in open(args.benign_rows, encoding="utf-8"):
            r = json.loads(line)
            if r["target_behavior"] == "comply":
                benign.append({**r, "refusal_opener": None, "refusal_design": None})
    else:
        benign = [{**r, "refusal_opener": None, "refusal_design": None}
                  for r in benign_from_responses(args.benign_responses, verdicts)]

    rows = benign + refusals
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "rejected.jsonl", "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # balanced: per language, as many refusals as benign rows
    ben_lang = Counter(r["language"] for r in benign)
    by_lang_arm = defaultdict(lambda: defaultdict(list))
    for r in refusals:
        by_lang_arm[r["language"]][r["arm"]].append(r)
    picked = []
    for lang, arms in by_lang_arm.items():
        total = sum(len(v) for v in arms.values())
        want = min(ben_lang.get(lang, 0), total)
        for arm, pool in arms.items():
            k = round(want * len(pool) / total)
            rng.shuffle(pool)
            pool.sort(key=lambda r: not r["reference_unsafe_compliance"])   # stable
            picked.extend(pool[:k])
    bal = benign + picked
    rng.shuffle(bal)
    with open(out / "train_balanced.jsonl", "w", encoding="utf-8") as f:
        for r in bal:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def med_chars(rs):
        return statistics.median(len(r["messages"][1]["content"]) for r in rs) if rs else None

    stats = {
        "train_rows": len(rows), "train_balanced_rows": len(bal),
        "by_arm": dict(Counter(r["arm"] for r in rows)),
        "balanced_by_arm": dict(Counter(r["arm"] for r in bal)),
        "refusal_shapes": dict(Counter(r["refusal_shape"] for r in refusals)),
        "refusal_lengths": dict(Counter(r["refusal_length"] for r in refusals)),
        "refusal_median_chars": med_chars(refusals), "benign_median_chars": med_chars(benign),
        "refusals_generated": len(recs), "refusals_accepted": len(refusals),
        "rejected_by_reason": dict(Counter(r["reject_reason"] for r in rejected)),
        "reference_failures_in_train": sum(r["reference_unsafe_compliance"] for r in refusals),
        "reference_failures_in_balanced": sum(r["reference_unsafe_compliance"] for r in picked),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
