#!/usr/bin/env python3
"""Prompt pool for on-policy contrastive DPO sampling (train-source prompts only, 126,458 in the reference pool).

  arm                 target  prompts
  v2_harmful          refuse  MSafeTemp safety rows: 15,000 refuse rows (harmful + posthoc arms),
  v2_posthoc          refuse    lowest sha256("cpool:<id>") first
  v2_benign_direct    comply  MSafeTemp safety rows: 15,000 comply rows (benign arms), same ranking
  v2_benign_bounded   comply
  vanilla_harmful     refuse  all English short WildGuardMix-train prompts
  vanilla_benign      comply
  wrapped_harmful     refuse  12,000 + 12,000 jailbreak-wrapped train-family prompts (same ranking)
  wrapped_benign      comply
  vanillaml_harmful   refuse  all translated short prompts
  vanillaml_benign    comply
  short_contrast_en   comply  short benign contrast requests, English and translated, with the SFT
  short_contrast_ml   comply    bin's selection (sensitive twins + innocuous SWAP / IDIOM / PROTECT)

Rows: {id, messages: [{role: user, content}], arm, language, target_behavior}. The safety arms
carry the `v2_` prefix of the safety SFT set's refusal design.

    python -m msafetemp.training.dpo.build_pool --safety-sft train_balanced.jsonl \
        --vanilla $D/wg_vanilla_hf --vanilla-ml $D/wg_vanilla_ml_hf \
        --wrapped $D/wrapped_train_1k_hf --wrap-train-ids train_ids.txt \
        --short-contrast $D/short_contrast_en_hf $D/short_contrast_ml_hf --output pool.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from msafetemp.training._utils import refuse_overwrite

KEEP_INNOCUOUS = {"SWAP", "IDIOM", "PROTECT"}
N_SAFETY_PER_SIDE = 15000
N_WRAPPED_PER_SIDE = 12000


def rank(s: str) -> str:
    return hashlib.sha256(f"cpool:{s}".encode()).hexdigest()


def take(xs, n):
    return sorted(xs, key=lambda r: rank(r["id"]))[:n]


def prompt_rows(records, arm_prefix, n=None):
    h = [r for r in records if r["category"] != "benign"]
    b = [r for r in records if r["category"] == "benign"]
    if n:
        h, b = take(h, n), take(b, n)
    return [{"id": r["id"], "messages": [{"role": "user", "content": r["text"]}],
             "arm": f"{arm_prefix}_{'harmful' if tb == 'refuse' else 'benign'}", "language": r["language"],
             "target_behavior": tb}
            for r, tb in [(x, "refuse") for x in h] + [(x, "comply") for x in b]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety-sft", required=True,
                    help="balanced safety SFT rows (fields id, messages, arm, language, target_behavior)")
    ap.add_argument("--vanilla", required=True, help="English short prompt dataset (short_prompts.py vanilla)")
    ap.add_argument("--vanilla-ml", required=True, help="translated short prompt dataset")
    ap.add_argument("--wrapped", required=True, help="wrapped 1k prompt dataset (wrapped_prompts.py wrap1k)")
    ap.add_argument("--wrap-train-ids", required=True)
    ap.add_argument("--short-contrast", nargs=2, required=True, metavar=("EN_DIR", "ML_DIR"))
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    from datasets import load_from_disk

    out = refuse_overwrite(a.output)
    rows = []
    v2 = [json.loads(line) for line in open(a.safety_sft, encoding="utf-8")]
    ref = [r for r in v2 if r["target_behavior"] == "refuse"]
    com = [r for r in v2 if r["target_behavior"] == "comply"]
    for r in take(ref, N_SAFETY_PER_SIDE) + take(com, N_SAFETY_PER_SIDE):
        rows.append({"id": r["id"], "messages": [r["messages"][0]], "arm": "v2_" + r["arm"],
                     "language": r["language"], "target_behavior": r["target_behavior"]})
    rows += prompt_rows(load_from_disk(a.vanilla)["train"].to_list(), "vanilla")
    keep = set(Path(a.wrap_train_ids).read_text().split())
    wrapped = [r for r in load_from_disk(a.wrapped)["train"].to_list() if r["id"] in keep]
    rows += prompt_rows(wrapped, "wrapped", N_WRAPPED_PER_SIDE)
    for r in load_from_disk(a.vanilla_ml)["train"].to_list():  # dataset order
        tb = "comply" if r["category"] == "benign" else "refuse"
        rows.append({"id": r["id"], "messages": [{"role": "user", "content": r["text"]}],
                     "arm": f"vanillaml_{'benign' if tb == 'comply' else 'harmful'}", "language": r["language"],
                     "target_behavior": tb})
    for d, arm in zip(a.short_contrast, ("short_contrast_en", "short_contrast_ml")):
        for r in load_from_disk(d)["train"].to_list():
            if not r["judge_sensitive"] and r["contrast_type"] not in KEEP_INNOCUOUS:
                continue
            rows.append({"id": r["id"], "messages": [{"role": "user", "content": r["text"]}], "arm": arm,
                         "language": r["language"], "target_behavior": "comply"})
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(out, len(rows), dict(Counter(r["arm"] for r in rows)))


if __name__ == "__main__":
    main()
