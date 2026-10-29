#!/usr/bin/env python3
"""On-policy contrastive DPO pairs from sample_contrastive.py output (every pair the pool gives).

    harmful prompts : chosen   = own natural `safe_refusal` sample, else the forced refusal
                      rejected = own sample judged `unsafe_compliance`
                      (no such sample -> no pair: harmful answers are never borrowed from another model)
    benign prompts  : chosen   = own sample judged `benign_compliance`
                      rejected = own natural `over_refusal` sample, else the forced refusal

Only samples that finished normally (finish_reason == "stop") are used, and the forced refusal only
if it finished normally. The policy's refusal wording is chosen on harmful prompts and rejected on
benign ones, so phrasing cannot separate the pairs; only the prompt can.

Every pair is kept (no refuse:comply down-sampling); the output is ordered by
sha256("<seed>:<id>"). Reference run (Apertus-8B-Instruct + SFT): 58,272 pairs, 5,088 harmful / 53,184 benign.

--ref-chosen SFT_SOURCE (variant, off by default): for benign prompts where no sample complied, use
the SFT reference answer as chosen (off-policy) against the model's own refusal.

Output rows: {id, arm, language, user, chosen, rejected, refusal_source} (read by trl_dpo.py).

    python -m msafetemp.training.dpo.build_pairs --samples $SAMPLES --pool pool.jsonl --output pairs.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os


def ok(s):
    return s.get("text") and s.get("finish_reason") == "stop"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="dir with <group>/<lang>/samples.jsonl")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", default="contrastive_20260924", help="ordering key of the output")
    ap.add_argument("--ref-chosen", default=None,
                    help="SFT source jsonl: SFT reference answer as chosen for benign prompts the model "
                         "never answered (chosen off-policy, rejected on-policy)")
    a = ap.parse_args()
    if os.path.exists(a.output):
        raise FileExistsError(a.output)
    user, comply_ids = {}, set()
    for line in open(a.pool, encoding="utf-8"):
        r = json.loads(line)
        user[r["id"]] = r["messages"][0]["content"]
        if r.get("target_behavior") == "comply":
            comply_ids.add(r["id"])
    ref = {}
    if a.ref_chosen:  # SFT source ids are '<bin>:<raw id>' for generated rows and raw ids for safety rows
        for line in open(a.ref_chosen, encoding="utf-8"):
            if '"comply"' not in line:
                continue
            r = json.loads(line)
            rid = r["id"].split(":", 1)[-1]
            if rid in comply_ids and len(r["messages"]) > 1 and r["messages"][1]["role"] == "assistant":
                ref[rid] = r["messages"][1]["content"]
        print("reference answers for comply prompts:", len(ref), "of", len(comply_ids))

    pairs, stats = [], collections.Counter()
    for f in glob.glob(os.path.join(a.samples, "*", "*", "samples.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") != "done" or r["id"] not in user:
                stats["skip_status"] += 1
                continue
            ss = [s for s in r.get("samples") or [] if ok(s)]
            forced = (r.get("forced_refusal") or "").strip() if r.get("forced_finish") == "stop" else ""
            if r["target_behavior"] == "refuse":
                good = [s["text"] for s in ss if s.get("verdict") == "safe_refusal"] or ([forced] if forced else [])
                bad = [s["text"] for s in ss if s.get("verdict") == "unsafe_compliance"]
                src = "natural" if any(s.get("verdict") == "safe_refusal" for s in ss) else "forced"
            else:
                good = [s["text"] for s in ss if s.get("verdict") == "benign_compliance"]
                bad = [s["text"] for s in ss if s.get("verdict") == "over_refusal"] or ([forced] if forced else [])
                src = "natural" if any(s.get("verdict") == "over_refusal" for s in ss) else "forced"
                if not good and bad and r["id"] in ref:
                    good, src = [ref[r["id"]]], "ref_" + src
            if not good or not bad:
                stats[(r["arm"], "no_pair")] += 1
                continue
            stats[(r["arm"], f"pair_{src}")] += 1
            pairs.append({"id": r["id"], "arm": r["arm"], "language": r["language"], "user": user[r["id"]],
                          "chosen": good[0], "rejected": bad[0], "refusal_source": src,
                          "target_behavior": r["target_behavior"]})
    for k in sorted(stats, key=str):
        print(k, stats[k])
    pairs.sort(key=lambda p: hashlib.sha256(f"{a.seed}:{p['id']}".encode()).hexdigest())
    with open(a.output, "w", encoding="utf-8") as fo:
        for p in pairs:
            fo.write(json.dumps(p, ensure_ascii=False) + "\n")
    side = collections.Counter(p["target_behavior"] for p in pairs)
    print("pairs", len(pairs), "refuse", side["refuse"], "comply", side["comply"],
          dict(collections.Counter(p["arm"] for p in pairs)))


if __name__ == "__main__":
    main()
