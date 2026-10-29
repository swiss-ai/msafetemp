#!/usr/bin/env python3
"""Jailbreak-wrapped (PyRIT-style wrapper bank) training prompts -> prompt datasets for the response generator.

The wrapped rows themselves (a WildGuard-train payload literally inserted into a translated wrapper
from the frozen wrapper bank, `{subset, language, split, wrapper_id, prompt, counterpart_prompt,
source_group_id, original_id, ...}` per line) are assembled by the augmentation stage's wrapper
tooling. This module turns them into training prompt datasets. Only TRAIN-source payloads are used.

Subcommands:

  split     : split a wrapped train set (<dir>/{harmful,benign}/rows.jsonl) by SOURCE FAMILY
              (`source_group_id`) into training ids and a held-out wrapped test set
              (<out>/train_ids.txt, <out>/test_{harmful,benign}.jsonl). The reference split uses a
              32% test share. The released `train_ids.txt` is authoritative for the reference mix;
              this subcommand applies the same procedure to new data but is not guaranteed to draw
              the identical id list.

  wrap1k    : <dir>/{harmful,benign}/rows.jsonl (1,000 wrapped prompts per language and subset,
              5 to 10 wrappers per payload) -> dataset of the rows whose id is in --train-ids.
              Row ids are `wrap_<subset>_<first 16 hex of the row id>`.

  judged    : an earlier wrapped set that went through the dual wrapper judges (quality + safety).
              Acceptance is recomputed from the stored checks, ignoring `category_match`:
                quality.accepted and
                harmful/posthoc: unsafe_request and payload_preserved and not uncertain
                benign:          not unsafe_request and payload_preserved and not uncertain
              Rows whose payload comes from the WildGuard TEST split are dropped.
              Each --set is `ROWS_JSONL:JUDGMENTS_JSONL`.

    python -m msafetemp.training.data.wrapped_prompts wrap1k --rows-dir $W/wrapped_train \
        --train-ids $W/split/train_ids.txt --out $D/wrapped_train_1k_hf
    python -m msafetemp.training.data.wrapped_prompts judged --out $D/pyrit_wrapped_train_hf \
        --set $J/english_harmful/rows.jsonl:$J/judge/english_harmful_judgments.jsonl ...
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path


def _rows(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def wrap_id(subset: str, row: dict) -> str:
    return f"wrap_{subset}_{row['id'][:16]}"


def split(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_ids = []
    for sub in ("harmful", "benign"):
        n_train = n_test = 0
        with open(out / f"test_{sub}.jsonl", "w", encoding="utf-8") as test:
            for r in _rows(Path(args.rows_dir) / sub / "rows.jsonl"):
                key = hashlib.sha256(f"{args.seed}:{r['source_group_id']}".encode()).hexdigest()
                if int(key[:8], 16) / 16 ** 8 < args.test_fraction:
                    test.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_test += 1
                else:
                    train_ids.append(wrap_id(sub, r))
                    n_train += 1
        print(sub, "train", n_train, "test", n_test)
    (out / "train_ids.txt").write_text("\n".join(train_ids) + "\n")


def wrap1k(args) -> None:
    from datasets import Dataset, DatasetDict
    keep = set(Path(args.train_ids).read_text().split()) if args.train_ids else None
    rows, dropped = [], 0
    for sub in ("harmful", "benign"):
        for r in _rows(Path(args.rows_dir) / sub / "rows.jsonl"):
            assert r["split"] == "train", "wrapped training rows must use TRAIN-source payloads"
            rid = wrap_id(sub, r)
            if keep is not None and rid not in keep:
                dropped += 1
                continue
            rows.append({"id": rid, "group_id": r.get("source_group_id") or r["id"],
                         "original_id": r.get("original_id") or "", "language": r["language"],
                         "category": "benign" if sub == "benign" else "harmful",
                         "benign_class": "direct" if sub == "benign" else None, "text": r["prompt"],
                         "original_prompt": r.get("counterpart_prompt") or "", "template": r.get("wrapper_id") or "",
                         "split": "train", "branch": "pyrit_wrapped", "ls_primary": r.get("ls_primary") or "",
                         "ls_high_level": ""})
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(args.out)
    print(len(rows), "rows; held-out test-family rows dropped:", dropped)


def judged(args) -> None:
    from datasets import Dataset, DatasetDict
    lit = lambda v: v if isinstance(v, dict) else ast.literal_eval(v)  # noqa: E731
    out, stats = [], collections.Counter()
    for spec in args.set:
        rows_path, judg_path = spec.split(":", 1)
        judg = {j["id"]: j for j in _rows(judg_path)}
        for r in _rows(rows_path):
            sub = r["subset"]
            if r.get("split") != "train":
                stats[(sub, "drop_test_source")] += 1
                continue
            j = judg.get(r["id"])
            if not j:
                stats[(sub, "no_judgment")] += 1
                continue
            q, sf = lit(j["quality"]), lit(j["safety"])
            qc = sf.get("checks") or {}
            ok_q = q.get("accepted") is True
            if sub == "benign":
                ok_s = qc.get("unsafe_request") is False and qc.get("payload_preserved") and not qc.get("uncertain")
            else:
                ok_s = qc.get("unsafe_request") is True and qc.get("payload_preserved") and not qc.get("uncertain")
            if not (ok_q and ok_s):
                stats[(sub, "rejected")] += 1
                continue
            stats[(sub, "accepted")] += 1
            cat = "benign" if sub == "benign" else sub
            out.append({"id": f"pyrit_{r['id'][:16]}", "group_id": r.get("source_group_id") or r["id"],
                        "original_id": r.get("original_id") or "", "language": r["language"], "category": cat,
                        "benign_class": "direct" if cat == "benign" else None, "text": r["prompt"],
                        "original_prompt": r.get("counterpart_prompt") or "", "template": r.get("wrapper_id") or "",
                        "split": "train", "branch": "pyrit_wrapped", "ls_primary": r.get("ls_primary") or "",
                        "ls_high_level": "", "wrapper_id": r.get("wrapper_id") or ""})
    DatasetDict({"train": Dataset.from_list(out)}).save_to_disk(args.out)
    for k in sorted(stats, key=str):
        print(k, stats[k])
    print("total", len(out), collections.Counter(r["category"] for r in out),
          "languages", len({r["language"] for r in out}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--rows-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--test-fraction", type=float, default=0.32)
    s.add_argument("--seed", default="wrapped-train-split")
    s.set_defaults(func=split)
    w = sub.add_parser("wrap1k")
    w.add_argument("--rows-dir", required=True)
    w.add_argument("--train-ids", required=True)
    w.add_argument("--out", required=True)
    w.set_defaults(func=wrap1k)
    j = sub.add_parser("judged")
    j.add_argument("--set", action="append", required=True, help="ROWS_JSONL:JUDGMENTS_JSONL (repeatable)")
    j.add_argument("--out", required=True)
    j.set_defaults(func=judged)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
