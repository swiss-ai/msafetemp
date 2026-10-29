#!/usr/bin/env python3
"""General instruction data for the SFT mix: Dolci-Instruct-SFT (no-tools split), safety-cleaned.

Two subcommands:

  extract : every single-turn user/assistant pair of allenai/Dolci-Instruct-SFT-No-Tools
            (ODC-BY, already tool-free), in the SFT row format. Rows longer than one exchange
            are skipped rather than truncated (a truncated context would change what the
            assistant turn answers). The reference mix takes the full set (--target above the
            row count); rows are ordered by a seeded hash so a smaller --target is a
            reproducible uniform sample.

  clean   : drop Dolci's own safety data so it cannot teach a second refusal style:
              * the synthetic WildGuardMix-train subset (id contains 'wildguardmixtrain'; it
                also overlaps the WildGuardMix source of our safety rows),
              * any row whose assistant reply opens with a decline ("I'm sorry", "I can't",
                "As an AI", ...).
            Left in, Dolci's short "I'm sorry, but I can't assist with that." dominates the
            English refusals of the mix and drives over-refusal.

    python -m msafetemp.training.data.dolci_general extract \
        --dolci-dir $HF_DATASETS_DIR/Dolci-Instruct-SFT-No-Tools/data --output dolci_general_full.jsonl
    python -m msafetemp.training.data.dolci_general clean \
        --input dolci_general_full.jsonl --output dolci_general_clean.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
from pathlib import Path

from msafetemp.training._utils import refuse_overwrite

DECLINE_OPENER = re.compile(r"^\s*(i'?m sorry|i am sorry|sorry,|i can'?t|i cannot|i won'?t|i apologi[sz]e|as an ai)",
                            re.I)


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def extract(args) -> None:
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(args.dolci_dir) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {args.dolci_dir}")
    out = refuse_overwrite(args.output)
    pool, skipped = [], collections.Counter()
    for path in files:
        for row in pq.read_table(path, columns=["id", "messages", "source"]).to_pylist():
            messages = row["messages"]
            if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                skipped["not_single_turn"] += 1
                continue
            user = (messages[0]["content"] or "").strip()
            assistant = (messages[1]["content"] or "").strip()
            if not user or not assistant:
                skipped["empty"] += 1
                continue
            # Dolci `id` is a per-source row number and collides across sources, so the key
            # includes the content or distinct rows would be lost.
            content_key = hashlib.sha256((user + "\x00" + assistant).encode()).hexdigest()[:16]
            pool.append((str(row["id"]) + "-" + content_key, row.get("source"), user, assistant))
    print(f"eligible single-turn rows: {len(pool)} | skipped {dict(skipped)}", flush=True)
    pool.sort(key=lambda r: stable_rank(str(r[0]), args.seed))
    chosen = pool[:args.target]
    per_source = collections.Counter()
    with out.open("w", encoding="utf-8") as handle:
        for row_id, source, user, assistant in chosen:
            handle.write(json.dumps({
                "id": f"dolci:{row_id}",
                "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                "language": args.language,
                "family_id": f"dolci:{row_id}",
                "response_mode": "general",
                "training_role": "general_comply",
                "arm": "general", "target_behavior": "comply",
                "dataset_source": source,
            }, ensure_ascii=False) + "\n")
            per_source[source] += 1
    manifest = {"dataset": "allenai/Dolci-Instruct-SFT-No-Tools", "license": "ODC-BY", "eligible": len(pool),
                "written": len(chosen), "seed": args.seed, "skipped": dict(skipped),
                "per_source": dict(per_source.most_common(15))}
    out.with_name(out.stem + "_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_source"}, indent=1))


def clean(args) -> None:
    out = refuse_overwrite(args.output)
    counts = collections.Counter()
    with open(args.input, encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if "wildguardmixtrain" in row["id"]:
                counts["drop_wildguardmix_subset"] += 1
                continue
            if DECLINE_OPENER.match(row["messages"][1]["content"]):
                counts["drop_decline_opener"] += 1
                continue
            counts["kept"] += 1
            dst.write(line)
    print(dict(counts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dolci-dir", required=True, help="directory with the dataset's *.parquet files")
    e.add_argument("--output", required=True)
    e.add_argument("--target", type=int, default=5_000_000, help="rows to keep (default: all)")
    e.add_argument("--seed", type=int, default=20260922)
    e.add_argument("--language", default="general")
    e.set_defaults(func=extract)
    c = sub.add_parser("clean")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.set_defaults(func=clean)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
