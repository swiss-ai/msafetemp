"""Fix the canonical set of WildGuard seed prompts.

Every downstream stage (benign pairs, post-hoc localization, taxonomy, dataset build,
evaluation) is restricted to ONE fixed subset of original prompts: the first N
status == "pass" originals of the reference language's regen.jsonl, in file order. The
generation stream is a seeded WildGuardMix shuffle, so "first N passing" is already a
random subset. status == "pass" guarantees the English extractor produced a templateable
prompt that cleared the validity judge. Join key everywhere is `original_prompt`; ids are
wg_NNNN = position in this list.

    python -m msafetemp.augmentation.select_canonical \\
        --raw_dir $MSAFETEMP_DATA_DIR/wildguard/raw --out $MSAFETEMP_DATA_DIR/wildguard/canonical_2500.json
"""

from __future__ import annotations

import argparse
import json
import os

from msafetemp.common.io import read_jsonl


def collect_passing_originals(regen_path: str, limit: int) -> list[str]:
    originals: list[str] = []
    seen: set[str] = set()
    for r in read_jsonl(regen_path, skip_bad=True):
        if r.get("status") != "pass":
            continue
        op = r.get("original_prompt")
        if not op or op in seen:
            continue
        seen.add(op)
        originals.append(op)
        if len(originals) >= limit:
            break
    return originals


def availability(regen_path: str, canonical: set[str]) -> int:
    if not os.path.exists(regen_path):
        return -1
    return sum(1 for r in read_jsonl(regen_path, skip_bad=True)
               if r.get("status") == "pass" and r.get("original_prompt") in canonical)


def load_canonical(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["original_prompts"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", required=True, help="generate.py --track wildguard output dir")
    p.add_argument("--reference", default="english", help="language whose regen.jsonl defines the order")
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--check", nargs="*", default=[], help="languages to report availability for")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ref_path = os.path.join(args.raw_dir, args.reference, "regen.jsonl")
    if not os.path.exists(ref_path):
        raise SystemExit(f"reference regen.jsonl not found: {ref_path}")
    originals = collect_passing_originals(ref_path, args.n)
    print(f"reference={args.reference}  collected={len(originals)} (target {args.n})")
    canon = set(originals)
    for lang in args.check:
        n = availability(os.path.join(args.raw_dir, lang, "regen.jsonl"), canon)
        print(f"  {lang:14s} " + ("regen.jsonl missing" if n < 0 else f"{n}/{len(originals)} present and passing"))

    payload = {
        "reference": args.reference,
        "definition": f"first {args.n} status==pass originals in {args.reference}/regen.jsonl, file order",
        "count": len(originals),
        "original_prompts": originals,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=0)
    print(f"wrote {len(originals)} originals -> {args.out}")


if __name__ == "__main__":
    main()
