"""WildGuard track: refusal / ASR / over-refusal evaluation on the held-out WildGuardMix test split.

Reads the grouped views (grouped_train / grouped_test: one row per language x canonical original)
of the WildGuard-derived HF dataset and builds, per split:

    original             the untouched English WildGuardMix prompts (500 test originals)
    harmful_<language>   TR arm: translated main template + paraphrases, all fills
    posthoc_<language>   PH arm: culturally localized variants
    benign_<language>    benign minimal pairs, class == "direct" only

Output: <output>/<split>/<arm>/{eval_results.jsonl,stats.json} and <output>/<split>/summary.json.
Evaluate --splits test only; the train split supplies the training data.

Protocols (configs/evaluation/protocols.yaml):
  full                  full test split, --max_tokens 8192
  light (checkpoints)   seeded subsample --max_groups_per_arm 150 --max_variants_per_behavior 2,
                        --max_tokens 4096 (row ids equal those of a full run)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from msafetemp.evaluation._utils import format_duration
from msafetemp.evaluation.engine import (
    add_common_eval_args,
    make_eval_clients,
    mk_row,
    print_rankings,
    run_all_arms,
)
from msafetemp.evaluation.harmbench_eval import cap_groups, language_subsets, order_languages, variant_index

GROUPED = {"train": "grouped_train", "test": "grouped_test"}
ORIGINAL_ARM = "original"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", default=os.environ.get("MSAFETEMP_WG_DATASET"),
                   help="HF dataset dir with grouped_train / grouped_test (default: $MSAFETEMP_WG_DATASET)")
    p.add_argument("--splits", nargs="+", default=["test"], choices=list(GROUPED),
                   help="canonical-original splits to evaluate, each into <output>/<split>/")
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--no_original", action="store_true", help="skip the untouched-WildGuard arm")
    p.add_argument("--no_harmful", action="store_true", help="skip the harmful (TR) arms")
    p.add_argument("--no_posthoc", action="store_true", help="skip the post-hoc (PH) arms")
    p.add_argument("--no_benign", action="store_true", help="skip the benign over-refusal arms")
    p.add_argument("--main_only", action="store_true", help="only main-branch harmful variants")
    p.add_argument("--max_variants_per_behavior", type=int, default=0,
                   help="cap variants per original per branch/list (0 = all)")
    p.add_argument("--max_groups_per_arm", type=int, default=0,
                   help="cap canonical originals sampled per language arm (0 = all), seeded by --seed")
    p.add_argument("--seed", type=int, default=111)
    add_common_eval_args(p)
    args = p.parse_args(argv)
    if not args.dataset_path:
        p.error("--dataset_path (or $MSAFETEMP_WG_DATASET) is required")
    return args


def build_harmful_rows(subset, language, split, args):
    arm = f"harmful_{language}"
    cap = args.max_variants_per_behavior or None
    rows = []
    for g in subset:
        oid, cat = g["original_id"], g.get("wg_subcategory")
        for v in (g.get("harmful_main") or [])[:cap]:
            rows.append(mk_row(arm, split, language, oid, cat, "main", variant_index(v), -1, v["text"]))
        if not args.main_only:
            for v in (g.get("harmful_paraphrases") or []):
                # cap applies per paraphrase group, mirroring the HarmBench per-branch cap
                if cap is not None and variant_index(v) >= cap:
                    continue
                rows.append(mk_row(arm, split, language, oid, cat, "paraphrase",
                                   variant_index(v), v.get("paraphrase_index", 0), v["text"]))
    return arm, rows


def build_posthoc_rows(subset, language, split, args):
    arm = f"posthoc_{language}"
    cap = args.max_variants_per_behavior or None
    rows = []
    for g in subset:
        for v in (g.get("posthoc") or [])[:cap]:
            rows.append(mk_row(arm, split, language, g["original_id"], g.get("wg_subcategory"),
                               "posthoc", variant_index(v), -1, v["text"]))
    return arm, rows


def build_benign_rows(subset, language, split, args):
    """Only benign_class == 'direct' variants are scored."""
    arm = f"benign_{language}"
    cap = args.max_variants_per_behavior or None
    rows, excluded = [], 0
    for g in subset:
        direct = []
        for v in (g.get("benign") or []):
            if v.get("benign_class") == "direct":
                direct.append(v)
            else:
                excluded += 1
        for v in direct[:cap]:
            rows.append(mk_row(arm, split, language, g["original_id"], g.get("wg_subcategory"),
                               "benign_main", variant_index(v), -1, v["text"], eval_kind="benign"))
    print(f"  {arm}: {len(rows)} direct variants (excluded non-direct: {excluded})")
    return arm, rows


def build_original_rows(grouped, languages, split, args):
    """Untouched WildGuardMix prompts; original_prompt is identical across languages, so dedup by
    original_id (2000 train / 500 test)."""
    seen = {}
    keep = set(languages) if languages else None
    for oid, op, cat, lang in zip(grouped["original_id"], grouped["original_prompt"],
                                  grouped["wg_subcategory"], grouped["language"]):
        if keep is not None and lang not in keep:
            continue
        if oid not in seen and op:
            seen[oid] = (op, cat)
    items = list(seen.items())
    if args.max_groups_per_arm and args.max_groups_per_arm < len(items):
        idxs = list(range(len(items)))
        random.Random(args.seed).shuffle(idxs)
        items = [items[i] for i in sorted(idxs[:args.max_groups_per_arm])]
    rows = [mk_row(ORIGINAL_ARM, split, ORIGINAL_ARM, oid, cat, "original", 0, -1, op)
            for oid, (op, cat) in items]
    return ORIGINAL_ARM, rows


def build_arms(grouped, split, args):
    subsets = language_subsets(grouped, args.languages)
    languages = order_languages(subsets.keys())
    arms = []
    if not args.no_harmful and not args.no_original:
        arms.append(build_original_rows(grouped, args.languages, split, args))
    if not args.no_harmful:
        for language in languages:
            sub = cap_groups(subsets[language], args.max_groups_per_arm, args.seed)
            arms.append(build_harmful_rows(sub, language, split, args))
    if not args.no_posthoc:
        for language in languages:
            sub = cap_groups(subsets[language], args.max_groups_per_arm, args.seed)
            arm, rows = build_posthoc_rows(sub, language, split, args)
            if rows:
                arms.append((arm, rows))
    if not args.no_benign:
        for language in languages:
            sub = cap_groups(subsets[language], args.max_groups_per_arm, args.seed)
            arm, rows = build_benign_rows(sub, language, split, args)
            if rows:
                arms.append((arm, rows))
    return arms


def main(argv=None):
    from datasets import load_from_disk

    args = parse_args(argv)
    target_spec, judge_spec, judge_model = make_eval_clients(args)
    ds = load_from_disk(args.dataset_path)
    base_output = Path(args.output)
    start = time.time()
    for split in args.splits:
        key = GROUPED[split]
        if key not in ds:
            print(f"[{split}] WARNING: no '{key}' split in dataset, skipping")
            continue
        grouped = ds[key]
        out = base_output / split
        out.mkdir(parents=True, exist_ok=True)
        arms = build_arms(grouped, split, args)
        print(f"\n[{split}] {len(grouped)} groups, {len(arms)} arm(s)")
        for label, rows in arms:
            print(f"  {label:<26} {len(rows):>7} prompts")
        all_stats = run_all_arms(arms, target_spec, judge_spec, judge_model, args, out)
        (out / "summary.json").write_text(json.dumps(all_stats, ensure_ascii=False, indent=2))
        print_rankings(all_stats)
    print(f"\nall splits complete  elapsed={format_duration(time.time() - start)}  output={base_output}")


if __name__ == "__main__":
    main()
