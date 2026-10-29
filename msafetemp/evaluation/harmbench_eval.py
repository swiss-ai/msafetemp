"""HarmBench track: refusal / ASR / over-refusal evaluation of one target model.

Reads the grouped view (split "grouped_test") of the HarmBench-derived HF dataset: one row per
(behaviour, language) with the lists harmful_main, harmful_paraphrases, posthoc and benign, plus
the English source `behavior`. Builds one arm per augmentation and language:

    original_standard            the untouched English HarmBench behaviours (200)
    standard_<language>          TR arm: translated main template + paraphrases, all fills
    posthoc_standard_<language>  PH arm: culturally localized variants        (--posthoc)
    benign_standard_<language>   benign minimal pairs, class == "direct" only (--benign)

and runs every arm through the shared engine (engine.py). HarmBench is evaluation only, so there
is a single split and output is flat: <output>/<arm>/{eval_results.jsonl,stats.json}, plus
<output>/summary.json.

Protocols (configs/evaluation/protocols.yaml):
  full                --posthoc --benign --max_tokens 8192
  light (checkpoints) --benign --max_tokens 4096 ; PH arm separately with --ph_only --no_original
Aggregate with msafetemp.evaluation.aggregate.
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

CONFIG = "standard"          # the only HarmBench functional category in the data
GROUPED_SPLIT = "grouped_test"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", default=os.environ.get("MSAFETEMP_HB_DATASET"),
                   help="HF dataset dir (load_from_disk) with a grouped_test split "
                        "(default: $MSAFETEMP_HB_DATASET)")
    p.add_argument("--languages", nargs="+", default=None,
                   help="restrict to these languages (default: all). 'english' runs first.")
    p.add_argument("--no_original", action="store_true",
                   help="skip the untouched-HarmBench baseline arm (original_standard)")
    p.add_argument("--no_harmful", action="store_true",
                   help="skip the harmful arms (also drops post-hoc arms); with --benign this is "
                        "an over-refusal-only run")
    p.add_argument("--posthoc", action="store_true",
                   help="also evaluate the post-hoc cultural arms (posthoc_standard_<language>)")
    p.add_argument("--ph_only", action="store_true",
                   help="evaluate ONLY the post-hoc arms (implies --posthoc; use with --no_original "
                        "and without --benign). Used for the training checkpoints, whose TR and "
                        "benign arms were run first.")
    p.add_argument("--benign", action="store_true",
                   help="also evaluate the benign over-refusal arms (benign_standard_<language>). Only "
                        "benign_class == 'direct' variants are scored; 'bounded' variants are "
                        "dual-use (refusing them is defensible) and are never evaluated.")
    p.add_argument("--main_only", action="store_true",
                   help="evaluate only main-branch variants (skip paraphrases)")
    p.add_argument("--max_variants_per_behavior", type=int, default=0,
                   help="cap variants per behaviour per branch/list (0 = all)")
    p.add_argument("--max_groups_per_arm", type=int, default=0,
                   help="cap behaviours sampled per language arm (0 = all), seeded by --seed")
    p.add_argument("--seed", type=int, default=111)
    add_common_eval_args(p)
    args = p.parse_args(argv)
    if not args.dataset_path:
        p.error("--dataset_path (or $MSAFETEMP_HB_DATASET) is required")
    if args.ph_only:
        args.posthoc = True
        if args.no_harmful:
            p.error("--ph_only cannot be combined with --no_harmful")
    return args


def order_languages(langs):
    langs = set(langs)
    ordered = ["english"] if "english" in langs else []
    ordered.extend(sorted(lang for lang in langs if lang != "english"))
    return ordered


def language_subsets(grouped, languages):
    """language -> grouped subset (Arrow-backed .select)."""
    lang_col = grouped["language"]
    present = order_languages(set(lang_col))
    if languages:
        present = [lang for lang in present if lang in languages]
    subsets = {}
    for language in present:
        idxs = [i for i, lang in enumerate(lang_col) if lang == language]
        if idxs:
            subsets[language] = grouped.select(idxs)
    return subsets


def cap_groups(subset, n, seed):
    """Deterministically sample up to n group rows from a language subset."""
    if not n or n >= len(subset):
        return subset
    idxs = list(range(len(subset)))
    random.Random(seed).shuffle(idxs)
    return subset.select(sorted(idxs[:n]))


def variant_index(item, default=0):
    v = item.get("variant_index", default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _harmful_items(group, args):
    """(branch, paraphrase_index, variant_index, text) for one behaviour's harmful rows."""
    cap = args.max_variants_per_behavior or None
    items = [("main", -1, variant_index(v), v["text"]) for v in (group.get("harmful_main") or [])[:cap]]
    if not args.main_only:
        for v in (group.get("harmful_paraphrases") or []):
            if cap is not None and variant_index(v) >= cap:
                continue
            items.append(("paraphrase", v.get("paraphrase_index", 0), variant_index(v), v["text"]))
    return items


def build_harmful_rows(subset, language, args):
    arm = f"{CONFIG}_{language}"
    rows = []
    for g in subset:
        for branch, pi, vi, text in _harmful_items(g, args):
            rows.append(mk_row(arm, CONFIG, language, g["hb_id"], g.get("hb_category"),
                               branch, vi, pi, text))
    return arm, rows


def build_posthoc_rows(subset, language, args):
    arm = f"posthoc_{CONFIG}_{language}"
    cap = args.max_variants_per_behavior or None
    rows = []
    for g in subset:
        for v in (g.get("posthoc") or [])[:cap]:
            rows.append(mk_row(arm, CONFIG, language, g["hb_id"], g.get("hb_category"),
                               "posthoc", variant_index(v), -1, v["text"]))
    return arm, rows


def build_benign_rows(subset, language, args):
    """Benign over-refusal arm: only benign_class == 'direct' variants."""
    arm = f"benign_{CONFIG}_{language}"
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
            rows.append(mk_row(arm, CONFIG, language, g["hb_id"], g.get("hb_category"),
                               "benign_main", variant_index(v), -1, v["text"], eval_kind="benign"))
    print(f"  {arm}: {len(rows)} direct variants (excluded non-direct: {excluded})")
    return arm, rows


def build_original_rows(grouped, args):
    """Untouched HarmBench behaviours, taken from the dataset's own `behavior` column so the
    baseline covers exactly the behaviours the augmented arms were built from."""
    arm = f"original_{CONFIG}"
    keep = set(args.languages) if args.languages else None
    seen = {}
    for hb_id, behavior, cat, lang in zip(grouped["hb_id"], grouped["behavior"],
                                          grouped["hb_category"], grouped["language"]):
        if keep is not None and lang not in keep:
            continue
        if hb_id not in seen and behavior:
            seen[hb_id] = (behavior, cat)
    items = sorted(seen.items())
    if args.max_groups_per_arm and args.max_groups_per_arm < len(items):
        idxs = list(range(len(items)))
        random.Random(args.seed).shuffle(idxs)
        items = [items[i] for i in sorted(idxs[:args.max_groups_per_arm])]
    rows = [mk_row(arm, CONFIG, "original", hb_id, cat, "original", 0, -1, behavior)
            for hb_id, (behavior, cat) in items]
    return arm, rows


def build_arms(grouped, args):
    """Ordered (arm_label, rows): original, TR arms, post-hoc arms, benign arms."""
    subsets = language_subsets(grouped, args.languages)
    languages = order_languages(subsets.keys())
    arms = []
    if not args.no_harmful:
        if not args.no_original:
            arms.append(build_original_rows(grouped, args))
        if not args.ph_only:
            for language in languages:
                sub = cap_groups(subsets[language], args.max_groups_per_arm, args.seed)
                arms.append(build_harmful_rows(sub, language, args))
        if args.posthoc:
            for language in languages:
                sub = cap_groups(subsets[language], args.max_groups_per_arm, args.seed)
                arm, rows = build_posthoc_rows(sub, language, args)
                if rows:
                    arms.append((arm, rows))
    if args.benign:
        for language in languages:
            sub = cap_groups(subsets[language], args.max_groups_per_arm, args.seed)
            arm, rows = build_benign_rows(sub, language, args)
            if rows:
                arms.append((arm, rows))
    return arms


def main(argv=None):
    from datasets import load_from_disk

    args = parse_args(argv)
    target_spec, judge_spec, judge_model = make_eval_clients(args)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(args.dataset_path)
    if GROUPED_SPLIT not in ds:
        raise SystemExit(f"[eval] dataset has no '{GROUPED_SPLIT}' split (found {list(ds.keys())})")
    grouped = ds[GROUPED_SPLIT]
    print(f"groups={len(grouped)}  main_only={args.main_only}  harmful={not args.no_harmful}  "
          f"posthoc={args.posthoc}  ph_only={args.ph_only}  benign={args.benign}")
    arms = build_arms(grouped, args)
    if not arms:
        raise SystemExit("[eval] no arms built")
    print(f"\n{len(arms)} arm(s):")
    for label, rows in arms:
        print(f"  {label:<34} {len(rows):>7} prompts")

    start = time.time()
    all_stats = run_all_arms(arms, target_spec, judge_spec, judge_model, args, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(all_stats, ensure_ascii=False, indent=2))
    print_rankings(all_stats)
    print(f"\nall arms complete  elapsed={format_duration(time.time() - start)}  output={output_dir}")


if __name__ == "__main__":
    main()
