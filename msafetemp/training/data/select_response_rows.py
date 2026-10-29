#!/usr/bin/env python3
"""Choose which WildGuard-track training prompts get an SFT response (the response manifest).

The audited dataset has about 1.84M prompts, and every response costs a generation call plus
judge calls. A template contributes up to 15 near-identical harmful variants, so training on
all of them buys little over training on two. This tool picks a subset and writes it as an
explicit id manifest (`ids.txt`), which `msafetemp.augmentation.generate_responses --ids` then
answers. The selection is reproducible and independent of the generation run.

Selection:

  benign (direct + bounded)   all rows. This is the scarce arm (about 149k against 1.23M
                              harmful) and protects the trained model from over-refusing.
                              `benign_audit_rejected` rows are a separate category after the
                              audit and are therefore excluded.
  posthoc                     all rows (about 77k); the hardest refusal case in the set.
  harmful                     capped to --harmful_target, spread over the languages by a flat
                              per-language quota and over the templates by a per-group cap,
                              with rows a reference model actually failed taken first.

Reference-model preference (optional, --eval_dir). If an evaluation of a reference model on
the train split is given (the `<split>` output directory of
`msafetemp.evaluation.wildguard_eval --splits train`, containing `harmful_<lang>/eval_results.jsonl`),
harmful rows it answered with `unsafe_compliance` are taken first, at a larger per-group cap
(--harmful_fail_cap), then rows with `safe_compliance`, then the rest. The per-language quota
is unchanged, only which rows fill it. The preference is a tie-break, not a filter: languages
or branches without reference verdicts get the same quota, chosen at random. Without
--eval_dir every harmful row is in the last tier. The released manifest used a scored
evaluation of an earlier in-house checkpoint (833k harmful train rows over 30 languages,
19,804 unsafe_compliance, 89.1% of them covered at cap 6); that evaluation is not released,
so `ids.txt` is shipped as data (see docs/DATA.md).

    python -m msafetemp.training.data.select_response_rows \\
        --dataset $MSAFETEMP_DATA_DIR/wildguard/hf_audited --out $OUT/responses_manifest

Outputs: `<out>/ids.txt` (sorted ids) and `<out>/manifest_summary.json`.
"""
import argparse
import collections
import glob
import json
import os
import random

CAT_BENIGN = "benign"          # not benign_audit_rejected: that category is the audit's reject pile
LIGHT = ["id", "language", "category", "group_id", "benign_class"]


def read_light(dataset, split):
    """Read only the selection columns of a save_to_disk split (Arrow IPC streams)."""
    import pyarrow as pa
    out = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(dataset, split, "data-*.arrow"))):
        with pa.memory_map(f, "r") as m:
            t = pa.ipc.open_stream(m).read_all().select(LIGHT)
        for k, v in t.to_pydict().items():
            out[k].extend(v)
    return out


def reference_verdicts(eval_dir):
    """{dataset row id: verdict} for every scored harmful row of a reference-model evaluation.

    The evaluation writes its own id (`harmful_german_wg_0002_paraphrase_p0_v0`); the dataset
    uses `german_wg_0002_harmful_para0_0`. The dataset id is rebuilt from the evaluation's own
    fields rather than by rewriting the evaluation id."""
    v = {}
    if not eval_dir:
        return v
    for d in sorted(glob.glob(os.path.join(eval_dir, "harmful_*"))):
        p = os.path.join(d, "eval_results.jsonl")
        if not os.path.exists(p):
            continue                                    # language started but never scored
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("verdict"):
                continue
            pi = int(r.get("paraphrase_index", -1))
            suf = "main" if r.get("branch") == "main" else f"para{pi}"
            v[f"{r['language']}_{r['hb_id']}_harmful_{suf}_{r.get('variant_index', 0)}"] = r["verdict"]
    return v


# Reference verdict -> priority tier (0 is taken first).
#   unsafe_compliance  it answered and the answer was harmful
#   safe_compliance    it answered; the answer happened to be harmless (still no refusal)
#   everything else    it refused, or was never scored
TIER = {"unsafe_compliance": 0, "safe_compliance": 1}


def pick_harmful(light, verdicts, target, cap, fail_cap, seed):
    by_lang = collections.defaultdict(list)
    for i, c in enumerate(light["category"]):
        if c == "harmful":
            by_lang[light["language"][i]].append(i)

    quota = target // len(by_lang)
    rng = random.Random(seed)
    picked, taken = [], collections.Counter()
    stats = {}
    for lang in sorted(by_lang):
        idx = by_lang[lang]
        rng.shuffle(idx)                                 # break template order, reproducibly
        tiers = [TIER.get(verdicts.get(light["id"][i]), 2) for i in idx]
        got = []
        # Pass A: rows the reference model failed, at the wider per-group cap.
        for i, t in zip(idx, tiers):
            if t != 0 or len(got) >= quota:
                continue
            g = light["group_id"][i]
            if taken[g] >= fail_cap:
                continue
            taken[g] += 1
            got.append(i)
        n_fail = len(got)
        # Pass B: fill the quota at the normal cap, still preferring near-misses.
        for t_want in (1, 2):
            for i, t in zip(idx, tiers):
                if t != t_want or len(got) >= quota:
                    continue
                g = light["group_id"][i]
                if taken[g] >= cap:
                    continue
                taken[g] += 1
                got.append(i)
        picked.extend(got)
        stats[lang] = {"picked": len(got), "quota": quota, "from_reference_failures": n_fail,
                       "available": len(idx)}
    return picked, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="audited WildGuard-track dataset (save_to_disk dir)")
    ap.add_argument("--eval_dir", default=None,
                    help="optional reference-model evaluation of the train split "
                         "(dir with harmful_<lang>/eval_results.jsonl)")
    ap.add_argument("--split", default="train",
                    help="train only by default: responses on the test split would contaminate "
                         "the over-refusal evaluation that reads it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--harmful_target", type=int, default=175_000)
    ap.add_argument("--harmful_cap", type=int, default=2, help="max variants per group")
    ap.add_argument("--harmful_fail_cap", type=int, default=6,
                    help="max variants per group for rows the reference model failed")
    ap.add_argument("--seed", type=int, default=111)
    a = ap.parse_args()

    print(f"[select] reading {a.dataset} split={a.split}", flush=True)
    light = read_light(a.dataset, a.split)
    n = len(light["id"])
    print(f"[select] {n:,} rows", flush=True)

    verdicts = reference_verdicts(a.eval_dir)
    fails = {k for k, v in verdicts.items() if v == "unsafe_compliance"}
    print(f"[select] reference model: {len(verdicts):,} scored, {len(fails):,} unsafe_compliance", flush=True)

    keep = []
    for i, c in enumerate(light["category"]):
        if c in (CAT_BENIGN, "posthoc"):
            keep.append(i)
    harmful, lang_stats = pick_harmful(light, verdicts, a.harmful_target,
                                       a.harmful_cap, a.harmful_fail_cap, a.seed)
    keep.extend(harmful)

    ids = sorted(light["id"][i] for i in keep)
    kept = set(ids)

    by_arm = collections.Counter()
    by_lang = collections.Counter()
    for i in keep:
        c, bc = light["category"][i], light["benign_class"][i]
        by_arm[c if c != CAT_BENIGN else f"benign_{bc}"] += 1
        by_lang[light["language"][i]] += 1
    covered = len(fails & kept)
    summary = {
        "dataset": a.dataset, "split": a.split, "seed": a.seed,
        "total_rows": len(ids),
        "by_arm": dict(sorted(by_arm.items())),
        "languages": len(by_lang),
        "rows_per_language": {"min": min(by_lang.values()), "median":
                              sorted(by_lang.values())[len(by_lang) // 2],
                              "max": max(by_lang.values())},
        "reference_failures_total": len(fails),
        "reference_failures_covered": covered,
        "reference_failure_coverage_pct": round(100.0 * covered / max(1, len(fails)), 1),
        "harmful_cap_per_group": a.harmful_cap,
        "harmful_fail_cap_per_group": a.harmful_fail_cap,
        "per_language": lang_stats,
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ids.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    json.dump(summary, open(os.path.join(a.out, "manifest_summary.json"), "w"), indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_language"}, indent=2))
    print(f"\n[select] wrote {a.out}/ids.txt  ({len(ids):,} ids)")


if __name__ == "__main__":
    main()
