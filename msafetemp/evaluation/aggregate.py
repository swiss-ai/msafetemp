"""Aggregate per-arm stats.json files of HarmBench / WildGuard runs into summary rates.

Only scored rows count everywhere (degenerate, target_failed and judge_failed rows are excluded).
ASR = unsafe_compliance / scored; OR = over_refusal / scored on benign `direct` pairs.

Two aggregation protocols:

  --protocol breakdown  (Orig / EN / TR / PH / OR split per model)
      HarmBench: Orig = original_standard; EN = standard_english;
                 TR = POOLED over the non-English standard_<lang> arms;
                 PH = POOLED over posthoc_standard_<lang>; OR = POOLED over benign_standard_<lang>
      WildGuard: Orig = original; EN = harmful_english;
                 TR = MACRO mean over the non-English harmful_<lang> arms;
                 PH = MACRO over posthoc_<lang>; OR = MACRO over benign_<lang> (English included);
                 Valid = scored / all rows over every arm
  --protocol summary    (English vs multilingual, e.g. to compare training checkpoints)
      EN = English arm; ML = MACRO mean over all language arms, English included
      (60 HarmBench, 44 WildGuard); PH = POOLED over the post-hoc arms (if present).
      An arm with no scored row reports rate 0.0 in stats.json and enters the macro mean as 0.

Per-language rates (TR / PH / OR per model and language) are written with --per_language_csv.

Usage:
  python -m msafetemp.evaluation.aggregate --track hb --protocol breakdown \
      --run Apertus-8B=$OUT/hb/Apertus-8B-Instruct-2509 --run Qwen3.5-27B=$OUT/hb/Qwen3.5-27B \
      --out_csv hb.csv --per_language_csv hb_perlang.csv
For WildGuard, point --run at the eval output root (the client's --output); the split
subdirectory (--split, default test) is appended. Extra directories can be merged into one run
with a second --run of the same name (e.g. the separate PH-only HarmBench run of a checkpoint).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import OrderedDict

HB_PREFIX = {"orig": "original_standard", "tr": "standard_", "ph": "posthoc_standard_",
             "or": "benign_standard_", "en": "standard_english", "en_or": "benign_standard_english"}
WG_PREFIX = {"orig": "original", "tr": "harmful_", "ph": "posthoc_", "or": "benign_",
             "en": "harmful_english", "en_or": "benign_english"}


def load_stats(dirs):
    """arm -> stats.json dict, merged over one or more run directories."""
    arms = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*", "stats.json")):
            try:
                with open(f) as fh:
                    arms[os.path.basename(os.path.dirname(f))] = json.load(fh)
            except Exception:
                pass
    return arms


def _rate_key(s):
    return "over_refusal_rate" if s.get("eval_kind") == "benign" else "asr"


def _bad_key(s):
    return "over_refusal" if s.get("eval_kind") == "benign" else "unsafe_compliance"


def pooled(arms):
    bad = sum((s.get("verdict_counts") or {}).get(_bad_key(s), 0) for s in arms.values())
    n = sum(s.get("scored", 0) for s in arms.values())
    return bad / n if n else None


def macro(arms):
    vals = [s[_rate_key(s)] for s in arms.values() if s.get(_rate_key(s)) is not None]
    return statistics.mean(vals) if vals else None


def single(arms, name):
    s = arms.get(name)
    return s.get(_rate_key(s)) if s else None


def select(arms, prefix, exclude=()):
    return {k: v for k, v in arms.items() if k.startswith(prefix) and k not in exclude
            and not (prefix == "standard_" and k.startswith("posthoc_"))}


def language_of(arm, prefix):
    return arm[len(prefix):]


def summarize_run(arms, track, protocol):
    P = HB_PREFIX if track == "hb" else WG_PREFIX
    tr = select(arms, P["tr"])
    ph = select(arms, P["ph"])
    ben = select(arms, P["or"])
    out = OrderedDict()
    out["orig_asr"] = single(arms, P["orig"])
    out["en_asr"] = single(arms, P["en"])
    if protocol == "breakdown":
        tr_ne = select(arms, P["tr"], exclude={P["en"]})
        agg = pooled if track == "hb" else macro
        out["tr_asr"] = agg(tr_ne) if tr_ne else None
        out["ph_asr"] = agg(ph) if ph else None
        out["or"] = agg(ben) if ben else None
    else:
        out["ml_asr"] = macro(tr) if tr else None
        out["ml_asr_pooled"] = pooled(tr) if tr else None
        out["ph_asr"] = pooled(ph) if ph else None
        out["en_or"] = single(arms, P["en_or"])
        out["ml_or"] = macro(ben) if ben else None
    total = sum(s.get("total_rows", 0) for s in arms.values())
    scored = sum(s.get("scored", 0) for s in arms.values())
    out["valid"] = scored / total if total else None
    out["scored_share_harmful"] = (sum(s.get("scored", 0) for s in tr.values())
                                   / max(sum(s.get("total_rows", 0) for s in tr.values()), 1)) if tr else None
    out["scored_share_benign"] = (sum(s.get("scored", 0) for s in ben.values())
                                  / max(sum(s.get("total_rows", 0) for s in ben.values()), 1)) if ben else None
    out["arms_lt50pct_scored"] = sum(1 for s in list(tr.values()) + list(ben.values())
                                     if s.get("scored", 0) < 0.5 * max(s.get("total_rows", 1), 1))
    out["n_tr_arms"], out["n_ph_arms"], out["n_benign_arms"] = len(tr), len(ph), len(ben)
    out["failed_rows"] = sum(s.get("target_failed", 0) + s.get("judge_failed", 0) for s in arms.values())
    return out


def per_language(arms, track):
    P = HB_PREFIX if track == "hb" else WG_PREFIX
    rows = {}
    for kind, prefix in (("tr", P["tr"]), ("ph", P["ph"]), ("or", P["or"])):
        for arm, s in select(arms, prefix).items():
            lang = language_of(arm, prefix)
            r = rows.setdefault(lang, {"language": lang})
            r[f"{kind}_rate"] = s.get(_rate_key(s))
            r[f"{kind}_scored"] = s.get("scored", 0)
            r[f"{kind}_rows"] = s.get("total_rows", 0)
    return [rows[k] for k in sorted(rows, key=lambda x: (x != "english", x))]


def pct(x):
    return "" if x is None else f"{100 * x:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["hb", "wg"], required=True)
    ap.add_argument("--protocol", choices=["breakdown", "summary"], required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                    help="model display name and eval output directory (repeatable)")
    ap.add_argument("--split", default="test", help="WildGuard split subdirectory")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--per_language_csv", default=None)
    a = ap.parse_args(argv)

    runs = OrderedDict()
    for spec in a.run:
        name, _, d = spec.partition("=")
        if not d:
            ap.error(f"--run needs NAME=DIR, got {spec!r}")
        if a.track == "wg":
            d = os.path.join(d, a.split)
        runs.setdefault(name, []).append(d)

    table, perlang = [], []
    for name, dirs in runs.items():
        arms = load_stats(dirs)
        if not arms:
            print(f"WARNING: no stats.json under {dirs}")
            continue
        row = OrderedDict(model=name, **summarize_run(arms, a.track, a.protocol))
        table.append(row)
        for r in per_language(arms, a.track):
            perlang.append({"model": name, **r})

    if table:
        cols = list(table[0].keys())
        print("  ".join(f"{c:>12}" for c in cols))
        for r in table:
            print("  ".join(f"{(pct(v) if isinstance(v, float) else str(v)):>12}" for v in r.values()))
    if a.out_csv and table:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            for r in table:
                w.writerow({k: (round(100 * v, 4) if isinstance(v, float) and k not in ("model",) else v)
                            for k, v in r.items()})
    if a.per_language_csv and perlang:
        cols = ["model", "language"] + [f"{k}_{m}" for k in ("tr", "ph", "or") for m in ("rate", "scored", "rows")]
        with open(a.per_language_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in perlang:
                w.writerow({c: (round(100 * r[c], 4) if c.endswith("_rate") and r.get(c) is not None
                                else r.get(c, "")) for c in cols})


if __name__ == "__main__":
    main()
