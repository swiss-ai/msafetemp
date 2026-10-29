"""Jailbreak-wrapped HarmBench evaluation (HB-JB): 300 prompts per language.

Harmful: HarmBench-derived PH-JB rows (the culturally localized post-hoc request inserted into a
PyRIT / in-the-wild jailbreak wrapper), test split, 58 languages -> 17,400 rows.
Benign:  wrapped benign HarmBench prompts (over-refusal), 60 languages -> 18,000 rows.
Every model receives the same ids.

Selection (outcome blind, input identity only): per language, the 300 rows with the lowest
    sha256(canonical_json([selection_version, seed, track, language, id]))
with seed 20260914 and selection_version "phjb-language-sample-v1" (harmful) or
"hb300-benign-wrapped-sample-v1" (benign). Prompts are copied unchanged.

Subcommands
  select     build inputs/{harmful,benign}.jsonl (+ id lists, manifest) from candidate JSONL
             files; candidate rows need id, track, language, prompt.
  summarize  per model: ASR / OR macro over languages (headline) and pooled, scored counts,
             judge failures, target errors, empty and truncated answers, and the sensitivity
             rate without empty/truncated answers. Reads the jsonl_eval.py outputs
             <runs_dir>/<model_key>/{targets,judgments}_<kind>.jsonl.

Run the evaluation itself with msafetemp.evaluation.jsonl_eval (one call per model).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from msafetemp.evaluation._utils import digest, file_sha, read_jsonl

CAP = 300
SEED = 20260914
VERSIONS = {"harmful": "phjb-language-sample-v1", "benign": "hb300-benign-wrapped-sample-v1"}


def select(rows, version, cap=CAP, seed=SEED):
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append((digest([version, seed, r["track"], r["language"], r["id"]]), r))
    chosen, counts = [], {}
    for lang in sorted(by_lang):
        ranked = sorted(by_lang[lang], key=lambda x: x[0])[:cap]
        counts[lang] = {"available": len(by_lang[lang]), "selected": len(ranked)}
        chosen.extend(ranked)
    return chosen, counts


def write_selection(out_dir: Path, name, chosen, kind):
    path = out_dir / f"{name}.jsonl"
    ids = []
    with path.open("w") as f:
        for rank, r in sorted(chosen, key=lambda x: (x[1]["language"], x[0])):
            rec = {"id": r["id"], "kind": kind, "track": r["track"], "language": r["language"],
                   "prompt": r["prompt"], "source_family_id": r.get("source_family_id"),
                   "source_id": r.get("source_id"), "selection_rank_sha256": rank,
                   "source_row_sha256": digest(r)}
            if kind == "benign":
                rec["benchmark_subset"] = r.get("benchmark_subset")
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            ids.append(r["id"])
    blob = ("\n".join(sorted(ids)) + "\n").encode()
    (out_dir / f"{name}_ids.txt").write_bytes(blob)
    return {"path": str(path), "rows": len(ids), "unique_ids": len(set(ids)),
            "file_sha256": file_sha(path), "sorted_id_list_sha256": hashlib.sha256(blob).hexdigest()}


def cmd_select(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"per_language_cap": a.cap, "seed": a.seed}
    for kind, paths in (("harmful", a.harmful), ("benign", a.benign)):
        if not paths:
            continue
        rows = [r for p in paths for r in read_jsonl(p) if a.track is None or r["track"] == a.track]
        sel, counts = select(rows, VERSIONS[kind], a.cap, a.seed)
        manifest[kind] = {"selection_version": VERSIONS[kind], "languages": len(counts),
                          "per_language": counts, **write_selection(out, kind, sel, kind)}
        print(kind, manifest[kind]["rows"], "rows,", len(counts), "languages,",
              "id-list sha256", manifest[kind]["sorted_id_list_sha256"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def load_journal(path: Path):
    out = {}
    if path.exists():
        for v in read_jsonl(path):
            out[v["id"]] = v
    return out


def metric(kind, rows, targets, judged):
    flag = "attack_success" if kind == "harmful" else "is_over_refusal"
    per_lang = defaultdict(lambda: [0, 0])
    c = defaultdict(int)
    sens = [0, 0]
    for r in rows:
        c["rows"] += 1
        t, j = targets.get(r["id"]), judged.get(r["id"])
        if t is None:
            c["no_target"] += 1
        elif t["target_status"] != "target_complete":
            c["target_error"] += 1
        else:
            c["empty"] += bool(t.get("response_empty"))
            c["truncated"] += bool(t.get("truncated"))
        if j is None:
            if t is not None and t["target_status"] == "target_complete":
                c["unjudged"] += 1
            continue
        if j["status"] == "scored":
            c["scored"] += 1
            pos = bool(j[flag])
            c["positive"] += pos
            per_lang[r["language"]][0] += pos
            per_lang[r["language"]][1] += 1
            if not (t.get("response_empty") or t.get("truncated")):
                sens[0] += pos
                sens[1] += 1
        elif j["status"] == "judge_failed":
            c["judge_failed"] += 1
    rates = [p / n for p, n in per_lang.values() if n]
    return {
        **{k: c[k] for k in ("rows", "scored", "positive", "judge_failed", "target_error", "no_target",
                             "unjudged", "empty", "truncated")},
        "languages_scored": len(rates),
        "macro_pct": 100 * sum(rates) / len(rates) if rates else None,
        "pooled_pct": 100 * c["positive"] / c["scored"] if c["scored"] else None,
        "pooled_nonempty_nontrunc_pct": 100 * sens[0] / sens[1] if sens[1] else None,
        "per_language_pct": {lang: 100 * p / n for lang, (p, n) in sorted(per_lang.items()) if n},
    }


def cmd_summarize(a):
    inputs = {k: list(read_jsonl(Path(a.inputs_dir) / f"{k}.jsonl")) for k in ("harmful", "benign")
              if (Path(a.inputs_dir) / f"{k}.jsonl").exists()}
    models = []
    for spec in a.model:
        key, _, name = spec.partition("=")
        models.append((key, name or key))
    results = {}
    for key, name in models:
        d = Path(a.runs_dir) / key
        if not d.exists():
            continue
        res = {"name": name}
        for kind, rows in inputs.items():
            tp = d / f"targets_{kind}.jsonl"
            if tp.exists():
                res[kind] = metric(kind, rows, load_journal(tp), load_journal(d / f"judgments_{kind}.jsonl"))
        results[key] = res
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    cols = ["rows", "scored", "judge_failed", "target_error", "unjudged", "empty", "truncated",
            "languages_scored", "macro_pct", "pooled_pct", "pooled_nonempty_nontrunc_pct"]
    with (out / "report.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind"] + cols)
        for res in results.values():
            for kind in ("harmful", "benign"):
                if kind in res:
                    m = res[kind]
                    w.writerow([res["name"], kind] + [round(m[c], 3) if isinstance(m[c], float) else m[c]
                                                      for c in cols])
    with (out / "per_language.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind", "language", "rate_pct"])
        for res in results.values():
            for kind in ("harmful", "benign"):
                for lang, v in res.get(kind, {}).get("per_language_pct", {}).items():
                    w.writerow([res["name"], kind, lang, round(v, 3)])
    fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    print("| Model | Wrapped harmful ASR (macro) | Wrapped benign OR (macro) |")
    print("|---|---:|---:|")
    for res in results.values():
        print(f"| {res['name']} | {fmt(res.get('harmful', {}).get('macro_pct'))} | "
              f"{fmt(res.get('benign', {}).get('macro_pct'))} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="freeze the 300-per-language subsample")
    s.add_argument("--harmful", nargs="*", default=[], help="candidate PH-JB JSONL file(s)")
    s.add_argument("--benign", nargs="*", default=[], help="candidate wrapped-benign JSONL file(s)")
    s.add_argument("--track", default="harmbench", help="keep only rows of this track (None = all)")
    s.add_argument("--cap", type=int, default=CAP)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out_dir", required=True)
    s.set_defaults(func=cmd_select)
    m = sub.add_parser("summarize", help="ASR / OR tables from jsonl_eval outputs")
    m.add_argument("--inputs_dir", required=True, help="directory with harmful.jsonl / benign.jsonl")
    m.add_argument("--runs_dir", required=True, help="directory with one jsonl_eval --out_dir per model key")
    m.add_argument("--model", action="append", required=True, metavar="KEY[=DISPLAY NAME]")
    m.add_argument("--out_dir", required=True)
    m.set_defaults(func=cmd_summarize)
    a = ap.parse_args(argv)
    if getattr(a, "track", "") in ("None", "none", ""):
        a.track = None
    a.func(a)


if __name__ == "__main__":
    main()
