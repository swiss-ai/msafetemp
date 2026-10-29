#!/usr/bin/env python3
"""Build the final SFT mix: one Megatron source file with per-bin blend weights, and a matched TRL sample.

Components (all responses come from the response generator: Gemma-4-31B-it answers, GPT-OSS-120B
judge, "decline, then help" refusals, accepted rows only):

  bin              target  source
  harmful          refuse  MSafeTemp safety rows (adapt_safety_rows.py), harmful arm
  posthoc          refuse  MSafeTemp safety rows, culturally localized (post-hoc) harmful arm
  benign_direct    comply  MSafeTemp safety rows, benign direct
  benign_bounded   comply  MSafeTemp safety rows, benign bounded
  vanilla_harmful  refuse  short unwrapped WildGuardMix-train prompts, English + translated
  vanilla_benign   comply  (short_prompts.py vanilla / translate)
  wrap_harmful     refuse  jailbreak-wrapped train prompts, 1k per language (wrapped_prompts.py wrap1k);
  wrap_benign      comply    the bin is taken from each row's category
  pyrit_harmful    refuse  earlier judged wrapped set (wrapped_prompts.py judged), harmful + posthoc;
                           its benign rows join wrap_benign
  short_contrast   comply  short benign contrast requests (short_contrast.py), EN + translated; every
                           sensitive-sounding twin plus innocuous SWAP / IDIOM / PROTECT twins
  general          comply  Dolci-Instruct-SFT, safety-cleaned (dolci_general.py)

Steps (the order and seeds reproduce the reference mix exactly):

  1. Megatron source = safety rows + generated rows + general rows + short-contrast rows, one jsonl.
     Blend weights = configs/training/sft_mix.json `base_weights` + `short_contrast_weight`;
     general is the remainder (0.735044).
  2. TRL (no bin weights): a --trl-rows sample with the proportions baked in. First the base mix
     (every bin except short_contrast, general = 1 - sum(base_weights)) is sampled per bin with
     random.Random(trl.seed), with repeats if a bin is smaller than its share, and shuffled; then
     round(short_contrast_weight * rows) general rows are swapped for short-contrast rows with
     random.Random(trl.short_contrast_seed) and the result is shuffled again.

Response directory layout under --responses-dir (each leaf holds <language>/responses.jsonl with
{id, language, prompt, response, accepted, category, group_id, ...}):

  vanilla/{harmful,benign}  vanilla_ml/{harmful,benign}  wrap1k/{harmful,benign}
  pyrit/{posthoc,harmful,benign}  short_contrast/{en,ml}/benign

Row order: response files are read in sorted path order. The reference run read them in directory-listing
order, which changes the order (not the set) of the generated rows and therefore the exact rows of the
seeded TRL sample. To reproduce the reference files row for row, pass --id-order with the generated-row ids
of the released mix in file order (`<bin>:<id>` per line).

Outputs in --out-dir: sft_source.jsonl, sft_trl_<rows/1000>k.jsonl, mix_manifest.json (weights,
per-bin counts, sha256). Render the Megatron run config from the manifest with
`python -m msafetemp.training.megatron_sft.render_config`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

from msafetemp.training._utils import refuse_overwrite, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "training" / "sft_mix.json"
SC = "short_contrast"

# (glob relative to --responses-dir, bin, target). Order fixes dedup precedence and file order.
GEN_SOURCES = [
    ("vanilla/harmful/*/responses.jsonl", "vanilla_harmful", "refuse"),
    ("vanilla/benign/*/responses.jsonl", "vanilla_benign", "comply"),
    ("wrap1k/harmful/*/responses.jsonl", "wrap_harmful", "refuse"),
    ("wrap1k/benign/*/responses.jsonl", "wrap_benign", "comply"),
    ("pyrit/posthoc/*/responses.jsonl", "pyrit_harmful", "refuse"),
    ("pyrit/harmful/*/responses.jsonl", "pyrit_harmful", "refuse"),
    ("pyrit/benign/*/responses.jsonl", "wrap_benign", "comply"),
    ("vanilla_ml/harmful/*/responses.jsonl", "vanilla_harmful", "refuse"),
    ("vanilla_ml/benign/*/responses.jsonl", "vanilla_benign", "comply"),
]
SC_SOURCES = "short_contrast/*/benign/*/responses.jsonl"


def accepted(r: dict) -> bool:
    return r.get("accepted") in (True, "True") and bool((r.get("response") or "").strip())


def sft_row(arm: str, target: str, r: dict) -> dict:
    return {"id": f"{arm}:{r['id']}", "language": r["language"],
            "family_id": f"{arm}:{r.get('group_id') or r['id']}",
            "messages": [{"role": "user", "content": r["prompt"]},
                         {"role": "assistant", "content": r["response"].strip()}],
            "response_mode": arm, "training_role": f"{arm}_{target}", "arm": arm, "target_behavior": target}


def generated_rows(responses_dir: Path, wrap_train_ids: set[str] | None):
    rows, stats, seen = [], collections.Counter(), set()
    for pattern, arm0, tb0 in GEN_SOURCES:
        for f in sorted(glob.glob(str(responses_dir / pattern))):
            is_wrap1k = "/wrap1k/" in f
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                arm, tb = arm0, tb0
                if not accepted(r):
                    stats[(arm, "rejected")] += 1
                    continue
                if r["id"] in seen:
                    continue
                # wrap1k: keep only train-family rows so the held-out wrapped test split stays clean
                if is_wrap1k and wrap_train_ids is not None and r["id"] not in wrap_train_ids:
                    stats[(arm, "drop_test_family")] += 1
                    continue
                # the generator may answer both categories into one output dir: bin by the row itself
                if is_wrap1k:
                    arm, tb = ("wrap_benign", "comply") if r.get("category") == "benign" else ("wrap_harmful", "refuse")
                seen.add(r["id"])
                stats[(arm, "kept")] += 1
                rows.append(sft_row(arm, tb, r))
    return rows, stats


def short_contrast_rows(responses_dir: Path, prompt_dirs: list[str], keep_innocuous: set[str]):
    """Accepted short-contrast answers; innocuous CREATIVE / ANALYZE twins are dropped (they drift
    off-topic, e.g. generic story premises)."""
    from datasets import load_from_disk
    meta = {}
    for d in prompt_dirs:
        for r in load_from_disk(d)["train"].to_list():
            meta[r["id"]] = (r["judge_sensitive"], r["contrast_type"])
    rows, stats, seen = [], collections.Counter(), set()
    for f in sorted(glob.glob(str(responses_dir / SC_SOURCES))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not accepted(r) or r["id"] in seen:
                stats["rejected_or_dup"] += 1
                continue
            sens, ctype = meta.get(r["id"], (None, None))
            if sens is None:
                stats["no_meta"] += 1
                continue
            if not sens and ctype not in keep_innocuous:
                stats["drop_innocuous_offtopic"] += 1
                continue
            stats["sensitive_kept" if sens else "innocuous_kept"] += 1
            seen.add(r["id"])
            stats[r["language"]] += 1
            rows.append(sft_row(SC, "comply", r))
    return rows, stats


def sample_to(pool: list, n: int, rng: random.Random) -> list:
    return rng.sample(pool, n) if n <= len(pool) else pool * (n // len(pool)) + rng.sample(pool, n % len(pool))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--safety-rows", required=True, help="adapted MSafeTemp safety SFT rows (adapt_safety_rows.py)")
    ap.add_argument("--general", required=True, help="safety-cleaned Dolci rows (dolci_general.py clean)")
    ap.add_argument("--responses-dir", required=True, type=Path)
    ap.add_argument("--wrap-train-ids", default=None, help="train_ids.txt of the wrapped 1k set (recommended)")
    ap.add_argument("--short-contrast-prompts", nargs="+", required=True,
                    help="short-contrast prompt datasets (EN and translated) with contrast_type / judge_sensitive")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--trl-rows", type=int, default=None, help="override trl.rows from the config")
    ap.add_argument("--id-order", default=None,
                    help="file with generated-row ids (<bin>:<id>) in the released order; reorders gen + short-contrast rows")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    trl_rows = a.trl_rows or cfg["trl"]["rows"]
    base_w = dict(cfg["base_weights"])
    wrap_ids = set(Path(a.wrap_train_ids).read_text().split()) if a.wrap_train_ids else None

    gen, gstats = generated_rows(a.responses_dir, wrap_ids)
    for k in sorted(gstats, key=str):
        print(k, gstats[k])
    sc_rows, sstats = short_contrast_rows(a.responses_dir, a.short_contrast_prompts,
                                          set(cfg["short_contrast_keep_innocuous_types"]))
    print("short_contrast rows", len(sc_rows), "| EN", sstats["english"],
          {k: v for k, v in sstats.items() if "_" in k})
    assert sc_rows, "no accepted short_contrast rows"
    if a.id_order:
        pos = {rid: i for i, rid in enumerate(Path(a.id_order).read_text().split())}
        key = lambda r: pos.get(r["id"], len(pos))  # noqa: E731
        gen.sort(key=key)  # stable: ids missing from the file keep their relative order at the end
        sc_rows.sort(key=key)

    # ---- 1. Megatron source ----
    a.out_dir.mkdir(parents=True, exist_ok=True)
    src = refuse_overwrite(a.out_dir / "sft_source.jsonl")
    per_bin = collections.Counter()
    with src.open("w", encoding="utf-8") as fo:
        for line in open(a.safety_rows, encoding="utf-8"):
            per_bin[json.loads(line)["response_mode"]] += 1
            fo.write(line)
        for r in gen:
            per_bin[r["arm"]] += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in open(a.general, encoding="utf-8"):
            per_bin["general"] += 1
            fo.write(line)
        for r in sc_rows:
            per_bin[SC] += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    base = {k: v for k, v in base_w.items() if per_bin[k] > 0}
    base["general"] = round(1.0 - sum(base.values()), 6)
    weights = {k: v for k, v in base.items() if k != "general"}
    weights[SC] = cfg["short_contrast_weight"]
    weights["general"] = round(1.0 - sum(weights.values()), 6)
    print("megatron source", src, dict(per_bin))
    print("weights", json.dumps(weights, sort_keys=True))

    # ---- 2. TRL sample: base mix, then short-contrast swap ----
    rng = random.Random(cfg["trl"]["seed"])
    by_bin = collections.defaultdict(list)
    for line in open(a.safety_rows, encoding="utf-8"):
        r = json.loads(line)
        by_bin[r["response_mode"]].append(r)
    for r in gen:
        by_bin[r["arm"]].append(r)
    by_bin["general"] = [json.loads(line) for line in open(a.general, encoding="utf-8")]
    trl = []
    for b, frac in base.items():
        for r in sample_to(by_bin[b], round(frac * trl_rows), rng):
            trl.append({"id": r["id"], "messages": r["messages"][:2], "language": r.get("language", "general"),
                        "arm": b, "target_behavior": r.get("target_behavior", "comply")})
    rng.shuffle(trl)
    del by_bin

    rng2 = random.Random(cfg["trl"]["short_contrast_seed"])
    n = round(cfg["short_contrast_weight"] * trl_rows)
    gen_idx = [i for i, r in enumerate(trl) if r.get("arm") == "general"]
    drop = set(rng2.sample(gen_idx, n))
    pick = sample_to(sc_rows, n, rng2)
    trl = [r for i, r in enumerate(trl) if i not in drop] + [
        {"id": r["id"], "messages": r["messages"], "language": r["language"], "arm": SC, "target_behavior": "comply"}
        for r in pick]
    rng2.shuffle(trl)
    out = refuse_overwrite(a.out_dir / f"sft_trl_{trl_rows // 1000}k.jsonl")
    with out.open("w", encoding="utf-8") as fo:
        for r in trl:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    trl_counts = dict(collections.Counter(r["arm"] for r in trl))
    print("trl mix", out, len(trl), trl_counts)

    expected = cfg.get("expected_bin_counts") or {}
    mismatch = {k: (per_bin.get(k, 0), v) for k, v in expected.items() if per_bin.get(k, 0) != v}
    if mismatch:
        print("NOTE: bin counts differ from the reference mix (got, reference):", mismatch)
    manifest = {"source_jsonl": str(src), "source_sha256": sha256_file(src), "rows": sum(per_bin.values()),
                "rows_per_bin": dict(sorted(per_bin.items())), "train_weights": weights,
                "trl_jsonl": str(out), "trl_rows_per_bin": trl_counts, "config": str(a.config),
                "inputs": {"safety_rows": a.safety_rows, "general": a.general,
                           "responses_dir": str(a.responses_dir), "wrap_train_ids": a.wrap_train_ids}}
    (a.out_dir / "mix_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("manifest", a.out_dir / "mix_manifest.json")


if __name__ == "__main__":
    main()
