#!/usr/bin/env python3
"""Render the Megatron-LM run config for one Apertus-8B safety SFT run.

Combines configs/training/apertus_megatron_sft.json (architecture, tokenization, hyperparameters,
family-split settings) with the mix manifest written by `build_sft_mix.py` (source file, sha256,
per-bin blend weights) and the local checkpoint paths. The prepare step later re-hashes the source
and refuses to run if it changed.

    python -m msafetemp.training.megatron_sft.render_config --kind instruct \
        --mix-manifest $MIX/mix_manifest.json --weight-checkpoint $APERTUS_INSTRUCT_DIR \
        --tokenizer-template $APERTUS_INSTRUCT_DIR --work-root $WORK_ROOT --output $WORK_ROOT/run_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE = REPO_ROOT / "configs" / "training" / "apertus_megatron_sft.json"


def render(template: dict, manifest: dict, kind: str, weight_checkpoint: str, tokenizer_template: str,
           work_root: str) -> dict:
    cfg = json.loads(json.dumps(template))
    cfg.pop("_comment", None)
    kinds = cfg.pop("kinds")
    if kind not in kinds:
        raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(kinds)}")
    weights = manifest["train_weights"]
    cfg["run_name"] = cfg["run_name"].format(kind=kind)
    cfg["work_root"] = work_root
    cfg["dataset_bins"] = {k: [k] for k in weights}
    cfg["training"]["train_weights"] = weights
    cfg["source"].update({"jsonl": manifest["source_jsonl"], "sha256": manifest["source_sha256"]})
    cfg["model"].update({
        # the prepare step tokenizes with the run-local tokenizer view (SFT markers recorded)
        "hf_checkpoint": str(Path(work_root) / "tokenizer_sft_view"),
        "tokenizer_template_checkpoint": tokenizer_template,
        "training_start": kind,
        "weight_checkpoint": weight_checkpoint,
        **kinds[kind],
    })
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--mix-manifest", required=True)
    ap.add_argument("--kind", choices=["base", "instruct"], required=True)
    ap.add_argument("--weight-checkpoint", required=True, help="local HF checkpoint to fine-tune")
    ap.add_argument("--tokenizer-template", required=True, help="local Apertus-8B-Instruct-2509 directory")
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = render(json.load(open(a.template)), json.load(open(a.mix_manifest)), a.kind,
                 a.weight_checkpoint, a.tokenizer_template, a.work_root)
    # sorted keys: the prepare step iterates dataset_bins in file order
    out.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(out, json.dumps(cfg["training"]["train_weights"], sort_keys=True))


if __name__ == "__main__":
    main()
