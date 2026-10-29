#!/usr/bin/env python3
"""Checkpoint hygiene for TRL outputs.

sanitise_generation_config(model)
    Some vendor checkpoints (e.g. Olmo 3) ship `do_sample=False` together with `temperature` /
    `top_p` in generation_config.json. `save_pretrained` validates the generation config strictly
    and raises `ValueError: GenerationConfig is invalid`, which loses a completed run at the very
    last step. Called before and after training: if the config is invalid, the sampling fields are
    nulled (greedy decoding is kept rather than silently switching the saved model to sampling);
    if it is still invalid, a minimal GenerationConfig with the special-token ids is used.

restore-config OUT_DIR VENDOR_CONFIG_JSON
    Newer transformers versions save some architectures (Olmo 3) with a nested `rope_parameters`
    block and without `rope_scaling` / `rope_theta`, which vLLM rejects at load time. This copies
    the vendor config.json back into the trained model directory (dtype bfloat16, `torch_dtype`
    dropped) and keeps the trained one as config.tf5.json. Weights are untouched.

    python -m msafetemp.training.hf_utils restore-config $OUT/Olmo-3-7B-Instruct-SFT $OLMO_DIR/config.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def sanitise_generation_config(model) -> None:
    config = getattr(model, "generation_config", None)
    if config is None:
        return

    def strict_validate(c):
        # save_pretrained validates with strict=True; plain validate() only warns
        try:
            c.validate(strict=True)
        except TypeError:  # older transformers without the strict flag
            c.validate()

    try:
        strict_validate(config)
        return
    except Exception as exc:
        print("[hf_utils] generation_config invalid (%s); dropping sampling fields" % exc, flush=True)
    for field in ("temperature", "top_p", "top_k", "typical_p", "epsilon_cutoff", "eta_cutoff"):
        if getattr(config, field, None) is not None:
            setattr(config, field, None)
    try:
        strict_validate(config)
    except Exception as exc:
        from transformers import GenerationConfig
        print("[hf_utils] still invalid (%s); using a default GenerationConfig" % exc, flush=True)
        model.generation_config = GenerationConfig(
            bos_token_id=getattr(config, "bos_token_id", None),
            eos_token_id=getattr(config, "eos_token_id", None),
            pad_token_id=getattr(config, "pad_token_id", None),
        )


def restore_config(out_dir: str, vendor_config: str) -> None:
    out = Path(out_dir)
    backup = out / "config.tf5.json"
    if not backup.exists():
        shutil.copy2(out / "config.json", backup)
    c = json.load(open(vendor_config))
    c["dtype"] = "bfloat16"
    c.pop("torch_dtype", None)
    json.dump(c, open(out / "config.json", "w"), indent=2)
    print("restored", out / "config.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("restore-config")
    r.add_argument("out_dir")
    r.add_argument("vendor_config")
    a = ap.parse_args()
    restore_config(a.out_dir, a.vendor_config)


if __name__ == "__main__":
    main()
