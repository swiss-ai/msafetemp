#!/usr/bin/env python3
"""Calculate the packed one-epoch SFT budget after tokenization.

    MEGATRON_ROOT=... python -m msafetemp.training.megatron_sft.plan --config run_config.json \
        --dataset-root $WORK_ROOT/dataset --output $WORK_ROOT/training_plan.json

Writes training_plan.json and training_plan.env (TRAIN_SAMPLES, TRAIN_ITERS, EVAL_INTERVAL,
EVAL_ITERS, GLOBAL_BATCH_SIZE). For the reference mix: 453,504 packed sequences = 3,543 steps at GBS 128.

Runs in the Megatron environment and uses Megatron's exact BFD counter
(`tools/preflight_sft_packing.py` of the Megatron-LM checkout named by MEGATRON_ROOT). The planned training
budget is one packed epoch of the unweighted source pool, rounded down to a
complete global batch; the configured train blend still controls which route
classes are sampled within that fixed budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _packed_counter():
    megatron_root = os.environ.get("MEGATRON_ROOT")
    if not megatron_root:
        raise RuntimeError("MEGATRON_ROOT must point to the pinned Megatron-LM checkout")
    sys.path.insert(0, megatron_root)
    from tools.preflight_sft_packing import packed_count_for_idx
    return packed_count_for_idx


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    packed_count_for_idx = _packed_counter()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    manifest = json.loads((arguments.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("dataset manifest is not complete")
    categories = list(config["dataset_bins"])
    train_stats: dict[str, dict[str, int]] = {}
    valid_stats: dict[str, dict[str, int]] = {}
    for split, output in (("train", train_stats), ("valid", valid_stats)):
        for category in categories:
            prefix = arguments.dataset_root / split / category
            docs, tokens, samples = packed_count_for_idx(
                prefix.with_suffix(".idx"),
                seq_length=int(config["training"]["seq_length"]),
                add_extra_token=1,
                strategy=config["training"]["packing_strategy"],
                seed=int(config["training"]["seed"]),
                max_docs_per_bin=int(config["training"]["max_docs_per_bin"]),
            )
            if docs <= 0 or tokens <= 0 or samples <= 0:
                raise ValueError(f"{split}/{category} produced an empty packed dataset")
            output[category] = {"documents": docs, "tokens": tokens, "packed_samples": samples}

    global_batch_size = int(config["training"]["global_batch_size"])
    train_packed_total = sum(row["packed_samples"] for row in train_stats.values())
    valid_packed_total = sum(row["packed_samples"] for row in valid_stats.values())
    train_samples = (train_packed_total // global_batch_size) * global_batch_size
    if train_samples == 0:
        raise ValueError("one packed epoch does not contain a full global batch")
    train_iters = train_samples // global_batch_size
    # Megatron retains this single-pass held-out iterator across evaluations.
    # It must therefore be evaluated once, at the final training iteration.
    eval_iters = min(32, valid_packed_total // global_batch_size)
    if eval_iters == 0:
        raise ValueError("validation split does not contain one complete global batch")
    eval_interval = train_iters
    plan = {
        "status": "complete",
        "dataset_manifest_sha256": __import__("hashlib").sha256(
            (arguments.dataset_root / "dataset_manifest.json").read_bytes()
        ).hexdigest(),
        "seq_length": int(config["training"]["seq_length"]),
        "packing_strategy": config["training"]["packing_strategy"],
        "max_docs_per_bin": int(config["training"]["max_docs_per_bin"]),
        "global_batch_size": global_batch_size,
        "train_bins": train_stats,
        "valid_bins": valid_stats,
        "train_packed_samples_unweighted_one_epoch": train_packed_total,
        "valid_packed_samples": valid_packed_total,
        "train_samples": train_samples,
        "train_iters": train_iters,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters,
        "train_weights": config["training"]["train_weights"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output, plan)
    arguments.output.with_suffix(".env").write_text(
        "\n".join(
            [
                f"TRAIN_SAMPLES={train_samples}",
                f"TRAIN_ITERS={train_iters}",
                f"EVAL_INTERVAL={eval_interval}",
                f"EVAL_ITERS={eval_iters}",
                "GLOBAL_BATCH_SIZE=" + str(global_batch_size),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
