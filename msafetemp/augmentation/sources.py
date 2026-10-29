"""Seed prompt sources.

WildGuardMix: harmful prompts of the `wildguardtrain` split, streamed through a seeded
shuffle (buffer 5,000). The released run drew n=10,000 prompts with at least 800
characters at seed 111; the 2,500 canonical seeds are then the first 2,500 English
originals that pass template validation (select_canonical.py).

HarmBench: the 200 `standard` behaviors (contextual and copyright behaviors are excluded:
contextual ones depend on an attached artifact, copyright ones are hash-matched against a
specific work), deterministically shuffled with the run seed. Behaviors are keyed by their
position in that shuffled order (`hb_id`).
"""

from __future__ import annotations

import random
from typing import Any


def load_wildguard_samples(n: int | None, seed: int, cache_dir: str | None = None,
                           harmful_only: bool = True,
                           min_prompt_length: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset
    kwargs: dict[str, Any] = {"streaming": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    # allenai/wildguardmix is gated: authenticate with `huggingface-cli login` or HF_TOKEN.
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", **kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    samples = []
    for row in ds:
        if harmful_only and row.get("prompt_harm_label") != "harmful":
            continue
        if min_prompt_length is not None and len(row.get("prompt", "")) < min_prompt_length:
            continue
        samples.append(dict(row))
        if n is not None and len(samples) >= n:
            break
    return samples


def load_harmbench_samples(n: int | None, seed: int,
                           cache_dir: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("walledai/HarmBench", "standard", split="train", **kwargs)
    rows = [dict(row) for row in ds]

    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None:
        rows = rows[:n]

    return [{
        "id": i,
        "behavior": row.get("prompt", ""),
        "hb_category": row.get("category"),
        "hb_functional_category": "standard",
    } for i, row in enumerate(rows)]
