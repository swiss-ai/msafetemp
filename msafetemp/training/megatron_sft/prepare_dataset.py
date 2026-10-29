#!/usr/bin/env python3
"""Tokenize the SFT mix into Megatron IndexedDataset bins (one .bin/.idx per dataset bin and split).

Run inside the Megatron environment (needs `megatron.core` from the Megatron-LM checkout on
PYTHONPATH and the tokenizer view from `tokenizer_view.py`):

    python -m msafetemp.training.megatron_sft.prepare_dataset --config run_config.json \
        --output-root $WORK_ROOT/dataset --workers 16

The source data is two-turn chat JSONL. Megatron's ApertusSFTDataset expects
already-tokenized IndexedDataset (.bin/.idx) documents, so this tool applies
the selected checkpoint's own chat template and writes one complete dialogue
per document. A sidecar metadata JSONL preserves each document's source ID and
route class without putting metadata into the model context.

This is intentionally fail-closed: source hash, message shape, family-level
split consistency, tokenizer SFT markers, and assistant-region boundaries are
validated before a document is emitted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from transformers import AutoTokenizer


_TOKENIZER = None
_BEGIN: list[int] | None = None
_END: list[int] | None = None
_ENABLE_THINKING = False
_MAX_TOKEN_ID: int | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(value: str, seed: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()


def _as_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValueError(f"tokenizer metadata {field!r} must be a non-empty integer list")
    return value


def _find_subsequence(tokens: list[int], sequence: list[int]) -> list[int]:
    width = len(sequence)
    return [
        position
        for position in range(len(tokens) - width + 1)
        if tokens[position : position + width] == sequence
    ]


def _normalise_token_ids(value: Any) -> list[int]:
    # Recent Apertus tokenizers return a Hugging Face BatchEncoding even with
    # tokenize=True.  ``attention_mask`` is standard auxiliary metadata, not
    # a second sequence, so accept it while rejecting any unknown payload.
    if isinstance(value, Mapping):
        unknown_fields = set(value).difference({"input_ids", "attention_mask"})
        if unknown_fields or "input_ids" not in value:
            raise ValueError("chat template BatchEncoding must contain input_ids and no unknown fields")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template returned a batch instead of one sequence")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("chat template did not return a token-ID list")
    return value


def _worker_init(tokenizer_path: str, enable_thinking: bool, max_token_id: int | None) -> None:
    global _TOKENIZER, _BEGIN, _END, _ENABLE_THINKING, _MAX_TOKEN_ID
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    init_kwargs = getattr(_TOKENIZER, "init_kwargs", {})
    _BEGIN = _as_int_list(init_kwargs.get("sft_assistant_begin_sequence"), "sft_assistant_begin_sequence")
    _END = _as_int_list(init_kwargs.get("sft_eot_token"), "sft_eot_token")
    _ENABLE_THINKING = enable_thinking
    _MAX_TOKEN_ID = max_token_id


def _render_and_tokenize(record: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    if _TOKENIZER is None or _BEGIN is None or _END is None:
        raise RuntimeError("tokenizer worker was not initialized")

    messages = record.get("messages")
    if not (
        isinstance(messages, list)
        and len(messages) == 2
        and all(isinstance(message, dict) for message in messages)
        and messages[0].get("role") == "user"
        and messages[1].get("role") == "assistant"
        and all(isinstance(message.get("content"), str) and message["content"] for message in messages)
    ):
        raise ValueError(f"{record.get('id', '<unknown>')}: expected exactly one non-empty user/assistant pair")

    token_ids = _normalise_token_ids(
        _TOKENIZER.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=_ENABLE_THINKING,
        )
    )
    begin_positions = _find_subsequence(token_ids, _BEGIN)
    end_positions = _find_subsequence(token_ids, _END)
    if len(begin_positions) != 1:
        raise ValueError(f"{record['id']}: expected one assistant-begin marker, found {len(begin_positions)}")
    if len(end_positions) != 1 or end_positions[0] <= begin_positions[0]:
        raise ValueError(f"{record['id']}: invalid assistant-end marker placement")
    if token_ids[-len(_END) :] != _END:
        raise ValueError(f"{record['id']}: rendered conversation does not end at the assistant EOT marker")
    if len(token_ids) <= len(_BEGIN) + len(_END):
        raise ValueError(f"{record['id']}: rendered conversation has no assistant content")
    if _MAX_TOKEN_ID is not None and max(token_ids) > _MAX_TOKEN_ID:
        raise ValueError(
            f"{record['id']}: token ID {max(token_ids)} exceeds the allowed text-only maximum "
            f"{_MAX_TOKEN_ID}"
        )
    return record, token_ids


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"source", "model", "dataset_bins", "tokenization", "training"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing config sections: {missing}")
    return config


def _category_for_mode(mode: str, dataset_bins: dict[str, list[str]]) -> str:
    matches = [category for category, modes in dataset_bins.items() if mode in modes]
    if len(matches) != 1:
        raise ValueError(f"response_mode {mode!r} must map to exactly one dataset bin, found {matches}")
    return matches[0]


def _scan_families(
    source_path: Path, dataset_bins: dict[str, list[str]]
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Validate family identity while allowing a family to span route classes.

    A WildGuard family corresponds to one localized prompt and may deliberately
    include multiple safe response routes. Keeping its members together avoids
    prompt leakage, while language-level family stratification keeps validation
    coverage representative without assuming one route per family.
    """
    families: dict[str, str] = {}
    family_categories: dict[str, set[str]] = defaultdict(set)
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            family_id = record.get("family_id")
            language = record.get("language")
            response_mode = record.get("response_mode")
            if not all(isinstance(value, str) and value for value in (family_id, language, response_mode)):
                raise ValueError(f"line {line_number}: missing family_id, language, or response_mode")
            category = _category_for_mode(response_mode, dataset_bins)
            prior_language = families.setdefault(family_id, language)
            if prior_language != language:
                raise ValueError(
                    f"family {family_id} crosses languages: existing={prior_language}, new={language}"
                )
            family_categories[family_id].add(category)
    if not families:
        raise ValueError("source JSONL contains no families")
    return families, family_categories


def _select_validation_families(families: dict[str, str], validation_fraction: float, seed: str) -> set[str]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    groups: dict[str, list[str]] = defaultdict(list)
    for family_id, language in families.items():
        groups[language].append(family_id)

    selected: set[str] = set()
    for group, family_ids in groups.items():
        family_ids.sort(key=lambda family_id: _stable_rank(family_id, seed))
        if len(family_ids) < 2:
            continue
        count = max(1, int(math.floor(len(family_ids) * validation_fraction)))
        count = min(count, len(family_ids) - 1)
        selected.update(family_ids[:count])
    if not selected:
        raise ValueError("stratified family split selected no validation families")
    return selected


def _ensure_validation_bin_coverage(
    selected: set[str],
    family_categories: Mapping[str, set[str]],
    categories: list[str],
    seed: str,
) -> set[str]:
    """Add deterministic whole families until every configured bin has validation data.

    Families are never split, so this preserves the prompt-leakage invariant.
    The base language split is retained; this only adds a rare-route family
    when language-only sampling happened to omit an entire dataset bin.
    """
    selected = set(selected)
    all_families_for_category = {
        category: {family_id for family_id, present in family_categories.items() if category in present}
        for category in categories
    }
    for category, family_ids in all_families_for_category.items():
        if len(family_ids) < 2:
            raise ValueError(f"dataset bin {category!r} needs at least two distinct families")
        if selected.intersection(family_ids):
            continue
        candidates = sorted(
            family_ids.difference(selected),
            key=lambda family_id: _stable_rank(family_id, f"{seed}\0validation-bin\0{category}"),
        )
        candidate = next(
            (
                family_id
                for family_id in candidates
                if all(
                    len(all_families_for_category[member_category].difference(selected | {family_id})) > 0
                    for member_category in family_categories[family_id]
                )
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"could not select a validation family for dataset bin {category!r}")
        selected.add(candidate)
    for category, family_ids in all_families_for_category.items():
        if not selected.intersection(family_ids):
            raise ValueError(f"validation split lacks dataset bin {category!r}")
        if not family_ids.difference(selected):
            raise ValueError(f"training split lacks dataset bin {category!r}")
    return selected


def _selected_records(
    source_path: Path,
    validation_families: set[str],
    dataset_bins: dict[str, list[str]],
) -> Iterable[dict[str, Any]]:
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record["_sft_split"] = "valid" if record["family_id"] in validation_families else "train"
            record["_sft_category"] = _category_for_mode(record["response_mode"], dataset_bins)
            yield record


def _build_data_args(output_root: Path, categories: list[str], weights: dict[str, float]) -> dict[str, list[str]]:
    if set(categories) != set(weights):
        raise ValueError("training train_weights must name exactly the configured dataset bins")
    train: list[str] = []
    for category in categories:
        weight = weights[category]
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(f"train weight for {category!r} must be positive")
        train.extend([str(weight), f"sft:{output_root / 'train' / category}"])
    valid = [f"sft:{output_root / 'valid' / category}" for category in categories]
    # Megatron constructs all three split blends even when the training run
    # never consumes a test loop. Reuse the already held-out validation bins
    # for that dormant test slot; they remain absent from the training blend.
    return {"train": train, "valid": valid, "test": list(valid)}


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def build_dataset(config_path: Path, output_root: Path, workers: int, max_token_id: int | None = None) -> None:
    config = _load_config(config_path)
    if config["source"].get("split_key") != "family_id":
        raise ValueError("this converter requires source.split_key to be exactly 'family_id'")
    if config["source"].get("stratification") != "language":
        raise ValueError("this converter requires source.stratification to be exactly 'language'")
    if config["tokenization"].get("add_generation_prompt") is not False:
        raise ValueError("SFT conversion requires tokenization.add_generation_prompt=false")
    source_path = Path(config["source"]["jsonl"])
    expected_source_sha = config["source"]["sha256"]
    if _sha256_file(source_path) != expected_source_sha:
        raise ValueError("source JSONL SHA256 does not match the pinned configuration")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    if workers < 1:
        raise ValueError("workers must be positive")

    from megatron.core.datasets import indexed_dataset  # Megatron-LM checkout on PYTHONPATH

    categories = list(config["dataset_bins"])
    families, family_categories = _scan_families(source_path, config["dataset_bins"])
    validation_families = _select_validation_families(
        families,
        float(config["source"]["validation_fraction"]),
        str(config["source"]["split_seed"]),
    )
    if config["source"].get("require_validation_per_bin") is True:
        validation_families = _ensure_validation_bin_coverage(
            validation_families,
            family_categories,
            categories,
            str(config["source"]["split_seed"]),
        )

    stage_root = output_root.with_name(f"{output_root.name}.incomplete-{os.getpid()}")
    if stage_root.exists():
        raise FileExistsError(f"refusing to reuse incomplete staging path: {stage_root}")
    stage_root.mkdir(parents=True)

    tokenizer_path = config["model"]["hf_checkpoint"]
    dtype = None
    builders: dict[tuple[str, str], Any] = {}
    metadata_handles: dict[tuple[str, str], Any] = {}
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    token_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"min": sys.maxsize, "max": 0, "total": 0})

    try:
        probe_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        dtype = indexed_dataset.DType.optimal_dtype(len(probe_tokenizer))
        _as_int_list(getattr(probe_tokenizer, "init_kwargs", {}).get("sft_assistant_begin_sequence"), "sft_assistant_begin_sequence")
        _as_int_list(getattr(probe_tokenizer, "init_kwargs", {}).get("sft_eot_token"), "sft_eot_token")
        del probe_tokenizer

        for split in ("train", "valid"):
            split_root = stage_root / split
            split_root.mkdir()
            for category in categories:
                prefix = split_root / category
                key = (split, category)
                builders[key] = indexed_dataset.IndexedDatasetBuilder(str(prefix) + ".bin", dtype=dtype)
                metadata_handles[key] = (split_root / f"{category}.metadata.jsonl").open("w", encoding="utf-8")

        records = _selected_records(source_path, validation_families, config["dataset_bins"])
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(tokenizer_path, bool(config["tokenization"]["enable_thinking"]), max_token_id),
        ) as executor:
            for record, token_ids in executor.map(_render_and_tokenize, records, chunksize=32):
                split = record.pop("_sft_split")
                category = record.pop("_sft_category")
                key = (split, category)
                doc_index = stats[key]["documents"]
                builders[key].add_document(np.asarray(token_ids), [len(token_ids)])
                metadata = {
                    "document_index": doc_index,
                    "id": record["id"],
                    "family_id": record["family_id"],
                    "language": record["language"],
                    "response_mode": record["response_mode"],
                    "training_role": record["training_role"],
                    "candidate_id": record.get("candidate_id"),
                    "candidate_index": record.get("candidate_index"),
                }
                metadata_handles[key].write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                stats[key]["documents"] += 1
                stats[key]["tokens"] += len(token_ids)
                token_stats[key]["min"] = min(token_stats[key]["min"], len(token_ids))
                token_stats[key]["max"] = max(token_stats[key]["max"], len(token_ids))
                token_stats[key]["total"] += len(token_ids)

        for (split, category), builder in builders.items():
            key = (split, category)
            if stats[key]["documents"] == 0:
                raise ValueError(f"{split}/{category} has no documents")
            builder.finalize(str(stage_root / split / category) + ".idx")
            metadata_handles[key].close()

        data_args = _build_data_args(output_root, categories, config["training"]["train_weights"])
        _write_json(stage_root / "per_split_data_args.json", data_args)
        (stage_root / "packing_prefixes.txt").write_text(
            "".join(f"{output_root / 'train' / category} 1\n" for category in categories),
            encoding="utf-8",
        )
        manifest = {
            "run_name": config["run_name"],
            "source_jsonl": str(source_path),
            "source_sha256": expected_source_sha,
            "config_sha256": _sha256_file(config_path),
            "tokenizer": tokenizer_path,
            "tokenizer_config_sha256": _sha256_file(Path(tokenizer_path) / "tokenizer_config.json"),
            "chat_template_sha256": _sha256_file(Path(tokenizer_path) / "chat_template.jinja"),
            "family_split": {
                "key": config["source"]["split_key"],
                "stratification": config["source"].get("stratification"),
                "seed": config["source"]["split_seed"],
                "validation_fraction": config["source"]["validation_fraction"],
                "families_total": len(families),
                "families_validation": len(validation_families),
                "families_train": len(families) - len(validation_families),
            },
            "dataset_bins": config["dataset_bins"],
            "documents": {
                f"{split}/{category}": {
                    "count": stats[(split, category)]["documents"],
                    "tokens": stats[(split, category)]["tokens"],
                    "min_tokens": token_stats[(split, category)]["min"],
                    "max_tokens": token_stats[(split, category)]["max"],
                    "mean_tokens": token_stats[(split, category)]["total"] / stats[(split, category)]["documents"],
                }
                for split in ("train", "valid")
                for category in categories
            },
            "total_documents": sum(counter["documents"] for counter in stats.values()),
            "status": "complete",
        }
        _write_json(stage_root / "dataset_manifest.json", manifest)
        os.replace(stage_root, output_root)
    except Exception:
        for handle in metadata_handles.values():
            if not handle.closed:
                handle.close()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--max-token-id",
        type=int,
        default=None,
        help="Reject a rendered dialogue that uses a token above this inclusive maximum.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.max_token_id is not None and arguments.max_token_id < 0:
        raise ValueError("--max-token-id must be non-negative")
    build_dataset(arguments.config, arguments.output_root, arguments.workers, arguments.max_token_id)
