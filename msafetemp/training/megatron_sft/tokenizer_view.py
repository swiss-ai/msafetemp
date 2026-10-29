#!/usr/bin/env python3
"""Create a run-local Apertus-8B tokenizer view with auditable SFT markers.

Copies the tokenizer files and chat template of the Instruct checkpoint and records the
assistant-begin (id 67) and assistant-end (id 68) marker sequences in tokenizer_config.json, which
the prepare step uses to validate every rendered dialogue. The same view serves the base and the
Instruct run (the base checkpoint is trained with the Instruct chat template).

    python -m msafetemp.training.megatron_sft.tokenizer_view --source $APERTUS_INSTRUCT_DIR \
        --output $WORK_ROOT/tokenizer_sft_view
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from collections.abc import Mapping

from transformers import AutoTokenizer


BEGIN = [67]
END = [68]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        help="Chat-template source; defaults to the tokenizer source.",
    )
    parser.add_argument("--expected-vocab-size", type=int, default=131072)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    template = (args.template or source).resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite tokenizer view %s" % output)
    if args.expected_vocab_size <= 0:
        raise ValueError("expected vocabulary size must be positive")
    missing = [name for name in TOKENIZER_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("missing tokenizer files: %s" % missing)
    if not (template / "chat_template.jinja").is_file():
        raise FileNotFoundError("missing chat template: %s" % (template / "chat_template.jinja"))

    stage = output.with_name(".%s.incomplete-%s" % (output.name, os.getpid()))
    if stage.exists():
        raise FileExistsError("staging tokenizer view already exists: %s" % stage)
    stage.mkdir(parents=True)
    for name in TOKENIZER_FILES:
        shutil.copy2(str(source / name), str(stage / name))
    shutil.copy2(str(template / "chat_template.jinja"), str(stage / "chat_template.jinja"))
    for optional in ("generation_config.json",):
        if (source / optional).is_file():
            shutil.copy2(str(source / optional), str(stage / optional))
    config_path = stage / "tokenizer_config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("tokenizer config must be an object")
    config["sft_assistant_begin_sequence"] = BEGIN
    config["sft_eot_token"] = END
    config["sft_template_source"] = str(source)
    write_json(config_path, config)

    tokenizer = AutoTokenizer.from_pretrained(str(stage), trust_remote_code=False)
    if tokenizer.init_kwargs.get("sft_assistant_begin_sequence") != BEGIN:
        raise ValueError("tokenizer view did not retain the assistant-begin marker")
    if tokenizer.init_kwargs.get("sft_eot_token") != END:
        raise ValueError("tokenizer view did not retain the assistant-end marker")
    messages = [
        {"role": "user", "content": "Reply with exactly one safe word."},
        {"role": "assistant", "content": "Safe."},
    ]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if isinstance(tokens, Mapping):
        if set(tokens).difference({"input_ids", "attention_mask"}) or "input_ids" not in tokens:
            raise ValueError("chat template BatchEncoding has an unexpected shape")
        tokens = tokens["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    if not isinstance(tokens, list):
        raise ValueError("chat template did not produce a token list")
    begin_positions = [index for index in range(len(tokens)) if tokens[index : index + len(BEGIN)] == BEGIN]
    end_positions = [index for index in range(len(tokens)) if tokens[index : index + len(END)] == END]
    if len(begin_positions) != 1 or len(end_positions) != 1 or end_positions[0] <= begin_positions[0]:
        raise ValueError("assistant markers are not uniquely ordered")
    if tokens[-len(END) :] != END or len(tokenizer) != args.expected_vocab_size:
        raise ValueError(
            "unexpected SFT end marker or tokenizer vocabulary: tail=%r vocab=%d expected=%d"
            % (tokens[-len(END) :], len(tokenizer), args.expected_vocab_size)
        )
    write_json(
        stage / "sft_tokenizer_view_manifest.json",
        {
            "source": str(source),
            "template": str(template),
            "source_tokenizer_sha256": sha256_file(source / "tokenizer.json"),
            "template_sha256": sha256_file(template / "chat_template.jinja"),
            "assistant_begin_sequence": BEGIN,
            "assistant_end_sequence": END,
            "vocab_size": len(tokenizer),
        },
    )
    os.replace(str(stage), str(output))


if __name__ == "__main__":
    main()
