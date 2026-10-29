#!/usr/bin/env python3
"""Adapt the MSafeTemp safety SFT rows to the fields the SFT mix and the Megatron prepare step read.

Input: the balanced safety SFT set (one user turn + one generated assistant turn per row, arms
`harmful`, `posthoc`, `benign_direct`, `benign_bounded`; refusals in the "decline, then help"
design, benign answers helpful; refuse:comply balanced 1:1 per language). Each row carries
`group_id` (the language-scoped template family), `arm` and `target_behavior`.

The prepare step needs `family_id` (the leakage-split key, which may never cross languages),
`response_mode` (maps to exactly one dataset bin) and `training_role`. This tool adds those three
keys and leaves every other field untouched.

Validation is fail-closed: a malformed conversation, a duplicate id, an unknown arm or a family
that appears in more than one language aborts the run rather than being dropped.

    python -m msafetemp.training.data.adapt_safety_rows --source train_balanced.jsonl \
        --output $MSAFETEMP_DATA_DIR/sft/safety_rows.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from msafetemp.training._utils import sha256_file

FAMILY_SOURCE = "group_id"
MODE_SOURCE = "arm"
BEHAVIOR_SOURCE = "target_behavior"
KNOWN_ARMS = {"harmful", "posthoc", "benign_direct", "benign_bounded"}
KNOWN_BEHAVIORS = {"refuse", "comply"}


def check_messages(record: dict, line_number: int) -> None:
    messages = record.get("messages")
    ok = (isinstance(messages, list) and len(messages) == 2
          and all(isinstance(m, dict) for m in messages)
          and messages[0].get("role") == "user" and messages[1].get("role") == "assistant"
          and all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages))
    if not ok:
        raise ValueError("line %d (%s): expected exactly one non-empty user/assistant pair"
                         % (line_number, record.get("id", "<no id>")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-source-sha256", help="abort unless the source file hashes to this value")
    args = ap.parse_args()

    if args.output.exists():
        raise FileExistsError("refusing to overwrite %s" % args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(args.source)
    if args.expect_source_sha256 and source_sha != args.expect_source_sha256:
        raise ValueError("source sha256 %s does not match expected %s" % (source_sha, args.expect_source_sha256))

    seen_ids, family_language = set(), {}
    per_arm, per_language, per_behavior = collections.Counter(), collections.Counter(), collections.Counter()
    rows = 0
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with args.source.open(encoding="utf-8") as handle, temporary.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id, language = record.get("id"), record.get("language")
            family, mode, behavior = record.get(FAMILY_SOURCE), record.get(MODE_SOURCE), record.get(BEHAVIOR_SOURCE)
            for name, value in (("id", record_id), ("language", language), (FAMILY_SOURCE, family),
                                (MODE_SOURCE, mode), (BEHAVIOR_SOURCE, behavior)):
                if not isinstance(value, str) or not value:
                    raise ValueError("line %d: missing or non-string %s" % (line_number, name))
            if mode not in KNOWN_ARMS:
                raise ValueError("line %d: unknown arm %r" % (line_number, mode))
            if behavior not in KNOWN_BEHAVIORS:
                raise ValueError("line %d: unknown target_behavior %r" % (line_number, behavior))
            if record_id in seen_ids:
                raise ValueError("line %d: duplicate id %r" % (line_number, record_id))
            previous = family_language.setdefault(family, language)
            if previous != language:
                raise ValueError("family %r crosses languages: %s and %s" % (family, previous, language))
            check_messages(record, line_number)
            seen_ids.add(record_id)
            per_arm[mode] += 1
            per_language[language] += 1
            per_behavior[behavior] += 1
            rows += 1
            record["family_id"] = family
            record["response_mode"] = mode
            record["training_role"] = "%s_%s" % (mode, behavior)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.rename(args.output)

    manifest = {
        "source": str(args.source), "source_sha256": source_sha,
        "output": str(args.output), "output_sha256": sha256_file(args.output),
        "rows": rows, "families": len(family_language), "languages": len(per_language),
        "rows_per_arm": dict(sorted(per_arm.items())),
        "rows_per_target_behavior": dict(sorted(per_behavior.items())),
        "rows_per_language_min": min(per_language.values()), "rows_per_language_max": max(per_language.values()),
        "field_mapping": {"family_id": FAMILY_SOURCE, "response_mode": MODE_SOURCE,
                          "training_role": "%s_%s" % (MODE_SOURCE, BEHAVIOR_SOURCE)},
    }
    args.output.with_name(args.output.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
