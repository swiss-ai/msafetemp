"""Wrap dataset prompts in jailbreak wrappers by literal insertion (no generation).

Every wrapped prompt is `fill_slot(wrapper_template_in_that_language, prompt)`: the
prompt is inserted verbatim into the wrapper's {{ prompt }} slot(s). Wrapper choice is a
deterministic SHA-256 partial Fisher-Yates draw without replacement (rejection sampling,
no modulo bias), keyed by a sampling key, a seed and a per-source key, so it is
independent of input order. Two modes were used:

  per_source    k distinct wrappers per source prompt, drawn from ALL English wrapper
                ids; a draw whose translated form is missing or did not pass is recorded
                as a blocked combination, never resampled. Used for the matched
                multi-wrapper evaluation (k=10, key literal-wrapper-sampling-v1,
                seed 20260913, source key = [track, split, id]).
  per_language  a per-language quota of wrapped rows (default 1,000), k adaptive in
                [min_k, max_k] = [5, 10] from the number of available payloads, drawn only
                from the language's usable wrapper forms; payloads are sampled first
                (key "<language>:payload"), then k wrappers per payload
                ("<language>:wrappers:<payload id>"). Used for the wrapped SFT / DPO rows
                (key ph60-wrapper-sampling-v1, seed 20260922, train-source prompts only).

Payload input: a JSONL with id, language, prompt (or text) and optional provenance
fields (track, split, category/subset, group_id, original_id, ls_primary), or a dataset
built by build_dataset_*.py via --dataset/--categories/--splits.

    python -m msafetemp.augmentation.wrappers.assemble --mode per_language \\
        --bank $D/wrappers/wrapper_bank.jsonl --dataset $D/wildguard/hf_audited \\
        --categories posthoc --splits train --out $D/wrappers/wrapped_posthoc_train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from msafetemp.common.io import read_jsonl

SLOTS = re.compile(r"\{\{\s*prompt\s*\}\}")
SENTINEL = "\x00LITERAL_PAYLOAD_SENTINEL\x00"
SAMPLING_KEYS = {"per_source": "literal-wrapper-sampling-v1", "per_language": "ph60-wrapper-sampling-v1"}
DEFAULT_SEEDS = {"per_source": 20260913, "per_language": 20260922}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scaffold(template: str) -> str:
    return template.replace("{% raw %}", "").replace("{% endraw %}", "").strip()


def fill_slot(template: str, payload: str) -> str:
    """Literal insertion, identical semantics to corpus.fill_slot."""
    return SLOTS.sub(lambda match: payload, scaffold(template))


def slot_count(template: str) -> int:
    if SENTINEL in template:
        raise ValueError("Reserved sentinel in wrapper")
    count = len(SLOTS.findall(scaffold(template)))
    rendered = fill_slot(template, SENTINEL)
    if count < 1 or rendered.count(SENTINEL) != count or "{{" in rendered or "{%" in rendered:
        raise ValueError("Unsupported wrapper template or missing payload slot")
    return count


def sample_ids(ids: list[str], sampling_key: str, seed: int, source_key: str, count: int) -> list[str]:
    """SHA-256 partial Fisher-Yates over sorted unique ids, independent of input order."""
    if ids != sorted(set(ids)) or not 0 <= count <= len(ids):
        raise ValueError("Sampling requires sorted unique IDs and an available count")
    prefix = canonical([sampling_key, seed, source_key])
    swaps, output, counter = {}, [], 0
    for remaining in range(len(ids), len(ids) - count, -1):
        ceiling = (1 << 256) - (1 << 256) % remaining
        while True:
            value = int.from_bytes(hashlib.sha256(prefix + counter.to_bytes(8, "big")).digest(), "big")
            counter += 1
            if value < ceiling:
                break
        position = value % remaining
        output.append(ids[swaps.get(position, position)])
        swaps[position] = swaps.get(remaining - 1, remaining - 1)
    return output


def load_bank(path: str) -> tuple[dict, dict]:
    """(originals: wrapper_id -> English template, usable: language -> {wrapper_id: template}).
    English forms are the originals; other languages only `pass` forms with a valid slot."""
    originals, usable = {}, {}
    for row in read_jsonl(path):
        status = row.get("translation_status")
        template = row.get("template")
        if status == "original" and row.get("language") == "english":
            originals[row["wrapper_id"]] = template
        if status not in ("original", "pass") or not isinstance(template, str) or not template.strip():
            continue
        try:
            slot_count(template)
        except ValueError:
            continue
        usable.setdefault(row["language"], {})[row["wrapper_id"]] = template
    return originals, usable


def load_payloads(args) -> list[dict]:
    if args.payloads:
        rows = []
        for r in read_jsonl(args.payloads):
            r = dict(r)
            r.setdefault("prompt", r.get("text"))
            rows.append(r)
        return rows
    from datasets import load_from_disk
    data = load_from_disk(args.dataset)
    cats = set(args.categories) if args.categories else None
    rows = []
    for split in args.splits:
        d = data[split]
        if cats is not None:
            d = d.filter(lambda c: c in cats, input_columns=["category"])
        for r in d.to_list():
            rows.append({"id": r["id"], "prompt": r["text"], "language": r["language"],
                         "track": args.track, "split": r.get("split") or split,
                         "subset": r["category"], "family_id": r.get("group_id"),
                         "group_id": r.get("group_id"), "original_id": r.get("original_id"),
                         "ls_primary": r.get("ls_primary")})
    return rows


def assemble_per_source(payloads, originals, usable, seed, k, key):
    ids = sorted(originals)
    for row in payloads:
        source_key = json.dumps([row.get("track"), row.get("split"), row["id"]],
                                ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for index, wid in enumerate(sample_ids(ids, key, seed, source_key, k)):
            base = {"id": sha256_text(json.dumps(["literal-assembly-v1", seed, source_key, wid],
                                                 ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                    "source_id": row["id"], "language": row["language"], "wrapper_id": wid,
                    "wrapper_sample_index": index, "sampling_seed": seed,
                    "track": row.get("track"), "split": row.get("split"),
                    "source_subset": row.get("subset")}
            template = usable.get(row["language"], {}).get(wid)
            if template is None:
                yield {**base, "status": "blocked", "reason": "wrapper_form_missing_or_not_pass"}
                continue
            prompt = fill_slot(template, row["prompt"])
            yield {**base, "status": "assembled", "prompt": prompt, "counterpart_prompt": row["prompt"],
                   "prompt_sha256": sha256_text(prompt), "original_id": row.get("original_id"),
                   "source_group_id": row.get("group_id"), "ls_primary": row.get("ls_primary")}


def assemble_per_language(payloads, usable, seed, key, per_language, min_k, max_k, subset):
    by_lang = {}
    for r in payloads:
        by_lang.setdefault(r["language"], []).append(r)
    for language in sorted(by_lang):
        bank = usable.get(language)
        if not bank:
            print(f"  {language}: no wrappers in bank, skipped")
            continue
        rows = {r["id"]: r for r in by_lang[language]}
        ids = sorted(rows)
        wrapper_ids = sorted(bank)
        # k adapts to payload availability, then the payload count follows.
        k = min(max_k, max(min_k, math.ceil(per_language / max(len(ids), 1))))
        k = min(k, len(wrapper_ids))
        needed = min(len(ids), math.ceil(per_language / k))
        chosen = sample_ids(ids, key, seed, f"{language}:payload", needed)
        pairs = [(pid, wid) for pid in chosen
                 for wid in sample_ids(wrapper_ids, key, seed, f"{language}:wrappers:{pid}", k)]
        for pid, wid in pairs[:per_language]:
            payload = rows[pid]
            prompt = fill_slot(bank[wid], payload["prompt"])
            record = {"arm": "wrapped", "subset": subset or payload.get("subset"),
                      "language": language, "track": payload.get("track"),
                      "split": payload.get("split"), "wrapper_id": wid, "prompt": prompt,
                      "prompt_sha256": sha256_text(prompt), "counterpart_prompt": payload["prompt"],
                      "counterpart_id": pid, "source_family_id": payload.get("family_id"),
                      "source_group_id": payload.get("group_id"),
                      "original_id": payload.get("original_id"), "ls_primary": payload.get("ls_primary")}
            record["id"] = sha256_text(canonical([wid, pid, record["prompt_sha256"]]).decode())
            yield record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=sorted(SAMPLING_KEYS), required=True)
    p.add_argument("--bank", required=True, help="wrapper_bank.jsonl from translate.py")
    p.add_argument("--payloads", default=None, help="payload JSONL (id, language, prompt|text, ...)")
    p.add_argument("--dataset", default=None, help="or: HF dataset dir from build_dataset_*.py")
    p.add_argument("--track", default="wildguard", help="track label for --dataset rows")
    p.add_argument("--categories", nargs="+", default=None, help="--dataset category filter")
    p.add_argument("--splits", nargs="+", default=["train"], help="--dataset splits")
    p.add_argument("--seed", type=int, default=None, help="default: 20260913 per_source, 20260922 per_language")
    p.add_argument("--sampling_key", default=None, help="default per mode (see module docstring)")
    p.add_argument("--k", type=int, default=10, help="per_source: wrappers per prompt")
    p.add_argument("--per_language", type=int, default=1000, help="per_language: rows per language")
    p.add_argument("--min_k", type=int, default=5)
    p.add_argument("--max_k", type=int, default=10)
    p.add_argument("--subset", default=None, help="per_language: subset label to record")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not args.payloads and not args.dataset:
        raise SystemExit("pass --payloads or --dataset")

    seed = DEFAULT_SEEDS[args.mode] if args.seed is None else args.seed
    key = args.sampling_key or SAMPLING_KEYS[args.mode]
    originals, usable = load_bank(args.bank)
    payloads = load_payloads(args)
    print(f"bank: {len(originals)} originals, {len(usable)} languages; payloads: {len(payloads)}")
    if args.mode == "per_source":
        gen = assemble_per_source(payloads, originals, usable, seed, args.k, key)
    else:
        gen = assemble_per_language(payloads, usable, seed, key, args.per_language,
                                    args.min_k, args.max_k, args.subset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = blocked = 0
    with open(out, "x", encoding="utf-8") as f:
        for rec in gen:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            blocked += rec.get("status") == "blocked"
    print(f"wrote {n} rows ({blocked} blocked) -> {out}")


if __name__ == "__main__":
    main()
