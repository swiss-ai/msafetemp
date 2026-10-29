"""Dual-judge the assembled wrapped prompts: a quality judge (language, naturalness,
meaning preserved) and a safety judge (unsafe request, category match, payload
preserved), both returning strict booleans plus `uncertain` and no reasoning.

Acceptance (recorded per row, never inferred later):
  harmful / posthoc  quality accepted AND unsafe_request AND payload_preserved AND not
                     uncertain. category_match is recorded but not required, because the
                     payload category passed to the judge is the unverified LinguaSafe
                     label of the source seed.
  benign             quality accepted AND NOT unsafe_request AND payload_preserved AND not
                     uncertain (a benign row the judge calls unsafe is a finding, not a pass).
  `accepted_strict` records the judging-time rule: all quality checks, and for harmful
  rows all three safety checks (for benign rows: NOT unsafe_request).

    python -m msafetemp.augmentation.wrappers.judge --rows wrapped.jsonl --arm harmful \\
        --bank $D/wrappers/wrapper_bank.jsonl \\
        --quality_url $QC_URL --quality_model <multilingual quality model> \\
        --safety_url $JUDGE_URL --safety_model openai/gpt-oss-120b --output judged.jsonl

Resumable: row ids already in --output are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import read_jsonl
from msafetemp.common.llm import split_urls
from msafetemp.augmentation.prompts.wrappers import QUALITY_SYSTEM, REQUIRED, SAFETY_SYSTEM


def parse_judgment(role: str, content: str) -> dict:
    """Strict JSON; strings such as 'true' are not booleans."""
    required = REQUIRED[role]
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return {"status": "error", "accepted": False, "error": "invalid_json"}
    expected = set(required) | {"uncertain"}
    if (not isinstance(value, dict) or set(value) != expected
            or any(type(value.get(k)) is not bool for k in expected)):
        return {"status": "error", "accepted": False, "error": "invalid_schema"}
    return {"status": "scored",
            "accepted": not value["uncertain"] and all(value[k] for k in required),
            "checks": value}


def accepted(arm: str, quality: dict, safety: dict, strict: bool = False) -> bool:
    if quality.get("status") != "scored" or safety.get("status") != "scored":
        return False
    if not quality.get("accepted"):
        return False
    s = safety["checks"]
    if s["uncertain"]:
        return False
    if strict:
        if arm == "benign":
            return s["unsafe_request"] is False
        return s["unsafe_request"] and s["category_match"] and s["payload_preserved"]
    if not s["payload_preserved"]:
        return False
    if arm == "benign":
        return s["unsafe_request"] is False
    return s["unsafe_request"] is True


def judge_payload(row: dict, english: dict) -> dict:
    return {
        "target_language": row["language"],
        "source_wrapper_english": english.get(row["wrapper_id"], ""),
        "payload_category": row.get("ls_primary") or "unspecified",
        "payload": row["counterpart_prompt"],
        "final_prompt": row["prompt"],
    }


async def score_one(client, model, system, data, role, max_tokens, attempts=3):
    last = {"status": "error", "accepted": False, "error": "unattempted"}
    for attempt in range(attempts):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
                max_tokens=max_tokens * (attempt + 1), temperature=0.0)
            last = parse_judgment(role, (response.choices[0].message.content or "").strip())
            if last["status"] == "scored":
                return last
        except Exception as error:
            last = {"status": "error", "accepted": False, "error": type(error).__name__}
            await asyncio.sleep(2 * (attempt + 1))
    return last


async def run(args):
    from openai import AsyncOpenAI
    english = {r["wrapper_id"]: r["template"] for r in read_jsonl(args.bank)
               if r.get("translation_status") == "original"}
    rows = [r for r in read_jsonl(args.rows) if r.get("status", "assembled") == "assembled"]
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out, skip_bad=True)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"rows={len(rows)} already_judged={len(done)} to_judge={len(todo)}", flush=True)

    key = resolve_api_key(args.api_key)
    qclients = [AsyncOpenAI(base_url=u, api_key=key, timeout=180, max_retries=0) for u in split_urls(args.quality_url)]
    sclients = [AsyncOpenAI(base_url=u, api_key=key, timeout=180, max_retries=0) for u in split_urls(args.safety_url)]
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counts = {"scored": 0, "accepted": 0, "error": 0}
    started = time.time()
    with open(out, "a", encoding="utf-8") as handle:
        async def work(index, row):
            data = judge_payload(row, english)
            async with sem:
                quality, safety = await asyncio.gather(
                    score_one(qclients[index % len(qclients)], args.quality_model, QUALITY_SYSTEM,
                              data, "quality", args.max_tokens),
                    score_one(sclients[index % len(sclients)], args.safety_model, SAFETY_SYSTEM,
                              data, "safety", args.max_tokens))
            record = {"id": row["id"], "language": row["language"], "arm": args.arm,
                      "wrapper_id": row["wrapper_id"], "counterpart_id": row.get("counterpart_id"),
                      "quality": quality, "safety": safety,
                      "accepted": accepted(args.arm, quality, safety),
                      "accepted_strict": accepted(args.arm, quality, safety, strict=True)}
            async with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                counts["scored"] += 1
                counts["accepted"] += int(record["accepted"])
                counts["error"] += int(quality["status"] != "scored" or safety["status"] != "scored")
                if counts["scored"] % 200 == 0:
                    rate = counts["scored"] / max(time.time() - started, 1e-6)
                    print(f"  judged={counts['scored']} accepted={counts['accepted']} "
                          f"errors={counts['error']} rate={rate:.1f}/s", flush=True)

        await asyncio.gather(*(work(i, r) for i, r in enumerate(todo)))
    print(json.dumps(counts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, help="assemble.py output")
    p.add_argument("--bank", required=True, help="wrapper_bank.jsonl (English originals)")
    p.add_argument("--output", required=True)
    p.add_argument("--arm", choices=["harmful", "benign"], required=True,
                   help="harmful for harmful/posthoc payloads, benign for benign payloads")
    p.add_argument("--quality_url", required=True)
    p.add_argument("--quality_model", required=True)
    p.add_argument("--safety_url", required=True)
    p.add_argument("--safety_model", required=True)
    p.add_argument("--api_key", default=None)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
