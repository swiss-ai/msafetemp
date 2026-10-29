"""Prompt-side audit of the WildGuard benign set, step 1 of 3.

Finds benign rows where the harmful -> harmless substitution FAILED and harmful
instruction survived alongside the (intended) jailbreak wrapper. The judge answers
independent per-criterion booleans; `_verdict()` combines them here. The released
selection rule is applied afterwards by audit_select.py, and audit_reclassify.py moves the
selected rows to category `benign_audit_rejected` (nothing is deleted).

Input rows: {id, split, language, benign_class, text}. Export them from the built dataset
with --export_from_dataset, or pass an existing JSONL with --infile.

    python -m msafetemp.augmentation.audit_benign \\
        --export_from_dataset $D/wildguard/hf --infile $D/wildguard/audit/benign_all.jsonl \\
        --outfile $D/wildguard/audit/audit_full.jsonl \\
        --judge_url $JUDGE_URL --judge_model openai/gpt-oss-120b --reasoning_effort high

Released configuration: gpt-oss-120b, reasoning_effort high, temperature 0,
max_tokens 16384. Resumable: rows with a verdict other than error/unparsed are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.request

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import read_jsonl
from msafetemp.common.llm import AsyncRR, ClientSpec, split_urls
from msafetemp.augmentation.prompts.benign_audit import BOOLS, QUOTED, SYSTEM, USER_TMPL


def _parse(raw):
    if not raw:
        return None
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        o = json.loads(s[i:j + 1])
    except Exception:
        return None
    if not all(k in o for k in BOOLS):
        return None
    v = {k: bool(o[k]) for k in BOOLS}
    quotes = {k: str(o.get(k + "_quote", "") or "")[:400] for k in QUOTED}
    return (v | {"reasoning": str(o.get("reasoning", ""))[:600],
                 "deliverable": str(o.get("deliverable", ""))[:200],
                 "swapped_in": str(o.get("swapped_in", ""))[:200],
                 "quotes": quotes})


def _verdict(v, cur_class):
    """drop            harmful instruction survived the swap, or nothing safe can be said
       relabel_bounded safe request, but `direct` understates it
       keep            fine as labelled
    (This per-row verdict is recorded for analysis; the released selection is
    audit_select.py.)"""
    if v is None:
        return "unparsed"
    # A worked harmful specimen in the prompt text itself cannot be removed by care in the
    # reply, so the class does not matter.
    if v["embeds_harmful_example"] or v["leftover_technique_annotation"]:
        return "drop"
    # harmful_as_written is class-dependent: on `bounded` rows it mostly fires on
    # answerable-with-limits requests, which is what `bounded` means.
    if v["harmful_as_written"] and cur_class != "bounded":
        return "drop"
    if cur_class == "bounded":
        return "keep"
    tainted = (v["covert_toward_third_party"] or v["bypasses_consent"]
               or v["conceals_harm_from_someone"])
    if tainted:
        return "drop"
    if v["needs_bounded"]:
        return "relabel_bounded"
    return "keep"


class _Breaker:
    """Trip after N consecutive failures: that is an outage, not a bad row."""

    def __init__(self, limit=50):
        self.limit, self.streak, self.tripped = limit, 0, False

    def ok(self):
        self.streak = 0

    def fail(self):
        self.streak += 1
        if self.streak >= self.limit:
            self.tripped = True
        return self.tripped


async def _one(sem, rr, model, row, max_tokens, effort=None, breaker=None, retries=3):
    if breaker is not None and breaker.tripped:
        return row | {"judge": None, "verdict": "error", "error": "circuit breaker open"}
    async with sem:
        for a in range(retries):
            try:
                kw = {}
                if effort:
                    kw["extra_body"] = {"reasoning_effort": effort}
                r = await rr.next().chat.completions.create(
                    model=model, max_completion_tokens=max_tokens, temperature=0.0, **kw,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": USER_TMPL.format(
                                  text=row["text"], benign_class=row.get("benign_class") or "direct",
                                  language=row.get("language", "?"))}])
                fr = r.choices[0].finish_reason
                v = _parse(r.choices[0].message.content)
                if v is not None:
                    if breaker is not None:
                        breaker.ok()
                    return row | {"judge": v, "finish_reason": fr,
                                  "verdict": _verdict(v, row.get("benign_class"))}
            except Exception as e:
                if a == retries - 1:
                    if breaker is not None and breaker.fail():
                        print(f"[audit] CIRCUIT BREAKER: {breaker.limit} consecutive failures, "
                              "the endpoint is down. Stopping; rerun to resume.", flush=True)
                    return row | {"judge": None, "verdict": "error", "error": str(e)[:200]}
                await asyncio.sleep(1.5 * (a + 1))
        return row | {"judge": None, "verdict": "unparsed", "finish_reason": "unparsed"}


def export_benign_rows(dataset_dir: str, out_path: str) -> int:
    """Write {id, split, language, benign_class, text} for every benign row of the
    train and test splits of a dataset built by build_dataset_wildguard.py."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_dir)
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for split in ("train", "test"):
            d = ds[split].filter(lambda c: c == "benign", input_columns=["category"])
            for r in d.select_columns(["id", "split", "language", "benign_class", "text"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    return n


async def run(a):
    rows = list(read_jsonl(a.infile))
    if a.ids:
        keep = set(json.load(open(a.ids)))
        rows = [r for r in rows if r["id"] in keep]
    done = set()
    if os.path.exists(a.outfile):
        for o in read_jsonl(a.outfile, skip_bad=True):
            if o.get("verdict") not in ("error", "unparsed"):
                done.add(o["id"])
    rows = [r for r in rows if r["id"] not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[audit] {len(rows)} rows to do ({len(done)} already done)", flush=True)
    if not rows:
        return

    live = []
    for u in split_urls(a.judge_url):
        try:
            with urllib.request.urlopen(u.rstrip("/") + "/models", timeout=20) as r:
                if r.status == 200:
                    live.append(u)
        except Exception:
            pass
    print(f"[audit] {len(live)} replica URL(s) answering", flush=True)
    if not live:
        raise SystemExit("[audit] no judge replica is answering; refusing to start")
    rr = AsyncRR(ClientSpec(live, resolve_api_key(a.api_key), a.workers, 600.0))
    sem = asyncio.Semaphore(a.workers)
    breaker = _Breaker(50)
    t0 = time.time()
    n = 0
    with open(a.outfile, "a", encoding="utf-8") as f:
        tasks = [asyncio.create_task(_one(sem, rr, a.judge_model, r, a.max_tokens,
                                          a.reasoning_effort, breaker)) for r in rows]
        for fut in asyncio.as_completed(tasks):
            o = await fut
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                f.flush()
                print(f"  {n}/{len(rows)}  {n / (time.time() - t0):.1f} rows/s", flush=True)
    await rr.aclose()
    print(f"[audit] done {n} in {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge_url", required=True)
    p.add_argument("--judge_model", required=True)
    p.add_argument("--api_key", default=None)
    p.add_argument("--infile", required=True, help="benign rows JSONL (written first if --export_from_dataset)")
    p.add_argument("--export_from_dataset", default=None, help="WildGuard HF dataset dir to export benign rows from")
    p.add_argument("--outfile", required=True)
    p.add_argument("--ids", default=None, help="JSON list of row ids to restrict to")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=512)
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--reasoning_effort", default="high", help="low | medium | high (gpt-oss)")
    a = p.parse_args()
    if a.export_from_dataset:
        print(f"[audit] exported {export_benign_rows(a.export_from_dataset, a.infile)} benign rows -> {a.infile}")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
