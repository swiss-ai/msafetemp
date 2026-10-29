"""Resumable target generation + Safeguard judging for a fixed JSONL input set.

Used for evaluations where every model receives the exact same frozen inputs, e.g. the
jailbreak-wrapped HarmBench evaluation (wrapped_hb300.py).

Input rows: {"id", "language", "prompt", "kind": "harmful" | "benign", ...}. Rows without `kind`
use --default_kind.

Target protocol: one user message holding the full presented (wrapped) prompt, no system
message, temperature 0, max_tokens 8192, only the final `content` is kept; optional
chat_template_kwargs (Qwen3.5: {"enable_thinking": false}). Up to three attempts per row.

Judge protocol: response cleaned with strip_reasoning_tokens, full presented prompt,
response_judge.judge_with_retry (main, main, clinical; 16,384 tokens; temperature 0).
harmful -> compute_harmful_verdict, attack_success = verdict == unsafe_compliance;
benign  -> compute_benign_verdict, is_over_refusal.
Unlike the arm engine there is NO validity gate here: empty and truncated HTTP-success answers
are judged and stay in the denominator (scored as returned). Failed judgments are recorded as
judge_failed and target errors as unscorable_target; both are unknown, never safe.

Outputs in --out_dir: targets_<kind>.jsonl and judgments_<kind>.jsonl (append-only journals,
resumable by id).
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from msafetemp.evaluation._utils import api_key_from_env, read_jsonl, strip_reasoning_tokens, text_sha
from msafetemp.evaluation.response_judge import (
    HARMFUL_VERDICTS,
    APICircuitBreaker,
    compute_benign_verdict,
    compute_harmful_verdict,
    judge_with_retry,
    make_client,
    split_urls,
)

TARGET_MAX_TOKENS = 8192
TARGET_TEMPERATURE = 0.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_journal(path: Path) -> dict:
    out = {}
    if path.exists():
        for v in read_jsonl(path):
            out[v["id"]] = v
    return out


class Journal:
    """Append-only JSONL writer; drops a torn final line on open."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                path.write_bytes(data[:data.rfind(b"\n") + 1])
        self.f = path.open("ab")
        self.lock = threading.Lock()

    def write(self, rec):
        line = (json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode()
        with self.lock:
            self.f.write(line)
            self.f.flush()


class Pool:
    """Round-robin over endpoints with a per-endpoint concurrency semaphore."""

    def __init__(self, urls, api_key, per_endpoint, timeout):
        self.urls = urls
        self.clients = [make_client(u, api_key, max_connections=per_endpoint, timeout=timeout, max_retries=0)
                        for u in urls]
        self.sems = [threading.BoundedSemaphore(per_endpoint) for _ in urls]
        self.i = 0
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                start = self.i
                self.i = (self.i + 1) % len(self.urls)
            for k in range(len(self.urls)):
                j = (start + k) % len(self.urls)
                if self.sems[j].acquire(blocking=False):
                    return j
            time.sleep(0.05)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="input JSONL file(s)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_key", required=True, help="short key recorded in every output row")
    ap.add_argument("--model_id", default=None, help="canonical model identifier recorded in outputs")
    ap.add_argument("--base_url", default=None, help="target endpoint(s), comma-separated (omit for judge-only)")
    ap.add_argument("--model", default=None, help="served target model name")
    ap.add_argument("--api_key", default=None, help="default: $OPENAI_API_KEY or 'EMPTY'")
    ap.add_argument("--judge_url", required=True, help="judge endpoint(s), comma-separated")
    ap.add_argument("--judge_model", default="openai/gpt-oss-safeguard-120b")
    ap.add_argument("--judge_api_key", default=None)
    ap.add_argument("--kinds", default="harmful,benign")
    ap.add_argument("--default_kind", default="harmful")
    ap.add_argument("--chat_template_kwargs", default="{}")
    ap.add_argument("--target_conc", type=int, default=48, help="concurrent target requests per endpoint")
    ap.add_argument("--judge_conc", type=int, default=4, help="concurrent judge requests per endpoint")
    ap.add_argument("--target_timeout", type=float, default=1800)
    ap.add_argument("--judge_timeout", type=float, default=600)
    ap.add_argument("--max_target_attempts", type=int, default=3)
    ap.add_argument("--done_file", default=None, help="written when every row is judged")
    return ap.parse_args(argv)


def main(argv=None):
    import os
    args = parse_args(argv)
    kinds = args.kinds.split(",")
    tkw = json.loads(args.chat_template_kwargs)
    api_key = api_key_from_env(args.api_key)
    judge_key = args.judge_api_key or os.environ.get("JUDGE_API_KEY") or api_key

    rows = []
    for path in args.inputs:
        for r in read_jsonl(path):
            r.setdefault("kind", args.default_kind)
            if r["kind"] in kinds:
                rows.append(r)
    out_dir = Path(args.out_dir)
    tj = {k: Journal(out_dir / f"targets_{k}.jsonl") for k in kinds}
    jj = {k: Journal(out_dir / f"judgments_{k}.jsonl") for k in kinds}
    targets = {k: read_journal(out_dir / f"targets_{k}.jsonl") for k in kinds}
    judged = {k: read_journal(out_dir / f"judgments_{k}.jsonl") for k in kinds}

    def target_done(r):
        return targets[r["kind"]].get(r["id"], {}).get("target_status") == "target_complete"

    need_target = [r for r in rows if not target_done(r)
                   and targets[r["kind"]].get(r["id"], {}).get("target_attempts", 0) < args.max_target_attempts]
    log(f"{args.model_key}: rows={len(rows)} need_target={len(need_target)}")
    if need_target and not (args.base_url and args.model):
        raise SystemExit("targets missing but no --base_url/--model given")

    jpool = Pool(split_urls(args.judge_url), judge_key, args.judge_conc, args.judge_timeout)
    tpool = Pool(split_urls(args.base_url), api_key, args.target_conc, args.target_timeout) if need_target else None

    stats = {"target_new": 0, "target_err": 0, "judged_new": 0, "judge_failed": 0}
    slock = threading.Lock()

    def bump(k):
        with slock:
            stats[k] += 1

    def base(r):
        return {"id": r["id"], "kind": r["kind"], "language": r.get("language"),
                "model_key": args.model_key, "model_id": args.model_id or args.model}

    def judge_row(r, t):
        response = t["response"]
        cleaned = strip_reasoning_tokens(response)
        while True:
            j = jpool.acquire()
            try:
                parsed, raw = judge_with_retry(jpool.clients[j], args.judge_model, r["prompt"], cleaned,
                                               benign=r["kind"] == "benign")
                break
            except APICircuitBreaker as e:  # endpoint down: do not record, back off and retry
                log("judge circuit breaker:", str(e)[:200])
                time.sleep(60)
            finally:
                jpool.sems[j].release()
        rec = {**base(r), "response_sha256": text_sha(response), "graded_response_sha256": text_sha(cleaned),
               "judge_model": args.judge_model, "judge_raw": raw, "judge_parsed": parsed,
               "target_status": t["target_status"], "response_empty": t.get("response_empty"),
               "truncated": t.get("truncated"), "completed_at": time.time()}
        if parsed is None:
            rec.update(status="judge_failed", verdict=None)
            bump("judge_failed")
        elif r["kind"] == "harmful":
            v = compute_harmful_verdict(parsed["response_stance"], parsed["is_harmful_content"])
            rec.update(status="scored", verdict=v, response_stance=parsed["response_stance"],
                       is_harmful_content=parsed["is_harmful_content"],
                       attack_success=v in HARMFUL_VERDICTS)
        else:
            rec.update(status="scored", verdict=compute_benign_verdict(parsed["is_over_refusal"]),
                       is_over_refusal=parsed["is_over_refusal"])
        jj[r["kind"]].write(rec)
        judged[r["kind"]][r["id"]] = rec
        bump("judged_new")

    def target_row(r):
        attempts = targets[r["kind"]].get(r["id"], {}).get("target_attempts", 0)
        rec = None
        while attempts < args.max_target_attempts:
            attempts += 1
            j = tpool.acquire()
            started = time.time()
            rec = {**base(r), "served_name": args.model, "target_max_tokens": TARGET_MAX_TOKENS,
                   "target_temperature": TARGET_TEMPERATURE, "target_chat_template_kwargs": tkw,
                   "prompt_sha256": text_sha(r["prompt"]), "target_attempts": attempts}
            try:
                kw = dict(model=args.model, messages=[{"role": "user", "content": r["prompt"]}],
                          max_tokens=TARGET_MAX_TOKENS, temperature=TARGET_TEMPERATURE)
                if tkw:
                    kw["extra_body"] = {"chat_template_kwargs": tkw}
                resp = tpool.clients[j].chat.completions.create(**kw)
                ch = resp.choices[0]
                content = ch.message.content or ""
                rec.update(target_status="target_complete", response=content, response_sha256=text_sha(content),
                           finish_reason=ch.finish_reason, truncated=ch.finish_reason == "length",
                           response_empty=not content.strip(),
                           reasoning_content_length=len(getattr(ch.message, "reasoning_content", None) or ""),
                           target_usage=resp.usage.model_dump() if resp.usage else None)
            except Exception as e:
                rec.update(target_status="target_error", response="", response_empty=True, truncated=False,
                           error_type=type(e).__name__, error_detail=str(e)[:500])
            finally:
                tpool.sems[j].release()
            rec["elapsed_seconds"] = time.time() - started
            tj[r["kind"]].write(rec)
            targets[r["kind"]][r["id"]] = rec
            if rec["target_status"] == "target_complete":
                bump("target_new")
                return rec
            bump("target_err")
            time.sleep(min(60, 5 * attempts))
        return rec

    def work(r):
        t = targets[r["kind"]].get(r["id"])
        if not t or t.get("target_status") != "target_complete":
            if tpool is None:
                return
            t = target_row(r)
            if t is None or t.get("target_status") != "target_complete":
                rec = {**base(r), "status": "unscorable_target", "verdict": None,
                       "target_status": "target_error", "completed_at": time.time()}
                jj[r["kind"]].write(rec)
                judged[r["kind"]][r["id"]] = rec
                return
        if r["id"] not in judged[r["kind"]]:
            judge_row(r, t)

    todo = [r for r in rows if r["id"] not in judged[r["kind"]]]
    random.Random(20260925).shuffle(todo)  # mix languages/kinds across endpoints
    width = (len(tpool.urls) * args.target_conc if tpool else 0) + len(jpool.urls) * args.judge_conc + 8
    done = threading.Event()

    def reporter():
        while not done.wait(60):
            with slock:
                log(f"{args.model_key} progress", json.dumps(stats))

    threading.Thread(target=reporter, daemon=True).start()
    with ThreadPoolExecutor(max_workers=width) as ex:
        for f in [ex.submit(work, r) for r in todo]:
            f.result()
    done.set()
    missing = sum(1 for r in rows if r["id"] not in judged[r["kind"]])
    log(f"{args.model_key} finished", json.dumps(stats), "unjudged_rows", missing)
    if args.done_file and missing == 0:
        Path(args.done_file).write_text(json.dumps({"model_key": args.model_key, "stats": stats}) + "\n")


if __name__ == "__main__":
    main()
