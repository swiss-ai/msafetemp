"""Arm-based refusal / ASR evaluation engine shared by the HarmBench and WildGuard track clients.

A track client only builds a list of (arm_label, rows) and calls run_all_arms(); the global
async work pool, per-arm stats, resume and rankings live here, so both tracks behave
identically.

A row (mk_row) carries: id, arm, eval_kind (harmful | benign), config, language, hb_id (source
family: HarmBench behaviour id or WildGuardMix original id), hb_category, branch (main,
paraphrase, posthoc, benign_main, original), variant_index, paraphrase_index, prompt.

Per row: target call (single user message, no system prompt) -> validity gate
(response_quality) -> response judge (response_judge.ajudge) -> verdict. Results are appended to
<output>/<arm>/eval_results.jsonl (resume by row id); <arm>/stats.json and summary.json hold the
per-arm rates. Rates use only rows with status == "scored".
"""
from __future__ import annotations

import asyncio
import collections
import json
import sys
import time
from pathlib import Path

from msafetemp.evaluation._utils import api_key_from_env, format_duration, load_done_ids
from msafetemp.evaluation.response_judge import (
    HARMFUL_VERDICTS,
    OVER_REFUSAL_VERDICTS,
    APICircuitBreaker,
    AsyncRoundRobin,
    ClientSpec,
    acreate_retry,
    ajudge,
    compute_benign_verdict,
    compute_harmful_verdict,
    make_client,
    split_urls,
)
from msafetemp.evaluation.response_quality import DEGENERATE_REASONS, classify_response


def add_common_eval_args(p):
    """Target / judge / pool / output flags shared by both track clients."""
    p.add_argument("--base_url", required=True,
                   help="OpenAI-compatible endpoint of the target model (e.g. a vLLM server "
                        "http://host:8000/v1). A comma-separated list of replica URLs is "
                        "round-robined client-side.")
    p.add_argument("--api_key", default=None, help="default: $OPENAI_API_KEY or 'EMPTY'")
    p.add_argument("--model", required=True, help="served target model name")
    p.add_argument("--max_tokens", type=int, default=16384,
                   help="target generation budget. We use 8192 for full evaluations and 4096 for fast "
                        "checkpoint evaluations (see configs/evaluation/protocols.yaml)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--chat_template_kwargs", default=None,
                   help='optional JSON passed as extra_body.chat_template_kwargs, e.g. '
                        '\'{"enable_thinking": false}\'. Setting it server-side instead '
                        '(vLLM --default-chat-template-kwargs); both are equivalent.')

    # Degenerate-response handling (see response_quality.py).
    p.add_argument("--regen_degenerate", type=int, default=0, metavar="N",
                   help="on a degenerate target response, resample up to N times before recording "
                        "it as such. 0 (default) records it immediately.")
    p.add_argument("--regen_temperature", type=float, default=0.0,
                   help="temperature for --regen_degenerate resamples")
    p.add_argument("--regen_max_tokens", type=int, default=0, metavar="N",
                   help="token cap for resamples of a no_final_answer row. 0 = keep --max_tokens.")
    p.add_argument("--score_degenerate", action="store_true",
                   help="judge degenerate responses and fold them into the rate (no validity gate).")

    p.add_argument("--judge_url", default=None, help="judge endpoint(s); default: same as --base_url")
    p.add_argument("--judge_api_key", default=None,
                   help="default: $JUDGE_API_KEY, then --api_key")
    p.add_argument("--judge_model", default=None,
                   help="served judge model name (we use openai/gpt-oss-safeguard-120b); "
                        "default: same as --model")

    p.add_argument("--workers", type=int, default=16, help="concurrent rows (async coroutines)")
    p.add_argument("--request_timeout", type=float, default=600.0,
                   help="per-request timeout in seconds for target and judge calls")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--output", required=True, help="output directory")
    return p


def make_eval_clients(args):
    """(target_spec, judge_spec, judge_model). Live clients are created inside the event loop."""
    import os
    tgt_urls = split_urls(args.base_url)
    if not tgt_urls:
        raise SystemExit("[eval] --base_url resolved to no URLs")
    if args.judge_url is None:
        jdg_urls = tgt_urls
    else:
        # Provided-but-empty: refuse to silently grade with the target model as judge.
        jdg_urls = split_urls(args.judge_url)
        if not jdg_urls:
            raise SystemExit("[eval] --judge_url was given but resolved to no URLs")
    api_key = api_key_from_env(args.api_key)
    judge_key = args.judge_api_key or os.environ.get("JUDGE_API_KEY") or api_key
    target_spec = ClientSpec(tgt_urls, api_key, args.workers, args.request_timeout)
    judge_spec = ClientSpec(jdg_urls, judge_key, args.workers, args.request_timeout)
    judge_model = args.judge_model or args.model
    if len(tgt_urls) > 1:
        print(f"[target] round-robin across {len(tgt_urls)} replica URLs")
    if len(jdg_urls) > 1:
        print(f"[judge]  round-robin across {len(jdg_urls)} replica URLs")

    try:
        probe = make_client(tgt_urls[0], api_key, 1, timeout=30.0, max_retries=0)
        names = [m.id for m in probe.models.list().data]
        if args.model not in names:
            print(f"WARNING: {args.model} not in endpoint model list: {names}")
    except Exception as e:
        print(f"WARNING: could not list models at {tgt_urls[0]}: {e}")
    return target_spec, judge_spec, judge_model


def mk_row(arm, config, language, hb_id, hb_category, branch, variant_index, paraphrase_index,
           text, context=None, eval_kind="harmful"):
    prompt = f"{context}\n\n{text}" if context else text
    return {
        "id": f"{arm}_{hb_id}_{branch}_p{paraphrase_index}_v{variant_index}",
        "arm": arm,
        "eval_kind": eval_kind,
        "config": config,
        "language": language,
        "hb_id": hb_id,
        "hb_category": hb_category,
        "branch": branch,
        "variant_index": variant_index,
        "paraphrase_index": paraphrase_index,
        "prompt": prompt,
        "variant_text": text,
        "has_context": bool(context),
    }


def _base_record(row):
    return {k: row[k] for k in (
        "id", "arm", "eval_kind", "config", "language", "hb_id", "hb_category",
        "branch", "variant_index", "paraphrase_index", "prompt", "has_context",
    )}


async def _aprocess_row(row, target_rr, judge_rr, judge_model, args, extra_body):
    prompt = row["prompt"]
    benign = row.get("eval_kind") == "benign"
    try:
        completion = await acreate_retry(
            target_rr.next(), args.model, [{"role": "user", "content": prompt}],
            args.max_tokens, args.temperature, extra_body=extra_body)
    except Exception as e:
        return {**_base_record(row), "status": "target_failed", "error": str(e)}

    response = completion["content"]
    trunc_fields = {"finish_reason": completion["finish_reason"], "truncated": completion["truncated"]}

    # Validity gate before the judge: a degenerate reply carries no stance to judge.
    quality = classify_response(prompt, response,
                                finish_reason=completion["finish_reason"],
                                truncated=completion["truncated"])
    regen_attempts = 0
    if not quality["valid"] and args.regen_degenerate and not args.score_degenerate:
        cap = args.max_tokens
        if quality["reason"] == "no_final_answer" and args.regen_max_tokens:
            cap = args.regen_max_tokens
        for _ in range(args.regen_degenerate):
            regen_attempts += 1
            try:
                completion = await acreate_retry(
                    target_rr.next(), args.model, [{"role": "user", "content": prompt}],
                    cap, max(args.regen_temperature, args.temperature), extra_body=extra_body)
            except Exception:
                break
            response = completion["content"]
            trunc_fields = {"finish_reason": completion["finish_reason"],
                            "truncated": completion["truncated"]}
            quality = classify_response(prompt, response,
                                        finish_reason=completion["finish_reason"],
                                        truncated=completion["truncated"])
            if quality["valid"]:
                break

    if not quality["valid"] and not args.score_degenerate:
        return {**_base_record(row), "status": quality["reason"], "response": response,
                **trunc_fields,
                "repetition": round(quality["repetition"], 3),
                "reasoning_chars": len(completion.get("reasoning") or ""),
                "regen_attempts": regen_attempts}

    parsed, raw = await ajudge(judge_rr, judge_model, prompt, response, benign)
    if not parsed:
        return {**_base_record(row), "status": "judge_failed", "response": response,
                **trunc_fields, "judge_failure_raw": raw[-500:]}

    record = {
        **_base_record(row),
        "status": "scored",
        "response": response,
        **trunc_fields,
        "repetition": round(quality["repetition"], 3),
        "regen_attempts": regen_attempts,
    }
    if benign:
        is_over = parsed["is_over_refusal"]
        record["is_over_refusal"] = is_over
        record["reasoning"] = parsed.get("reasoning", "")
        record["verdict"] = compute_benign_verdict(is_over)
    else:
        stance = parsed["response_stance"]
        is_harmful = parsed["is_harmful_content"]
        record["response_stance"] = stance
        record["reasoning_stance"] = parsed.get("reasoning_stance", "")
        record["is_harmful_content"] = is_harmful
        record["reasoning_harm"] = parsed.get("reasoning_harm", "")
        record["verdict"] = compute_harmful_verdict(stance, is_harmful)
    return record


def _init_arm_state(arm_label, rows, output_dir):
    arm_dir = output_dir / arm_label
    arm_dir.mkdir(parents=True, exist_ok=True)
    out_file = arm_dir / "eval_results.jsonl"
    done_ids = load_done_ids(out_file)
    # Counted against the current row set: a results file may hold ids that are no longer built.
    pending = sum(1 for r in rows if r["id"] not in done_ids)
    return {
        "label": arm_label,
        "dir": arm_dir,
        "out_file": out_file,
        "progress_file": arm_dir / "eval_progress.json",
        "rows": rows,
        "done_ids": done_ids,
        "total": len(rows),
        "completed": len(rows) - pending,
        "pending": pending,
        "fh": None,
    }


def _finish_arm(state, all_stats):
    if state["fh"] is not None:
        state["fh"].close()
        state["fh"] = None
    state["progress_file"].write_text(
        json.dumps({"processed": state["completed"], "total": state["total"]}))
    records = []
    if state["out_file"].exists():
        with open(state["out_file"], encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    stats = summarize_arm(state["label"], records)
    print_arm_summary(stats)
    (state["dir"] / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    all_stats[state["label"]] = stats


# Flush each arm's JSONL every N rows; an unflushed tail is re-run on resume (dedup by row id).
_FLUSH_EVERY = 200


def _pending_rows(states):
    for s in states:
        if s["pending"] == 0:
            continue
        for row in s["rows"]:
            if row["id"] not in s["done_ids"]:
                yield s, row


def run_all_arms(arms, target_spec, judge_spec, judge_model, args, output_dir):
    """Run every arm's rows through one global async pool of args.workers coroutines."""
    output_dir = Path(output_dir)
    states = [_init_arm_state(label, rows, output_dir) for label, rows in arms]
    all_stats = {}

    grand_total = sum(s["total"] for s in states)
    already = sum(s["completed"] for s in states)
    pending_total = sum(s["pending"] for s in states)

    print(f"\n{'=' * 72}")
    print(f"global async pool: {pending_total} prompts across {len(states)} arm(s) "
          f"({grand_total} total, {already} already scored)  workers={args.workers}")
    for s in states:
        if s["completed"] and s["pending"]:
            print(f"  [{s['label']}] resuming, {s['completed']} done, {s['pending']} remaining")

    for s in states:
        if s["pending"] == 0:
            if s["total"] == 0:
                print(f"[{s['label']}] no rows, skipping")
            _finish_arm(s, all_stats)

    if pending_total == 0:
        return [all_stats[label] for label, _ in arms if label in all_stats]

    asyncio.run(_run_async(states, target_spec, judge_spec, judge_model, args,
                           pending_total, all_stats))
    return [all_stats[label] for label, _ in arms if label in all_stats]


async def _run_async(states, target_spec, judge_spec, judge_model, args, pending_total, all_stats):
    target_rr = AsyncRoundRobin(target_spec)
    judge_rr = AsyncRoundRobin(judge_spec)
    extra_body = None
    if getattr(args, "chat_template_kwargs", None):
        extra_body = {"chat_template_kwargs": json.loads(args.chat_template_kwargs)}
    row_iter = _pending_rows(states)
    start_time = time.time()
    prog = {"done": 0, "last_print": 0.0, "consec_fail": 0, "consec_judge_fail": 0}
    abort_after = max(50, args.workers)  # consecutive failures => endpoint down

    def _write(state, record):
        if state["fh"] is None:
            state["fh"] = open(state["out_file"], "a", encoding="utf-8")
            state["since_flush"] = 0
        state["fh"].write(json.dumps(record, ensure_ascii=False) + "\n")
        state["since_flush"] = state.get("since_flush", 0) + 1
        if state["since_flush"] >= _FLUSH_EVERY:
            state["fh"].flush()
            state["since_flush"] = 0

    def _close_all():
        for s in states:
            if s["fh"] is not None:
                try:
                    s["fh"].flush()
                    s["fh"].close()
                except Exception:
                    pass
                s["fh"] = None

    async def worker():
        while True:
            try:
                state, row = next(row_iter)
            except StopIteration:
                return
            record = await _aprocess_row(row, target_rr, judge_rr, judge_model, args, extra_body)
            prog["done"] += 1
            state["completed"] += 1
            state["pending"] -= 1
            _write(state, record)

            if record["status"] == "target_failed":
                prog["consec_fail"] += 1
                if prog["consec_fail"] >= abort_after:
                    raise APICircuitBreaker(
                        f"{prog['consec_fail']} consecutive target failures, target down?")
            else:
                prog["consec_fail"] = 0
            # Without this a dead judge fleet would write judge_failed rows (counted as done) for
            # the whole dataset.
            if record["status"] == "judge_failed":
                prog["consec_judge_fail"] += 1
                if prog["consec_judge_fail"] >= abort_after:
                    raise APICircuitBreaker(
                        f"{prog['consec_judge_fail']} consecutive judge failures, judge down?")
            elif record["status"] == "scored":
                prog["consec_judge_fail"] = 0

            if state["completed"] % args.checkpoint_every == 0:
                state["progress_file"].write_text(
                    json.dumps({"processed": state["completed"], "total": state["total"]}))

            if state["pending"] == 0:
                if state["fh"] is not None:
                    state["fh"].flush()
                print()
                _finish_arm(state, all_stats)

            now = time.time()
            if now - prog["last_print"] >= 0.5:
                prog["last_print"] = now
                rate = prog["done"] / (now - start_time) if now > start_time else 0
                eta = format_duration((pending_total - prog["done"]) / rate) if rate > 0 else "?"
                print(f"\r[global] {prog['done']}/{pending_total} "
                      f"({100 * prog['done'] / pending_total:.1f}%)  "
                      f"arms {len(all_stats)}/{len(states)}  rate={rate:.1f}/s  eta={eta}   ",
                      end="", flush=True)

    tasks = [asyncio.create_task(worker()) for _ in range(max(1, args.workers))]
    try:
        await asyncio.gather(*tasks)
    except APICircuitBreaker as e:
        for t in tasks:
            t.cancel()
        _close_all()
        print(f"\n[FATAL] circuit breaker tripped: {e}")
        print("Partial results are saved; rerun to resume.")
        sys.exit(1)
    finally:
        _close_all()
        await target_rr.aclose()
        await judge_rr.aclose()
    print()


def _rate(rows, verdicts):
    scored = [r for r in rows if r.get("status") == "scored"]
    n = len(scored)
    if n == 0:
        return {"n": 0, "rate": 0.0}
    hits = sum(1 for r in scored if r["verdict"] in verdicts)
    return {"n": n, "rate": hits / n}


def summarize_arm(arm_label, rows):
    """Per-arm stats. ASR (harmful arms) = unsafe_compliance / scored; over-refusal (benign arms)
    = over_refusal / scored. An arm without any scored row reports rate 0.0."""
    total = len(rows)
    eval_kind = next((r.get("eval_kind") for r in rows if r.get("eval_kind")), "harmful")
    benign = eval_kind == "benign"
    fail_verdicts = OVER_REFUSAL_VERDICTS if benign else HARMFUL_VERDICTS
    metric_key = "over_refusal_rate" if benign else "asr"

    status_counts = collections.Counter(r.get("status") for r in rows)
    scored = [r for r in rows if r.get("status") == "scored"]
    verdict_counts = collections.Counter(r["verdict"] for r in scored)

    n = len(scored)
    overall = _rate(rows, fail_verdicts)
    n_degenerate = sum(status_counts.get(k, 0) for k in DEGENERATE_REASONS)

    def _by(field):
        out = {}
        buckets = collections.defaultdict(list)
        for r in scored:
            buckets[r.get(field)].append(r)
        for k, rs in buckets.items():
            s = _rate(rs, fail_verdicts)
            out[k] = {"n": s["n"], metric_key: s["rate"]}
        return out

    stats = {
        "arm": arm_label,
        "eval_kind": eval_kind,
        "total_rows": total,
        "status_counts": dict(status_counts),
        "scored": n,
        "target_failed": status_counts.get("target_failed", 0),
        "judge_failed": status_counts.get("judge_failed", 0),
        "truncated": sum(1 for r in rows if r.get("truncated")),
        # Generation validity is reported beside the safety rate, never folded into it.
        "degenerate": n_degenerate,
        "degenerate_counts": {k: status_counts[k] for k in DEGENERATE_REASONS if status_counts.get(k)},
        "generation_validity": n / (n + n_degenerate) if n + n_degenerate else 0.0,
        "regenerated": sum(1 for r in rows if r.get("regen_attempts")),
        metric_key: overall["rate"],
        "verdict_counts": dict(verdict_counts),
        f"{metric_key}_by_category": _by("hb_category"),
        f"{metric_key}_by_branch": _by("branch"),
    }
    if benign:
        stats["benign_compliance_rate"] = _rate(rows, ("benign_compliance",))["rate"]
    else:
        stance_counts = collections.Counter(r.get("response_stance") for r in scored)
        stats["stance_fractions"] = {s: (stance_counts.get(s, 0) / n if n else 0.0)
                                     for s in ("full_compliance", "partial_refusal", "full_refusal")}
    return stats


def print_arm_summary(stats):
    benign = stats.get("eval_kind") == "benign"
    mkey = "over_refusal_rate" if benign else "asr"
    print(f"\n--- summary: {stats['arm']}  ({stats.get('eval_kind', 'harmful')}) ---")
    trunc = stats.get("truncated", 0)
    warn = ("   [WARNING: >2% of responses hit --max_tokens]"
            if stats["total_rows"] and trunc / stats["total_rows"] > 0.02 else "")
    print(f"  rows={stats['total_rows']}  scored={stats['scored']}  "
          f"target_failed={stats['target_failed']}  judge_failed={stats['judge_failed']}  "
          f"truncated={trunc}{warn}")
    if stats.get("degenerate"):
        gv = stats.get("generation_validity", 1.0)
        warn = "   [WARNING: rate is conditional on a shrinking subset]" if gv < 0.85 else ""
        print(f"  generation validity: {gv:.1%}  ({stats['degenerate']} degenerate excluded: "
              f"{stats.get('degenerate_counts', {})}){warn}")
    if benign:
        print(f"  over-refusal rate: {stats[mkey]:.1%}  (n={stats['scored']})")
    else:
        print(f"  attack success rate (unsafe_compliance): {stats[mkey]:.1%}  (n={stats['scored']})")
    for verdict, c in sorted(stats["verdict_counts"].items(), key=lambda x: -x[1]):
        print(f"    {verdict:<24} {c:>6}  ({c / (stats['scored'] or 1):.1%})")


def print_rankings(all_stats):
    """Harmful, post-hoc and benign arms are different populations and are ranked separately."""
    harmful = [s for s in all_stats if s.get("eval_kind") != "benign" and not s["arm"].startswith("posthoc_")]
    posthoc = [s for s in all_stats if s.get("eval_kind") != "benign" and s["arm"].startswith("posthoc_")]
    benign = [s for s in all_stats if s.get("eval_kind") == "benign"]
    for title, group, key in (("HARMFUL ARMS by ASR", harmful, "asr"),
                              ("POST-HOC ARMS by ASR", posthoc, "asr"),
                              ("BENIGN ARMS by over-refusal", benign, "over_refusal_rate")):
        if not group:
            continue
        print(f"\n{'=' * 72}\n{title} (highest first)")
        for s in sorted(group, key=lambda x: -x[key]):
            print(f"  {s['arm']:<34} {key} {s[key]:>6.1%}  (scored n={s['scored']})")
