"""SFT response generation for dataset prompts (refusal design v2 + benign answers).

Every dataset row is a PROMPT with no response. This tool writes the response an aligned
assistant should give it, so the pair can be trained on:

    harmful / posthoc / wrapped harmful  -> a v2 REFUSAL: decline the harmful part with a
                                           reason, then help with the legitimate need
                                           (prompts/refusal_v2.py)
    benign, class direct                 -> a full, genuinely helpful answer
    benign, class bounded                -> a substantive answer scoped not to transfer
                                           capability (prompts/benign_response.py)

Generation is native per language (the model replies IN the row's language). Every
response is graded by an independent judge whose per-criterion booleans are combined into
a verdict in code (`verdict()`), and a candidate refusal must additionally pass the
evaluation harm judge (it must not contain harmful content). Rejected responses are
regenerated up to --attempts times; what still fails is written with accepted=false and a
reject_reason. Benign rows whose PROMPT the judge finds not actually answerable
(`request_actually_unsafe`) are rejected rather than answered.

Released settings: generator gemma-4-31B-it (temperature 0.8), judge gpt-oss-120b
(temperature 0), --attempts 2, --seed 111, dataset = the audited WildGuard train split
(restricted to an id manifest), refusal design v2.

    python -m msafetemp.augmentation.generate_responses \\
        --dataset $D/wildguard/hf_audited --splits train --ids ids.txt \\
        --gen_url $GEN_URL --gen_model google/gemma-4-31B-it \\
        --judge_url $JUDGE_URL --judge_model openai/gpt-oss-120b \\
        --output $D/wildguard/responses

Any other prompt set with the same row schema (id, group_id, original_id, language,
category, split, branch, ls_primary, ls_high_level, text, benign_class) can be answered
by passing it as --rows_jsonl.

--gen_url / --judge_url accept comma-separated per-replica URLs; calls go to the least
loaded replica. Each language writes a resumable <output>/<language>/responses.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import format_duration, read_jsonl
from msafetemp.common.llm import (
    APICircuitBreaker, ClientSpec, LeastLoaded, check_model, make_client, parse_json_object,
    split_urls,
)
from msafetemp.augmentation.prompts import benign_response as PB
from msafetemp.augmentation.prompts import refusal_v2 as P2
# The evaluation's harm judge: a v2 refusal carries real content, so it must also pass the
# judge that scores the trained models.
from msafetemp.evaluation.judge_prompts import HARM_JUDGE_SYSTEM, HARM_JUDGE_USER

_BENIGN_MAX_TOKENS = 2048
_JUDGE_MAX_TOKENS = 8192
_EVAL_JUDGE_MAX_TOKENS = 16384
_GEN_TEMPERATURE = 0.8
_VALID_STANCES = {"full_compliance", "full_refusal", "partial_refusal"}

# Language names the generator recognises; anything else is title-cased ("german" -> "German").
LANG_NAME = {
    "chinese_simplified": "Simplified Chinese",
    "swiss_german": "Swiss German (Schweizerdeutsch)",
    "portuguese_brazilian": "Brazilian Portuguese",
}

_BENIGN_BOOLS = ["answers", "refuses_outright", "announces_limits", "transferable_harm",
                 "target_language", "meta_leak", "natural", "request_actually_unsafe"]


def lang_name(code: str) -> str:
    return LANG_NAME.get(code, code.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Deterministic pre-judge checks
# ---------------------------------------------------------------------------

_SCAFFOLD_RE = re.compile(
    r"^\s*(here'?s?\s+(is\s+)?(the|a|my)\b|sure[,!]|certainly[,!]|of course[,!]"
    r"|okay[,!]|\*\*?(reply|response|answer|refusal)\*\*?\s*:)", re.I)


def _degenerate(text: str) -> bool:
    """Catch a generation loop before spending a judge call on it. Conservative: must not
    fire on ordinary repetition like a numbered list."""
    t = text.strip()
    if len(t) < 15:
        return True
    if re.search(r"([.…。!?,\-_*=~])\1{9,}", t):
        return True
    words = t.split()
    if len(words) >= 40:
        if Counter(words).most_common(1)[0][1] / len(words) > 0.34:
            return True
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams and Counter(grams).most_common(1)[0][1] >= 5:
            return True
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if len(lines) >= 8 and len(set(lines)) <= len(lines) // 3:
        return True
    return False


def pre_checks(text: str) -> str:
    """Return a reject_reason, or "" if the response is worth a judge call. (v2 lengths
    are word budgets enforced by max_tokens; no sentence-count check applies.)"""
    if not text or not text.strip():
        return "empty"
    if _degenerate(text):
        return "degenerate"
    if _SCAFFOLD_RE.match(text):
        return "scaffolding"
    if text.strip().startswith("```") and text.strip().endswith("```"):
        return "scaffolding"
    return ""


# ---------------------------------------------------------------------------
# Generation + judging
# ---------------------------------------------------------------------------

def kind_of(row: dict) -> str:
    return "benign" if row["category"] == "benign" else "refusal"


def build_generation(row: dict, rng: random.Random):
    """(messages, max_tokens, meta) for one row; the sampled shape / length / opener live here."""
    lang = lang_name(row["language"])
    if kind_of(row) == "benign":
        cls = row.get("benign_class") or "direct"
        system = (PB.BENIGN_DIRECT_SYSTEM if cls == "direct" else PB.BENIGN_BOUNDED_SYSTEM)
        system = system.format(language=lang)
        meta = {"kind": "benign", "benign_class": cls, "shape": "", "length": "", "opener": ""}
        max_tokens = _BENIGN_MAX_TOKENS
    else:
        shape = rng.choices(list(P2.SHAPE_WEIGHTS), weights=list(P2.SHAPE_WEIGHTS.values()))[0]
        length = rng.choice(P2.SHAPE_LENGTHS[shape])
        opener = rng.choice(list(P2.OPENERS))
        system = P2.SYSTEM.format(language=lang, shape=P2.SHAPES[shape],
                                  length=P2.LENGTHS[length][0], opener=P2.OPENERS[opener])
        meta = {"kind": "refusal", "benign_class": "", "shape": shape, "length": length,
                "opener": opener}
        max_tokens = P2.LENGTHS[length][1]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": row["text"]}]
    return messages, max_tokens, meta


async def generate(gen_rr, model, row: dict, rng: random.Random) -> dict:
    messages, max_tokens, meta = build_generation(row, rng)
    out = await gen_rr.call(model, messages, max_tokens, _GEN_TEMPERATURE)
    return {**meta, "response": (out["content"] or "").strip(), "truncated": out["truncated"]}


def _parse_judge(kind: str, raw: str):
    parsed = parse_json_object(raw)
    if not parsed:
        return None
    out = {}
    for key in (P2.JUDGE_BOOLS if kind == "refusal" else _BENIGN_BOOLS):
        if not isinstance(parsed.get(key), bool):
            return None
        out[key] = parsed[key]
    out["reasoning"] = str(parsed.get("reasoning", ""))[:600]
    return out


async def judge(judge_rr, model, kind, language, request, reply, attempts=3):
    """Plain resend on corrupted / short output under heavy concurrency; each resend goes
    to the least loaded replica."""
    template = P2.JUDGE if kind == "refusal" else PB.BENIGN_JUDGE
    messages = [{"role": "user", "content": template.format(
        language=language, request=request[:6000], reply=reply[:6000])}]
    for _ in range(attempts):
        try:
            out = await judge_rr.call(model, messages, _JUDGE_MAX_TOKENS, 0.0)
        except Exception:
            continue
        result = _parse_judge(kind, out["content"])
        if result:
            return result
    return None


def _validate_harm_judge(parsed):
    if not parsed:
        return None
    if parsed.get("response_stance") not in _VALID_STANCES:
        return None
    if not isinstance(parsed.get("is_harmful_content"), bool):
        return None
    return parsed


async def eval_harm_judge(judge_rr, model, request, reply, attempts=3):
    messages = [{"role": "system", "content": HARM_JUDGE_SYSTEM},
                {"role": "user", "content": HARM_JUDGE_USER.format(prompt=request, response=reply)}]
    for _ in range(attempts):
        try:
            out = await judge_rr.call(model, messages, _EVAL_JUDGE_MAX_TOKENS, 0.0)
        except Exception:
            continue
        parsed = _validate_harm_judge(parse_json_object(out["content"]))
        if parsed:
            return parsed
    return None


def verdict(kind: str, meta: dict, j: dict) -> str:
    """The single place an accept / reject decision is made: "" accepts, else the first
    failing criterion."""
    if kind == "refusal":
        return P2.verdict(meta["shape"], j)
    if j["request_actually_unsafe"]: return "request_unsafe"
    if j["refuses_outright"]:        return "refused"
    if j["announces_limits"]:        return "announces_limits"
    if j["transferable_harm"]:       return "transferable_harm"
    if not j["answers"]:             return "not_answered"
    if not j["target_language"]:     return "wrong_language"
    if not j["natural"]:             return "unnatural"
    if j["meta_leak"]:               return "meta_leak"
    return ""


ROW_FIELDS = ("id", "group_id", "original_id", "language", "category", "split", "branch",
              "ls_primary", "ls_high_level")


async def process_row(row, gen_rr, gen_model, judge_rr, judge_model, attempts):
    kind = kind_of(row)
    language = lang_name(row["language"])
    rng = random.Random()
    record = {k: row.get(k) for k in ROW_FIELDS}
    record["prompt"] = row["text"]
    last = None
    for attempt in range(1, attempts + 1):
        try:
            gen = await generate(gen_rr, gen_model, row, rng)
        except Exception as e:
            # The generator is unreachable for this row; kept so the circuit breaker sees it.
            last = {"kind": kind, "benign_class": (row.get("benign_class") or "direct")
                    if kind == "benign" else "", "shape": "", "length": "", "opener": "",
                    "response": "", "judge": None, "reject_reason": "gen_failed",
                    "attempts": attempt, "error": str(e)[:300]}
            continue
        # A reply cut off at max_tokens is never worth a judge call.
        truncated = gen.pop("truncated")
        reason = "truncated" if truncated else pre_checks(gen["response"])
        judgement = None
        if not reason:
            judgement = await judge(judge_rr, judge_model, kind, language, row["text"], gen["response"])
            reason = "judge_unparsed" if judgement is None else verdict(kind, gen, judgement)
        eval_harm = None
        if not reason and kind == "refusal":
            eval_harm = await eval_harm_judge(judge_rr, judge_model, row["text"], gen["response"])
            if eval_harm is None:
                reason = "eval_judge_unparsed"
            elif eval_harm["is_harmful_content"]:
                reason = "eval_harmful"
        last = {**gen, "judge": judgement, "reject_reason": reason, "attempts": attempt}
        if eval_harm is not None:
            last["eval_harm"] = {k: eval_harm.get(k) for k in
                                 ("response_stance", "is_harmful_content", "reasoning_harm")}
        if not reason:
            break
    return {**record, **last, "accepted": not last["reject_reason"]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_rows(dataset_dir, splits, languages, categories, max_per_group, n, seed, ids=None):
    """Select rows on the light columns first, then materialise only the selected rows.
    `ids` (an explicit manifest) overrides the language / category / group filters."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_dir)
    rng = random.Random(seed)
    picked = []
    for split in splits:
        d = ds[split]
        light = d.select_columns(["language", "category", "group_id", "benign_class"]).to_dict()
        keep_l = set(languages) if languages else None
        keep_c = set(categories) if categories else None
        if ids is not None:
            row_id = d.select_columns(["id"]).to_dict()["id"]
            idx = [i for i, r in enumerate(row_id) if r in ids]
        else:
            idx = [i for i, (l, c) in enumerate(zip(light["language"], light["category"]))
                   if (keep_l is None or l in keep_l) and (keep_c is None or c in keep_c)]
        if max_per_group and ids is None:
            # Cap variants per (group, category, benign_class) so no template dominates.
            rng.shuffle(idx)
            seen = Counter()
            capped = []
            for i in idx:
                key = (light["group_id"][i], light["category"][i], light["benign_class"][i] or "")
                if seen[key] >= max_per_group:
                    continue
                seen[key] += 1
                capped.append(i)
            idx = capped
        picked.extend((split, i) for i in idx)
    if n and n < len(picked):
        picked = rng.sample(picked, n)
    rows = []
    for split in splits:
        sel = sorted(i for s_, i in picked if s_ == split)
        if sel:
            rows.extend(ds[split].select(sel).to_list())
    rows.sort(key=lambda r: r["id"])
    return rows


def _load_done(path: Path) -> set:
    """Row ids already answered; gen_failed rows are retried on the next start."""
    done = set()
    if path.exists():
        for r in read_jsonl(path, skip_bad=True):
            if r.get("reject_reason") != "gen_failed":
                done.add(r["id"])
    return done


def _new_counts():
    return {"processed": 0, "accepted": 0, "by_kind": {}, "reject": {}, "shape": {},
            "length": {}, "opener": {}, "attempts": {}}


def _tally(rec, counts):
    counts["processed"] += 1
    k = rec["kind"] if rec["kind"] == "refusal" else f"benign_{rec['benign_class']}"
    slot = counts["by_kind"].setdefault(k, {"n": 0, "ok": 0})
    slot["n"] += 1
    if rec["accepted"]:
        counts["accepted"] += 1
        slot["ok"] += 1
        if rec["shape"]:
            for key in ("shape", "length", "opener"):
                counts[key][rec[key]] = counts[key].get(rec[key], 0) + 1
    else:
        counts["reject"][rec["reject_reason"]] = counts["reject"].get(rec["reject_reason"], 0) + 1
    counts["attempts"][str(rec["attempts"])] = counts["attempts"].get(str(rec["attempts"]), 0) + 1


def print_summary(counts, elapsed):
    n = max(counts["processed"], 1)
    print(f"\nsummary  elapsed={format_duration(elapsed)}")
    print(f"  processed {counts['processed']:>7}   accepted {counts['accepted']:>7}"
          f"  ({counts['accepted'] / n:.1%})")
    for k, v in sorted(counts["by_kind"].items()):
        print(f"    {k:<20} {v['ok']:>7}/{v['n']:<7} {v['ok'] / max(v['n'], 1):>6.1%}")
    for r, c in sorted(counts["reject"].items(), key=lambda x: -x[1]):
        print(f"    reject {r:<20} {c:>7}  ({c / n:.1%})")
    for key in ("shape", "length", "opener"):
        if counts[key]:
            print(f"  accepted refusal {key}: " + "  ".join(f"{k}={v}" for k, v in sorted(counts[key].items())))


async def _run(pending, handles, gen_spec, judge_spec, args):
    """Global coroutine pool over every language. Returns (counts, aborted)."""
    gen_rr = LeastLoaded(gen_spec, args.max_inflight_per_replica)
    judge_rr = LeastLoaded(judge_spec, args.max_inflight_per_replica)
    row_iter = iter(pending)
    grand = _new_counts()
    prog = {"done": 0, "last_print": 0.0, "consec_fail": 0}
    abort_after = max(50, args.workers)   # this many consecutive gen_failed rows => fleet down
    start = time.time()

    async def worker():
        while True:
            try:
                row = next(row_iter)
            except StopIteration:
                return
            rec = await process_row(row, gen_rr, args.gen_model, judge_rr, args.judge_model, args.attempts)
            handles[rec["language"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
            _tally(rec, grand)
            prog["done"] += 1
            if rec["reject_reason"] == "gen_failed":
                prog["consec_fail"] += 1
                if prog["consec_fail"] >= abort_after:
                    raise APICircuitBreaker(f"{prog['consec_fail']} consecutive generation failures; "
                                            f"last error: {rec.get('error', '')}")
            else:
                prog["consec_fail"] = 0
            if prog["done"] % args.checkpoint_every == 0:
                for h in handles.values():
                    h.flush()
            now = time.time()
            if now - prog["last_print"] >= 0.5 or prog["done"] == len(pending):
                prog["last_print"] = now
                rate = prog["done"] / max(now - start, 1e-9)
                eta = format_duration((len(pending) - prog["done"]) / rate) if rate > 0 else "?"
                print(f"\r  {prog['done']}/{len(pending)}  acc={grand['accepted']}  rate={rate:.1f}/s  "
                      f"eta={eta}  inflight gen={gen_rr.inflight()} judge={judge_rr.inflight()}   ",
                      end="", flush=True)

    tasks = [asyncio.create_task(worker()) for _ in range(max(1, args.workers))]
    aborted = False
    try:
        await asyncio.gather(*tasks)
    except APICircuitBreaker as e:
        for t in tasks:
            t.cancel()
        print(f"\n\nABORTED: {e}\nPartial results are saved; rerun to resume.")
        aborted = True
    finally:
        await gen_rr.aclose()
        await judge_rr.aclose()
    return grand, aborted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="HF dataset dir (save_to_disk)")
    ap.add_argument("--rows_jsonl", default=None,
                    help="JSONL of rows instead of --dataset (also used as a cache of the "
                         "materialised manifest rows: written on first use if --dataset is given)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gen_url", required=True, help="one URL or comma-separated per-replica URLs")
    ap.add_argument("--gen_model", required=True)
    ap.add_argument("--judge_url", required=True)
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--languages", nargs="+", default=None)
    ap.add_argument("--categories", nargs="+", default=None, help="harmful / benign / posthoc")
    ap.add_argument("--max_per_group", type=int, default=None,
                    help="cap responses per (group_id, category, benign_class)")
    ap.add_argument("--ids", default=None,
                    help="file of row ids, one per line; overrides --languages / --categories / "
                         "--max_per_group (--n still subsamples it)")
    ap.add_argument("--n", type=int, default=None, help="random subsample of N rows (debug)")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=400, help="concurrent coroutines")
    ap.add_argument("--request_timeout", type=float, default=900.0)
    ap.add_argument("--max_inflight_per_replica", type=int, default=192)
    ap.add_argument("--checkpoint_every", type=int, default=200)
    args = ap.parse_args()

    gen_urls, judge_urls = split_urls(args.gen_url), split_urls(args.judge_url)
    if not gen_urls or not judge_urls:
        raise SystemExit("--gen_url / --judge_url resolved to no URLs")
    key = resolve_api_key(args.api_key)

    want = None
    if args.ids:
        want = {l.strip() for l in open(args.ids) if l.strip()}
        print(f"manifest: {args.ids} ({len(want):,} ids)")

    cache = Path(args.rows_jsonl) if args.rows_jsonl else None
    if cache and cache.exists():
        rows = list(read_jsonl(cache))
        if want is not None:
            rows = [r for r in rows if r["id"] in want]
            if args.n:
                random.Random(args.seed).shuffle(rows)
                rows = rows[:args.n]
    else:
        if not args.dataset:
            raise SystemExit("pass --dataset or an existing --rows_jsonl")
        rows = load_rows(args.dataset, args.splits, args.languages, args.categories,
                         args.max_per_group, args.n, args.seed, ids=want)
        if cache:
            tmp = cache.with_suffix(".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(cache)
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    print(f"rows: {len(rows)} across {len(by_lang)} languages   workers={args.workers}")
    check_model(make_client(gen_urls[0], key, 1, timeout=30.0, max_retries=0), args.gen_model, "generator")
    check_model(make_client(judge_urls[0], key, 1, timeout=30.0, max_retries=0), args.judge_model, "judge")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    pending, skipped, handles = [], 0, {}
    for language in sorted(by_lang):
        lang_dir = out_root / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        done = _load_done(lang_dir / "responses.jsonl")
        skipped += len(done)
        pending.extend(r for r in by_lang[language] if r["id"] not in done)
        handles[language] = open(lang_dir / "responses.jsonl", "a", encoding="utf-8")
    print(f"pending: {len(pending)}  (already done: {skipped})")
    if not pending:
        for h in handles.values():
            h.close()
        return
    random.Random(args.seed).shuffle(pending)   # spread languages across the pool evenly

    gen_spec = ClientSpec(gen_urls, key, args.workers, args.request_timeout)
    judge_spec = ClientSpec(judge_urls, key, args.workers, args.request_timeout)
    start = time.time()
    try:
        grand, aborted = asyncio.run(_run(pending, handles, gen_spec, judge_spec, args))
    finally:
        for h in handles.values():
            h.flush()
            h.close()
    print_summary(grand, time.time() - start)
    if aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
