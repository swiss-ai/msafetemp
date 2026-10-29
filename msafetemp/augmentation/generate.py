"""Run the template pipeline over the seed prompts of one track, per target language.

    python -m msafetemp.augmentation.generate --track wildguard \\
        --base_url $GEN_URL --model google/gemma-4-31B-it \\
        --judge_url $JUDGE_URL --judge_model openai/gpt-oss-120b \\
        --languages german --n 10000 --min_prompt_length 800 \\
        --num_variants 5 --paraphrasing 2 --prefixes 5 --suffixes 5 \\
        --output $MSAFETEMP_DATA_DIR/wildguard/raw

Settings of the released runs:
  wildguard  seed 111, n 10,000, min_prompt_length 800, 5 fills x (main + 2 paraphrases),
             5 prefixes + 5 suffixes, max_retries 3. The released WildGuard benign pairs
             do NOT come from this run's benign branch: they were generated with
             benign.py over the 2,500 canonical seeds (use --benign 0 here to save calls).
  harmbench  seed 111, all 200 standard behaviors, 5 fills x (main + 2 paraphrases),
             max_retries 3, no affixes. The released benign pairs come from a second,
             separate run of this script with --num_variants 0 --paraphrasing 0
             --benign 5 into its own output directory (it re-extracts and re-translates the
             templates); affixes were added afterwards with affixes.py.
  Both runs use context_source=none: no cultural card is injected into translation or
  generation; localization happens post hoc (posthoc.py).

Output per language (resumable by sample identity):
  wildguard  <output>/<language>/regen.jsonl
  harmbench  <output>/standard_<language>/regen.jsonl
plus progress.json (counts) and processed_keys.txt (resume ledger) in the same directory.

Record fields: original_prompt (HarmBench: also behavior / hb_id / hb_category),
translated_template, translation / translation_raw / translation_judge /
translation_attempts, language, context_source, status (extractor_failed | benign |
judge_failed | reject | translator_failed | pass), attempts, judge_failure_raw,
extraction / judge1 (English side), variants / rejected_variants, paraphrases (each with
template, judge1, variants, rejected_variants), prefixes, suffixes, benign_variants /
rejected_benign_variants. A pass sample with no accepted variant at all is not written.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import format_duration
from msafetemp.common.llm import APICircuitBreaker, check_model, get_prompt_usage_counts, make_client
from msafetemp.augmentation.multilingual import TRACKS, run_multilingual_pipeline
from msafetemp.augmentation.sources import load_harmbench_samples, load_wildguard_samples
from msafetemp.augmentation.stages import print_extraction, print_judge1, print_variants
from msafetemp.cultural_cards import SUPPORTED_LANGUAGES

CONTEXT_SOURCE = "none"
_KNOWN_JUDGE2_CRITERIA = {"HARMFUL", "LANGUAGE_CONSISTENCY", "COHERENT", "FRAMING_INTACT"}
_KNOWN_BENIGN_CRITERIA = {"TOXIC", "INCOHERENT", "LANGUAGE_INCONSISTENT", "FRAME_CHANGED", "UNJUDGED"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--base_url", required=True, help="generator endpoint(s), comma-separated replicas allowed")
    p.add_argument("--model", required=True, help="served generator model (extractor, translator, generators)")
    p.add_argument("--judge_url", default=None, help="judge endpoint(s); default: --base_url")
    p.add_argument("--judge_model", default=None, help="served judge model; default: --model")
    p.add_argument("--api_key", default=None, help="default: $MSAFETEMP_API_KEY / $OPENAI_API_KEY / EMPTY")
    p.add_argument("--languages", nargs="+", default=None, choices=SUPPORTED_LANGUAGES,
                   help="target languages ('english' skips translation). Default: all.")
    p.add_argument("--n", type=int, default=None, help="number of seed prompts (default: all)")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--min_prompt_length", type=int, default=None, help="wildguard only")
    p.add_argument("--cache_dir", default=None, help="HF datasets cache dir")
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--num_variants", type=int, default=5)
    p.add_argument("--paraphrasing", type=int, default=2)
    p.add_argument("--prefixes", type=int, default=0)
    p.add_argument("--suffixes", type=int, default=0)
    p.add_argument("--benign", type=int, default=5, help="benign exact-fill variants per template (0 disables)")
    p.add_argument("--skip_generator", action="store_true")
    p.add_argument("--skip_judge2", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--output", required=True, help="base output directory")
    p.add_argument("--debug", action="store_true", help="verbose per-sample output (n defaults to 6)")
    args = p.parse_args()
    if not args.languages:
        args.languages = list(SUPPORTED_LANGUAGES)
    if args.debug and args.n is None:
        args.n = 6
    return args


def _sample_key(track: str, sample: dict) -> str:
    return str(sample["id"]) if track == "harmbench" else sample.get("prompt") or ""


def _sample_prompt(track: str, sample: dict) -> str:
    return sample.get("behavior") if track == "harmbench" else sample.get("prompt")


def _split_variants(pairs, judge_key):
    ok = [{"text": v, judge_key: j} for v, j in (pairs or []) if j is None or j.get("verdict") == "pass"]
    bad = [{"text": v, judge_key: j} for v, j in (pairs or []) if j is not None and j.get("verdict") != "pass"]
    return ok, bad


def build_record(track: str, sample: dict, result: dict, language: str) -> dict:
    variants, rejected = _split_variants(result.get("variants"), "judge2")
    benign, rejected_benign = _split_variants(result.get("benign_variants"), "benign_judge")
    paraphrases = []
    for para in result.get("paraphrases") or []:
        pv, pr = _split_variants(para.get("variants"), "judge2")
        paraphrases.append({"template": para["template"], "judge1": para["judge1"],
                            "judge_failure_raw": para.get("judge_failure_raw"),
                            "variants": pv, "rejected_variants": pr})
    if track == "harmbench":
        head = {"hb_id": sample.get("id"), "behavior": sample.get("behavior"),
                "original_prompt": sample.get("behavior"),
                "hb_category": sample.get("hb_category"),
                "hb_functional_category": sample.get("hb_functional_category")}
    else:
        head = {"original_prompt": result.get("original_prompt") or sample.get("prompt") or ""}
    rec = {
        **head,
        "translated_template": result.get("translated_template"),
        "translation": result.get("translation"),
        "translation_raw": result.get("translation_raw"),
        "translation_judge": result.get("translation_judge"),
        "translation_attempts": result.get("translation_attempts") or [],
        "language": language,
        "context_source": CONTEXT_SOURCE,
    }
    if track == "wildguard":
        rec.update({"wg_harm_label": sample.get("prompt_harm_label"),
                    "wg_subcategory": sample.get("subcategory"),
                    "wg_adversarial": sample.get("adversarial"),
                    "wg_response_refusal": sample.get("response_refusal_label")})
    rec.update({
        "status": result["status"],
        "attempts": result["attempts"],
        "judge_failure_raw": result.get("judge_failure_raw"),
        "extraction": result.get("extraction"),
        "judge1": result.get("judge1"),
        "variants": variants,
        "rejected_variants": rejected,
        "paraphrases": paraphrases,
        "prefixes": result.get("prefixes") or [],
        "suffixes": result.get("suffixes") or [],
        "benign_variants": benign,
        "rejected_benign_variants": rejected_benign,
    })
    return rec


def _tally(counts: dict, rec: dict, args) -> None:
    status = rec["status"]
    counts[status] = counts.get(status, 0) + 1
    counts["processed"] += 1
    groups = [rec["variants"]] + [p["variants"] for p in rec["paraphrases"]]
    rejected = [rec["rejected_variants"]] + [p["rejected_variants"] for p in rec["paraphrases"]]
    counts["variants_total"] += sum(len(g) for g in groups)
    counts["judge2_rejected"] += sum(len(g) for g in rejected)
    for g in rejected:
        for v in g:
            for crit in (v["judge2"] or {}).get("failed_criteria", []):
                if crit in _KNOWN_JUDGE2_CRITERIA:
                    counts["failed_criteria"][crit] = counts["failed_criteria"].get(crit, 0) + 1
    counts["para_judge_failed"] += sum(1 for p in rec["paraphrases"] if p.get("judge_failure_raw"))
    counts["benign_variants_total"] += len(rec["benign_variants"])
    counts["benign_rejected"] += len(rec["rejected_benign_variants"])
    for v in rec["benign_variants"]:
        cls = (v["benign_judge"] or {}).get("class")
        if cls:
            counts[f"benign_{cls}"] = counts.get(f"benign_{cls}", 0) + 1
    for v in rec["rejected_benign_variants"]:
        for crit in (v["benign_judge"] or {}).get("failed_criteria", []):
            if crit in _KNOWN_BENIGN_CRITERIA:
                counts["benign_failed_criteria"][crit] = counts["benign_failed_criteria"].get(crit, 0) + 1
    if status == "pass" and not args.skip_generator:
        counts["variants_possible"] += args.num_variants * (1 + args.paraphrasing)
        counts["benign_variants_possible"] += args.benign


def _new_counts() -> dict:
    return {"processed": 0, "written": 0, "variants_total": 0, "variants_possible": 0,
            "judge2_rejected": 0, "para_judge_failed": 0, "failed_criteria": {},
            "benign_variants_total": 0, "benign_variants_possible": 0, "benign_rejected": 0,
            "benign_failed_criteria": {}, "prompt_usage": {}}


def _print_summary(language: str, counts: dict, elapsed: float) -> None:
    print(f"\nsummary  language={language}  elapsed={format_duration(elapsed)}")
    skip = {"failed_criteria", "benign_failed_criteria", "prompt_usage"}
    for k, v in sorted(counts.items()):
        if k not in skip and isinstance(v, int) and v:
            print(f"  {k:<26} {v:>7}")
    for key in ("failed_criteria", "benign_failed_criteria"):
        if counts.get(key):
            print(f"  {key}: " + ", ".join(f"{c}={n}" for c, n in sorted(counts[key].items(), key=lambda x: -x[1])))
    if counts["variants_possible"]:
        print(f"  harmful yield {counts['variants_total'] / counts['variants_possible']:.1%}")
    if counts["benign_variants_possible"]:
        print(f"  benign yield  {counts['benign_variants_total'] / counts['benign_variants_possible']:.1%}")


def run_language(language, samples, client, judge_client, judge_model, args, out_base: Path):
    sub = f"standard_{language}" if args.track == "harmbench" else language
    out_dir = out_base / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "regen.jsonl"
    keys_file = out_dir / "processed_keys.txt"
    progress_file = out_dir / "progress.json"

    done = set()
    if keys_file.exists():
        done = {json.loads(l) for l in keys_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    pending = [s for s in samples if _sample_key(args.track, s) not in done]
    counts = json.loads(progress_file.read_text()) if progress_file.exists() else _new_counts()
    baseline_usage = counts.get("prompt_usage", {})
    start_usage = get_prompt_usage_counts()
    print(f"\n[{language}] {len(pending)} pending of {len(samples)} -> {out_file}")
    if not pending:
        return

    lock = threading.Lock()
    start = time.time()

    def _write_progress():
        live = get_prompt_usage_counts()
        merged = {}
        for name in set(baseline_usage) | set(live):
            b, l, s = baseline_usage.get(name, {}), live.get(name, {}), start_usage.get(name, {})
            merged[name] = {k: b.get(k, 0) + l.get(k, 0) - s.get(k, 0) for k in ("main", "clinical")}
        counts["prompt_usage"] = merged
        progress_file.write_text(json.dumps(counts))

    def _process(sample):
        prompt = _sample_prompt(args.track, sample)
        if not prompt:
            return sample, None
        result = run_multilingual_pipeline(
            client=client, model=args.model, english_prompt=prompt, target_language=language,
            cultural_context=None, max_retries=args.max_retries, num_variants=args.num_variants,
            skip_generator=args.skip_generator, skip_judge2=args.skip_judge2,
            num_paraphrases=args.paraphrasing, num_prefixes=args.prefixes,
            num_suffixes=args.suffixes, num_benign=args.benign, verbose=args.debug,
            judge_client=judge_client, judge_model=judge_model, track=args.track)
        return sample, result

    with open(out_file, "a", encoding="utf-8") as fout, open(keys_file, "a", encoding="utf-8") as fkeys:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process, s) for s in pending]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    sample, result = fut.result()
                except APICircuitBreaker as e:
                    print(f"\n[FATAL] API circuit breaker tripped: {e}. Saving progress and aborting.")
                    with lock:
                        _write_progress()
                    sys.exit(1)
                with lock:
                    if result is not None:
                        rec = build_record(args.track, sample, result, language)
                        _tally(counts, rec, args)
                        has_output = (rec["status"] != "pass" or rec["variants"]
                                      or any(p["variants"] for p in rec["paraphrases"])
                                      or rec["benign_variants"])
                        if has_output:
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            counts["written"] += 1
                        if args.debug:
                            print(f"\n[{i}/{len(pending)}] status={rec['status']}")
                            if rec.get("extraction"):
                                print_extraction(rec["extraction"])
                            if rec.get("judge1"):
                                print_judge1(rec["judge1"])
                            print_variants([(v["text"], v["judge2"]) for v in rec["variants"] + rec["rejected_variants"]])
                    fkeys.write(json.dumps(_sample_key(args.track, sample)) + "\n")
                    if i % args.checkpoint_every == 0:
                        fout.flush()
                        fkeys.flush()
                        _write_progress()
                    if not args.debug:
                        print(f"\r[{language}] {i}/{len(pending)}", end="", flush=True)
    _write_progress()
    _print_summary(language, counts, time.time() - start)


def main() -> None:
    args = parse_args()
    key = resolve_api_key(args.api_key)
    client = make_client(args.base_url, key, args.workers)
    judge_client = make_client(args.judge_url, key, args.workers) if args.judge_url else None
    judge_model = args.judge_model or args.model
    check_model(client, args.model, "generator")

    if args.track == "wildguard":
        samples = load_wildguard_samples(args.n, args.seed, cache_dir=args.cache_dir,
                                         min_prompt_length=args.min_prompt_length)
        for s in samples:
            s["id"] = s.get("prompt")
    else:
        samples = load_harmbench_samples(args.n, args.seed, cache_dir=args.cache_dir)
    print(f"track={args.track}  seeds={len(samples)}  languages={len(args.languages)}  "
          f"variants={args.num_variants} x (1 + {args.paraphrasing} paraphrases)  "
          f"benign={args.benign}  affixes={args.prefixes}/{args.suffixes}")

    out_base = Path(args.output)
    for language in args.languages:
        run_language(language, samples, client, judge_client, judge_model, args, out_base)


if __name__ == "__main__":
    main()
