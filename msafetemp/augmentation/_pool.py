"""One global sliding-window thread pool over the pending samples of many languages.

A pool per language drains to its slowest sample before the next language starts and
left the serving fleet idle for a large share of the wall clock; chaining every
language's samples through a single window keeps it busy. Each language still writes its
own resumable <out>/<language>/<file>.jsonl; resume is by sample identity (see
msafetemp.common.io.load_done_keys), which is what makes interleaved writes safe.
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from msafetemp.common.io import format_duration, load_done_keys
from msafetemp.common.llm import APICircuitBreaker


def run_language_pool(languages: list[str], load_samples: Callable[[str], list[dict]],
                      sample_key: Callable[[dict], Any], process: Callable[[str, dict], dict],
                      out_base: Path, filename: str, workers: int,
                      new_counts: Callable[[], dict], tally: Callable[[dict, dict], None],
                      summary: Callable[[str, dict, float], None],
                      checkpoint_every: int = 25,
                      keep: Callable[[dict], bool] | None = None) -> None:
    """`keep(record) -> False` leaves a record unwritten, so a resubmit retries it."""
    states = []
    for lang in languages:
        samples = load_samples(lang)
        out_dir = out_base / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / filename
        done = load_done_keys(out_file, sample_key)
        pending = [s for s in samples if sample_key(s) not in done]
        progress_file = out_dir / "progress.json"
        counts = new_counts()
        if progress_file.exists():
            counts.update(json.loads(progress_file.read_text()))
        counts["processed"] = len(samples) - len(pending)
        states.append({"language": lang, "out_file": out_file, "progress_file": progress_file,
                       "pending": pending, "remaining": len(pending), "total": len(samples),
                       "counts": counts, "fh": None, "first_done_at": None})

    pending_total = sum(s["remaining"] for s in states)
    print(f"global pool: {pending_total} samples pending across {len(states)} language(s) "
          f"({sum(s['total'] for s in states)} total), workers={workers}")

    def _save(state):
        state["progress_file"].write_text(json.dumps(state["counts"]))

    def _finish(state, pool_start):
        if state["fh"] is not None:
            state["fh"].close()
            state["fh"] = None
        _save(state)
        summary(state["language"], state["counts"],
                time.time() - (state["first_done_at"] or pool_start))

    pool_start = time.time()
    for s in states:
        if s["remaining"] == 0:
            _finish(s, pool_start)
    if pending_total == 0:
        return

    sample_iter = ((s, sample) for s in states for sample in s["pending"])
    write_lock = threading.Lock()
    completed = 0
    window = max(workers * 2, 1)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            in_flight = {pool.submit(process, st["language"], sm): st
                         for st, sm in itertools.islice(sample_iter, window)}
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    state = in_flight.pop(fut)
                    try:
                        record = fut.result()
                    except APICircuitBreaker as e:
                        print(f"\n[FATAL] API circuit breaker tripped: {e}. Saving progress and aborting.")
                        for s in states:
                            _save(s)
                        sys.exit(1)
                    completed += 1
                    if state["first_done_at"] is None:
                        state["first_done_at"] = time.time()
                    state["counts"]["processed"] += 1
                    state["remaining"] -= 1
                    tally(record, state["counts"])
                    if keep is None or keep(record):
                        with write_lock:
                            if state["fh"] is None:
                                state["fh"] = open(state["out_file"], "a", encoding="utf-8")
                            state["fh"].write(json.dumps(record, ensure_ascii=False) + "\n")
                            state["fh"].flush()
                    if state["counts"]["processed"] % checkpoint_every == 0:
                        _save(state)
                    if state["remaining"] == 0:
                        print()
                        _finish(state, pool_start)
                    elapsed = time.time() - pool_start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = format_duration((pending_total - completed) / rate) if rate > 0 else "?"
                    print(f"\r[global] {completed}/{pending_total}  in_flight={len(in_flight)}  "
                          f"rate={rate:.1f}/s  eta={eta}   ", end="", flush=True)
                    nxt = next(sample_iter, None)
                    if nxt is not None:
                        in_flight[pool.submit(process, nxt[0]["language"], nxt[1])] = nxt[0]
    finally:
        for s in states:
            if s["fh"] is not None:
                s["fh"].close()
                s["fh"] = None
    print()
