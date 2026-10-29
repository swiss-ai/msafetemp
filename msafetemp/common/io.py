"""Small JSON / JSONL helpers used across stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def read_jsonl(path: str | os.PathLike, skip_bad: bool = False) -> Iterator[dict]:
    """Yield one dict per non-empty line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not skip_bad:
                    raise


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict], mode: str = "w") -> int:
    n = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def atomic_write_json(path: str | os.PathLike, value: Any, indent: int | None = 2) -> None:
    """Write-then-rename so a partially written file is never read as complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)


def load_done_keys(path: str | os.PathLike, key: Callable[[dict], Any]) -> set:
    """Identity keys of the records already written to `path` (resume by identity, not by
    row offset: results are written in completion order, so an interrupted run leaves an
    arbitrary subset on disk)."""
    done = set()
    if os.path.exists(path):
        for rec in read_jsonl(path, skip_bad=True):
            done.add(key(rec))
    return done


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
