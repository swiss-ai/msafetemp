"""Small shared helpers for the evaluation clients.

Kept inside msafetemp.evaluation so the evaluation stage has no dependency on other stages.
Behaviour-relevant pieces (response cleaning, JSON extraction) are shared by every client so
that all tracks are scored the same way.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------------------
# Response cleaning: harmony channel markers (gpt-oss) and <think> blocks (Qwen3 and similar)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CHANNEL_PREFIX_RE = re.compile(
    r"(?:<\|start\|>\w+\s*)?<\|channel\|>[^|<]*\s*(?:<\|constrain\|>[^|<]*\s*)?<\|message\|>",
)


def strip_reasoning_tokens(text: str) -> str:
    """Remove model-internal reasoning/routing tokens from a raw API response."""
    if "<|end|>" in text:
        text = text.split("<|end|>")[-1]
    text = _CHANNEL_PREFIX_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# JSON extraction: bracket-balanced scan so nested objects do not confuse it
# ---------------------------------------------------------------------------

_BANG_RUN_RE = re.compile(r"!{10,}")


def _is_bang_corrupted(obj: Any) -> bool:
    """True if any string value nested in obj contains a long run of '!'.

    gpt-oss-120b occasionally spirals into a run of thousands of '!' inside a string field,
    then recovers and closes the JSON validly, producing parseable but garbage content that a
    plain json.loads() would accept.
    """
    if isinstance(obj, str):
        return bool(_BANG_RUN_RE.search(obj))
    if isinstance(obj, dict):
        return any(_is_bang_corrupted(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_is_bang_corrupted(v) for v in obj)
    return False


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Extract the outermost valid JSON object from a (possibly noisy) string.

    Walks character by character from the first '{' to the matching '}' at depth 0,
    respecting string escapes, and falls back to json.loads on the cleaned string. Rejects
    (returns None) any result that parses but is bang-corrupted, so callers retry instead of
    persisting garbage.
    """
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(cleaned[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start: i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return None if _is_bang_corrupted(obj) else obj
                except json.JSONDecodeError:
                    pass
                break

    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            return None
        return None if _is_bang_corrupted(obj) else obj
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Hashing (canonical JSON digests are used for seeded, outcome-blind selections)
# ---------------------------------------------------------------------------

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    """sha256 of the canonical JSON encoding (selection ranks, row ids)."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------

def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, skipping blank lines and a torn final line."""
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def load_done_ids(out_file: Path) -> set[str]:
    """Row ids already present in a results file (resume is keyed by row id)."""
    done = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
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


def env_path(name: str, default: str | None = None) -> str | None:
    """Path from an environment variable (MSAFETEMP_DATA_DIR, MSAFETEMP_OUT_DIR, ...)."""
    v = os.environ.get(name)
    return v if v else default


def api_key_from_env(explicit: str | None, var: str = "OPENAI_API_KEY") -> str:
    """Key for an OpenAI-compatible server. Local vLLM servers accept any non-empty string."""
    return explicit or os.environ.get(var) or "EMPTY"
