"""Small helpers shared by the training-stage tools.

Kept local to `msafetemp.training` so the stage has no hard dependency on the shape of
`msafetemp.common`. Everything here is plain I/O, hashing and OpenAI-compatible client
plumbing; nothing decides a verdict.

Endpoints: every generator, judge and policy is reached through an OpenAI-compatible
chat-completions endpoint (for example `vllm serve <model> --port 8080`). Tools accept a
URL file with one `http://host:port/v1` per line (one per replica) and balance requests
across them client-side.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

# The 43 non-English training languages (the 44 training languages minus English). Used by
# the translation step for the short prompts.
TRAIN_LANGUAGES_NON_EN = [
    "albanian", "amharic", "arabic", "bengali", "bulgarian", "chinese_simplified", "croatian", "czech",
    "danish", "dutch", "estonian", "finnish", "french", "german", "greek", "hindi", "hungarian",
    "indonesian", "italian", "japanese", "korean", "latvian", "lithuanian", "macedonian", "norwegian",
    "persian", "polish", "portuguese_brazilian", "romanian", "russian", "serbian", "slovak", "slovenian",
    "spanish", "swahili", "swedish", "swiss_german", "tagalog", "thai", "turkish", "ukrainian",
    "vietnamese", "yoruba",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_overwrite(path: str | os.PathLike) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def env_path(name: str, default: str | None = None) -> str | None:
    """Path from an environment variable (MSAFETEMP_DATA_DIR, MSAFETEMP_OUT_DIR, ...)."""
    value = os.environ.get(name, default)
    return os.path.expanduser(value) if value else value


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def read_urls(spec: str | None) -> list[str]:
    """A URL file (one .../v1 URL per line) or a comma-separated URL list -> [url, ...]."""
    if not spec:
        return []
    if os.path.isfile(spec):
        text = Path(spec).read_text()
        return [u.strip() for u in text.replace(",", "\n").splitlines() if u.strip()]
    return [u.strip() for u in spec.split(",") if u.strip()]


def simple_clients(urls: list[str], api_key: str = "EMPTY", max_connections: int = 300,
                   timeout: float = 600.0, max_retries: int = 3):
    """One AsyncOpenAI client per replica URL (callers index them round-robin)."""
    import httpx
    from openai import AsyncOpenAI
    return [AsyncOpenAI(base_url=u, api_key=api_key, max_retries=max_retries,
                        http_client=httpx.AsyncClient(limits=httpx.Limits(max_connections=max_connections),
                                                      timeout=timeout))
            for u in urls]


class ClientSpec:
    """URL list + connection settings, resolved into live AsyncOpenAI clients inside the
    event loop (an httpx.AsyncClient binds to the running loop)."""

    def __init__(self, urls: list[str], api_key: str, max_conns: int, timeout: float):
        self.urls, self.api_key, self.max_conns, self.timeout = urls, api_key, max_conns, timeout


class LeastLoaded:
    """Least-in-flight replica selection with a per-replica cap, exposed as `run(fn)` so the
    caller can make any request shape (for example `n=k` sampling). Blind round robin piles
    long calls onto a few replicas; this keeps every replica equally busy."""

    def __init__(self, spec: ClientSpec, cap: int):
        import httpx
        from openai import AsyncOpenAI
        self._clients = [
            AsyncOpenAI(base_url=u, api_key=spec.api_key, max_retries=0,
                        http_client=httpx.AsyncClient(
                            limits=httpx.Limits(max_connections=spec.max_conns,
                                                max_keepalive_connections=spec.max_conns),
                            # connect=60: many coroutines opening sockets against a busy fleet
                            # can wait well over 10 s in the accept backlog.
                            timeout=httpx.Timeout(spec.timeout, connect=60.0)))
            for u in spec.urls]
        self._inflight = [0] * len(self._clients)
        self._cap = max(1, cap)
        self._freed = asyncio.Condition()

    async def _acquire(self) -> int:
        async with self._freed:
            while True:
                i = min(range(len(self._clients)), key=self._inflight.__getitem__)
                if self._inflight[i] < self._cap:
                    break
                await self._freed.wait()
            self._inflight[i] += 1
        return i

    async def _release(self, i: int) -> None:
        async with self._freed:
            self._inflight[i] -= 1
            self._freed.notify()

    async def run(self, fn):
        i = await self._acquire()
        try:
            return await fn(self._clients[i])
        finally:
            await self._release(i)

    def inflight(self) -> int:
        return sum(self._inflight)

    async def aclose(self) -> None:
        for c in self._clients:
            try:
                await c.close()
            except Exception:
                pass


class APICircuitBreaker(BaseException):
    """Raised to stop a run when an endpoint keeps failing (not caught by `except Exception`)."""


# ---------------------------------------------------------------------------
# Response cleaning and JSON extraction
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CHANNEL_PREFIX_RE = re.compile(
    r"(?:<\|start\|>\w+\s*)?<\|channel\|>[^|<]*\s*(?:<\|constrain\|>[^|<]*\s*)?<\|message\|>",
)
_BANG_RUN_RE = re.compile(r"!{10,}")


def strip_reasoning_tokens(text: str) -> str:
    """Remove model-internal reasoning/routing tokens (gpt-oss harmony channels, <think>)."""
    if "<|end|>" in text:
        text = text.split("<|end|>")[-1]
    text = _CHANNEL_PREFIX_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    return text.strip()


def _is_bang_corrupted(obj: Any) -> bool:
    """gpt-oss occasionally emits a run of '!' inside a string and then closes the JSON
    validly; such output parses but is garbage, so it is rejected."""
    if isinstance(obj, str):
        return bool(_BANG_RUN_RE.search(obj))
    if isinstance(obj, dict):
        return any(_is_bang_corrupted(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_is_bang_corrupted(v) for v in obj)
    return False


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Outermost JSON object in a noisy string (bracket-balanced scan), or None."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw or "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth, in_string, escape_next = 0, False, False
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
                try:
                    obj = json.loads(cleaned[start:i + 1])
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
