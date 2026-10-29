"""OpenAI-compatible client helpers shared by every generation and judging stage.

Two flavours are provided, matching how the pipeline stages were run:

* a synchronous client used from thread pools (template extraction, translation, fills,
  benign pairs, post-hoc localization, taxonomy). ``chat`` wraps each call in a tenacity
  retry (4 attempts, 1-10 s exponential backoff) and a process-wide circuit breaker that
  aborts the run after 5 consecutive fully-exhausted failures;
* an asyncio client (response generation, benign audit) with per-replica round robin and
  a least-loaded scheduler.

Every endpoint argument accepts either one base URL or a comma-separated list of
per-replica URLs. With several URLs, calls are spread client-side across replicas instead
of going through a router whose prefix-cache affinity would pin calls sharing a long
system prompt to a single replica.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


# ---------------------------------------------------------------------------
# Response cleaning and JSON extraction
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CHANNEL_PREFIX_RE = re.compile(
    r"(?:<\|start\|>\w+\s*)?<\|channel\|>[^|<]*\s*(?:<\|constrain\|>[^|<]*\s*)?<\|message\|>",
)


def strip_reasoning_tokens(text: str) -> str:
    """Remove model-internal reasoning / channel tokens from a raw API response
    (harmony-format gpt-oss output, <think> blocks)."""
    if "<|end|>" in text:
        text = text.split("<|end|>")[-1]
    text = _CHANNEL_PREFIX_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    return text.strip()


_BANG_RUN_RE = re.compile(r"!{10,}")


def _is_bang_corrupted(obj: Any) -> bool:
    """True if any string value nested in obj contains a long run of '!'.

    gpt-oss-120b occasionally spirals into a run of thousands of '!' inside a string
    field, then recovers and closes the JSON validly, producing parseable but garbage
    content that a plain json.loads() would accept.
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
    respecting string escapes, then falls back to json.loads on the cleaned string.
    Bang-corrupted results are rejected (None) so callers retry instead of persisting
    garbage.
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
                candidate = cleaned[start : i + 1]
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


def _escape_raw_control_in_strings(s: str) -> str:
    """Escape raw control chars (newline/CR/tab) that appear INSIDE JSON string literals.
    Models often emit multi-line list items (numbered jailbreak steps) with literal
    newlines inside the quotes, which is invalid JSON. Structural whitespace outside
    strings is left untouched, so valid JSON is unaffected."""
    out = []
    in_str = False
    esc = False
    repl = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    for ch in s:
        if esc:
            out.append(ch)
            esc = False
        elif ch == "\\":
            out.append(ch)
            esc = True
        elif ch == '"':
            in_str = not in_str
            out.append(ch)
        elif in_str and ch in repl:
            out.append(repl[ch])
        else:
            out.append(ch)
    return "".join(out)


def parse_string_list(raw: str) -> list[str] | None:
    """Parse a JSON array of strings out of a model reply (None if absent/invalid)."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        return None
    slice_ = cleaned[start:end + 1]
    for candidate in (slice_, _escape_raw_control_in_strings(slice_)):
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    return None


# ---------------------------------------------------------------------------
# Synchronous client (thread pools)
# ---------------------------------------------------------------------------

def split_urls(spec: str | None) -> list[str]:
    """Comma-separated URL list -> [url, ...]."""
    return [u.strip() for u in (spec or "").split(",") if u.strip()]


def make_client(base_url: str, api_key: str, max_connections: int,
                timeout: float | None = None, max_retries: int | None = None):
    """OpenAI client whose httpx pool is sized to the requested concurrency.

    The SDK's default pool (1000 connections, 100 keep-alive) silently caps concurrency
    below large --workers values, so the pool is sized explicitly. `timeout` caps a single
    request (SDK default 600 s otherwise); callers that do their own retrying should also
    pass max_retries=0. A comma-separated `base_url` returns a round-robin client over one
    OpenAI client per replica URL.
    """
    urls = split_urls(base_url)
    if not urls:
        raise ValueError(f"make_client: no URL in {base_url!r}")
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    http_kwargs: dict[str, Any] = {"limits": limits}
    if timeout is not None:
        http_kwargs["timeout"] = httpx.Timeout(timeout, connect=10.0)
    kwargs: dict[str, Any] = {}
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    clients = [OpenAI(base_url=u, api_key=api_key, http_client=httpx.Client(**http_kwargs), **kwargs)
               for u in urls]
    return clients[0] if len(clients) == 1 else RRClient(clients)


class RRClient:
    """Round-robins chat.completions.create across one OpenAI client per replica URL.
    Thread-safe cursor. Each call, including every tenacity retry, advances to the next
    replica, so a dead replica costs one attempt rather than a whole request. Exposes only
    client.chat.completions.create(...) and client.models (first replica)."""

    def __init__(self, clients: list[OpenAI]):
        self._clients = clients
        self._i = 0
        self._lock = threading.Lock()
        self.chat = self
        self.completions = self
        self.models = clients[0].models

    def __len__(self) -> int:
        return len(self._clients)

    def create(self, **kwargs):
        with self._lock:
            c = self._clients[self._i % len(self._clients)]
            self._i += 1
        return c.chat.completions.create(**kwargs)


class APICircuitBreaker(BaseException):
    """Raised after too many consecutive fully-exhausted API failures (endpoint down).
    Derives from BaseException so generic `except Exception` handlers in the stages do
    not swallow it."""


_API_ERROR_LIMIT = 5
_api_error_streak = 0
_api_error_lock = threading.Lock()

_prompt_usage_lock = threading.Lock()
_prompt_usage_counts: dict[str, dict[str, int]] = {}


def track_prompt_usage(name: str, used_clinical: bool) -> None:
    """Tally how often each judge used its main system prompt vs. the clinical fallback
    framing. Process-lifetime only."""
    with _prompt_usage_lock:
        counts = _prompt_usage_counts.setdefault(name, {"main": 0, "clinical": 0})
        counts["clinical" if used_clinical else "main"] += 1


def get_prompt_usage_counts() -> dict[str, dict[str, int]]:
    with _prompt_usage_lock:
        return {k: dict(v) for k, v in _prompt_usage_counts.items()}


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _chat_raw_full(client, model: str, messages: list[dict], max_tokens: int,
                   temperature: float) -> dict:
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
    )
    choice = resp.choices[0]
    msg = choice.message
    # When the server exposes a separate reasoning channel, msg.content is already just the
    # final-answer channel; otherwise inline reasoning / harmony markers are stripped.
    reasoning = getattr(msg, "reasoning_content", None) or ""
    finish_reason = choice.finish_reason
    return {
        "content": strip_reasoning_tokens(msg.content or ""),
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
    }


def chat_full(client, model: str, messages: list[dict], max_tokens: int,
              temperature: float) -> dict:
    """One chat call with retries + circuit breaker. Returns
    {content, reasoning, finish_reason, truncated}. Sampling uses plain temperature
    (top_p left at the server default of 1.0, no top-k)."""
    global _api_error_streak
    try:
        result = _chat_raw_full(client, model, messages, max_tokens, temperature)
        with _api_error_lock:
            _api_error_streak = 0
        return result
    except APICircuitBreaker:
        raise
    except Exception as e:
        with _api_error_lock:
            _api_error_streak += 1
            streak = _api_error_streak
        if streak >= _API_ERROR_LIMIT:
            raise APICircuitBreaker(
                f"API down: {streak} consecutive fully-exhausted failures, last error: {e}"
            ) from e
        raise


def chat(client, model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Like chat_full but returns only the final-answer text."""
    return chat_full(client, model, messages, max_tokens, temperature)["content"]


def check_model(client, model: str, role: str = "model") -> None:
    """Warn (do not fail) if `model` is not in the endpoint's served model list."""
    try:
        names = [m.id for m in client.models.list().data]
        if model not in names:
            print(f"WARNING [{role}]: '{model}' not in endpoint model list: {names}")
    except Exception as e:
        print(f"WARNING [{role}]: could not list models: {e}")


# ---------------------------------------------------------------------------
# Asynchronous client (asyncio)
# ---------------------------------------------------------------------------

class ClientSpec:
    """URL list + connection settings, resolved into live AsyncOpenAI clients INSIDE the
    event loop (an httpx.AsyncClient binds to the running loop)."""

    def __init__(self, urls: list[str], api_key: str, max_conns: int, timeout: float):
        self.urls = urls
        self.api_key = api_key
        self.max_conns = max_conns
        self.timeout = timeout


class AsyncRR:
    """Round-robins requests across one AsyncOpenAI client per replica URL."""

    def __init__(self, spec: ClientSpec):
        self._clients = [
            AsyncOpenAI(
                base_url=u, api_key=spec.api_key, max_retries=0,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=spec.max_conns,
                                        max_keepalive_connections=spec.max_conns),
                    # connect=60: with many coroutines opening sockets against a busy fleet
                    # the accept backlog can take well over 10 s to reach a new connection.
                    timeout=httpx.Timeout(spec.timeout, connect=60.0)),
            )
            for u in spec.urls
        ]
        self._i = 0

    def next(self):
        c = self._clients[self._i % len(self._clients)]
        self._i += 1
        return c

    async def aclose(self):
        for c in self._clients:
            try:
                await c.close()
            except Exception:
                pass


async def acreate(client, model, messages, max_tokens, temperature) -> dict:
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    choice = resp.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    fr = choice.finish_reason
    return {
        "content": strip_reasoning_tokens(msg.content or ""),
        "reasoning": reasoning,
        "finish_reason": fr,
        "truncated": fr == "length",
    }


async def acreate_retry(client, model, messages, max_tokens, temperature, attempts=4) -> dict:
    """Async equivalent of the sync tenacity wrapper: retry a transient failure up to
    `attempts` times with 1..10 s exponential backoff."""
    delay = 1.0
    for a in range(attempts):
        try:
            return await acreate(client, model, messages, max_tokens, temperature)
        except Exception:
            if a == attempts - 1:
                raise
            await asyncio.sleep(min(delay, 10.0))
            delay *= 2.0
    raise RuntimeError("unreachable")


class LeastLoaded:
    """Per-replica async clients with least-in-flight selection and a per-replica
    concurrency cap.

    Blind round robin gives every replica an equal share of NEW requests regardless of
    how many it is still working through; long judge calls then pile up unevenly and
    overfill some replicas' KV caches. Each call here takes the replica with the fewest
    in-flight requests, and a replica never holds more than `cap` at once; extra calls
    wait client-side for the first free slot on any replica."""

    def __init__(self, spec: ClientSpec, cap: int):
        self._rr = AsyncRR(spec)
        self._clients = self._rr._clients
        self._inflight = [0] * len(self._clients)
        self._cap = max(1, cap)
        self._freed = asyncio.Condition()

    async def call(self, model, messages, max_tokens, temperature):
        async with self._freed:
            while True:
                i = min(range(len(self._clients)), key=self._inflight.__getitem__)
                if self._inflight[i] < self._cap:
                    break
                await self._freed.wait()
            self._inflight[i] += 1
        try:
            return await acreate_retry(self._clients[i], model, messages, max_tokens, temperature)
        finally:
            async with self._freed:
                self._inflight[i] -= 1
                self._freed.notify()

    def inflight(self):
        return list(self._inflight)

    async def aclose(self):
        await self._rr.aclose()
