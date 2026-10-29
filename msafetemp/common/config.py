"""Environment-driven path and endpoint configuration.

Nothing in the package hard-codes a filesystem location or a serving endpoint. Paths come
from CLI arguments, falling back to these environment variables:

    MSAFETEMP_DATA_DIR    root of intermediate pipeline data (default: ./data)
    MSAFETEMP_OUT_DIR     root of run outputs (default: ./outputs)
    MSAFETEMP_CARDS_DIR   cultural cards directory (default: <repo>/data/cultural_cards)
    MSAFETEMP_API_KEY     API key for the OpenAI-compatible servers (default: "EMPTY";
                          OPENAI_API_KEY is used if set and MSAFETEMP_API_KEY is not)
    HF_HOME               Hugging Face cache (read by `datasets` directly)

Endpoints (OpenAI-compatible, e.g. vLLM / SGLang servers) come from --base-url style
arguments or from GEN_URL / JUDGE_URL in the example scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("MSAFETEMP_DATA_DIR", "data"))


def out_dir() -> Path:
    return Path(os.environ.get("MSAFETEMP_OUT_DIR", "outputs"))


def cards_dir() -> Path:
    return Path(os.environ.get("MSAFETEMP_CARDS_DIR", str(REPO_ROOT / "data" / "cultural_cards")))


def api_key(explicit: str | None = None) -> str:
    """Explicit value > MSAFETEMP_API_KEY > OPENAI_API_KEY > "EMPTY" (local vLLM servers
    accept any key). Never hard-code keys in scripts."""
    if explicit:
        return explicit
    return os.environ.get("MSAFETEMP_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
