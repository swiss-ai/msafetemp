"""Generate a draft cultural card by sending the rendered card generation prompt to an
OpenAI-compatible endpoint and converting the returned Python module to JSON.

The released cards were produced from exactly this prompt; the model reply was then
checked manually for coherence and validity of every term and sanity-checked by an
independent model before use. A draft written by this tool is NOT a
verified card: review it before running post-hoc localization with it.

    python -m msafetemp.cultural_cards.generate --language german \\
        --base_url $GEN_URL --model google/gemma-4-31B-it --out drafts/german.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.llm import chat, make_client
from msafetemp.cultural_cards import LANGUAGE_REGISTRY
from msafetemp.cultural_cards.module_to_json import build_card, parse_card_module, write_card
from msafetemp.cultural_cards.render_prompts import render

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_module(reply: str) -> str:
    """The prompt asks for the module inside a single ```python block."""
    m = _CODE_BLOCK.search(reply)
    return m.group(1) if m else reply


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=[l for l in LANGUAGE_REGISTRY if l != "english"])
    p.add_argument("--base_url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api_key", default=None)
    p.add_argument("--max_tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--attempts", type=int, default=3,
                   help="resend while the reply is not a parseable card module")
    p.add_argument("--out", required=True, help="JSON output path for the draft card")
    p.add_argument("--raw_out", default=None, help="optionally keep the raw model reply")
    args = p.parse_args()

    client = make_client(args.base_url, resolve_api_key(args.api_key), 4)
    prompt = render(args.language)
    last_error = None
    for attempt in range(1, args.attempts + 1):
        reply = chat(client, args.model, [{"role": "user", "content": prompt}],
                     max_tokens=args.max_tokens, temperature=args.temperature)
        if args.raw_out:
            Path(args.raw_out).write_text(reply, encoding="utf-8")
        try:
            parsed = parse_card_module(extract_module(reply))
        except (SyntaxError, ValueError) as e:
            last_error = e
            print(f"attempt {attempt}: unparseable card ({e}); retrying")
            continue
        out = write_card(build_card(args.language, parsed), args.out)
        n = sum(len(v) for v in parsed["taxonomy_supplements"].values())
        print(f"wrote draft card {out} ({n} entities). Review it before use.")
        return
    raise SystemExit(f"no parseable card after {args.attempts} attempts: {last_error}")


if __name__ == "__main__":
    main()
