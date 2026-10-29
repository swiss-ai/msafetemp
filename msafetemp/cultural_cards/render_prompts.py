"""Render CARD_GENERATION_PROMPT.md into one ready-to-send prompt per language by
substituting [LANGUAGE] -> display name and [AREA] -> region from LANGUAGE_REGISTRY.
Only the text from the "## 1. Your role" heading onward is sent to the model.

    python -m msafetemp.cultural_cards.render_prompts --out prompts_dir [german ukrainian ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from msafetemp.cultural_cards import LANGUAGE_REGISTRY

TEMPLATE = Path(__file__).with_name("CARD_GENERATION_PROMPT.md")
SPLIT_MARKER = "## 1. Your role"


def render(language: str) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    idx = template.find(SPLIT_MARKER)
    if idx == -1:
        raise SystemExit(f"marker '{SPLIT_MARKER}' not found in {TEMPLATE}")
    meta = LANGUAGE_REGISTRY[language]
    return template[idx:].replace("[LANGUAGE]", meta["name"]).replace("[AREA]", meta["region"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("languages", nargs="*", help="language keys (default: all but english)")
    p.add_argument("--out", required=True, help="directory for <language>.md prompt files")
    args = p.parse_args()
    langs = args.languages or [l for l in LANGUAGE_REGISTRY if l != "english"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        if lang not in LANGUAGE_REGISTRY:
            print(f"skip {lang}: not in LANGUAGE_REGISTRY")
            continue
        (out / f"{lang}.md").write_text(render(lang), encoding="utf-8")
        print(f"wrote {out / f'{lang}.md'}")


if __name__ == "__main__":
    main()
