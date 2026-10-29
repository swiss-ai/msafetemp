"""Convert a card returned as a Python module (the format the card generation prompt asks
for: top-level TAXONOMY_SUPPLEMENTS dict and CULTURAL_GUIDANCE str) into the JSON card
format read by the pipeline. The module is parsed with `ast` and evaluated with
ast.literal_eval, never executed.

    python -m msafetemp.cultural_cards.module_to_json --language german \\
        --module path/to/german_card.py [--out data/cultural_cards/german.json]
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from msafetemp.cultural_cards import LANGUAGE_REGISTRY, card_path

REQUIRED = ("TAXONOMY_SUPPLEMENTS", "CULTURAL_GUIDANCE")


def parse_card_module(source: str) -> dict:
    """{"taxonomy_supplements": ..., "cultural_guidance": ...} from module source text.
    Raises SyntaxError / ValueError on a malformed module (e.g. a truncated reply)."""
    tree = ast.parse(source)
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in REQUIRED:
                values[target.id] = ast.literal_eval(node.value)
    missing = [k for k in REQUIRED if k not in values]
    if missing:
        raise ValueError(f"card module does not define {missing}")
    tax, guidance = values["TAXONOMY_SUPPLEMENTS"], values["CULTURAL_GUIDANCE"]
    if not isinstance(tax, dict) or not isinstance(guidance, str):
        raise ValueError("TAXONOMY_SUPPLEMENTS must be a dict and CULTURAL_GUIDANCE a str")
    return {"taxonomy_supplements": tax, "cultural_guidance": guidance}


def build_card(language: str, parsed: dict) -> dict:
    meta = LANGUAGE_REGISTRY.get(language, {})
    return {
        "language": language,
        "name": meta.get("name", language),
        "script": meta.get("script", ""),
        "region": meta.get("region", ""),
        "taxonomy_supplements": parsed["taxonomy_supplements"],
        "cultural_guidance": parsed["cultural_guidance"],
    }


def write_card(card: dict, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True)
    p.add_argument("--module", required=True, help="Python card module to convert")
    p.add_argument("--out", default=None, help="default: <cards dir>/<language>.json")
    args = p.parse_args()
    parsed = parse_card_module(Path(args.module).read_text(encoding="utf-8"))
    out = write_card(build_card(args.language, parsed), args.out or card_path(args.language))
    n = sum(len(v) for v in parsed["taxonomy_supplements"].values())
    print(f"wrote {out}  ({n} entities, {len(parsed['cultural_guidance'])} guidance chars)")


if __name__ == "__main__":
    main()
