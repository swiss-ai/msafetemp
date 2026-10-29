"""Load the English jailbreak-wrapper corpus (PyRIT-style YAML templates).

Only single-slot wrappers are used: `parameters == ["prompt"]` and a `{{ prompt }}`
placeholder in `value`. Multi-parameter templates and wrappers that use Jinja constructs
beyond {% raw %} markers and the prompt slot are skipped. The released bank contains 557
eligible wrappers from the PyRIT jailbreak template collection (including its in-the-wild
subset); the YAML files are not redistributed here, point --corpus_dir at a local copy of
that template directory.

Wrapper ids are the YAML path relative to the corpus directory.
"""

from __future__ import annotations

import os
import random
import re

_PLACEHOLDERS = ("{{ prompt }}", "{{prompt}}")
_SLOT_PATTERN = re.compile("|".join(re.escape(ph) for ph in _PLACEHOLDERS))
CANONICAL_SLOT = "{{ prompt }}"


def _has_slot(value: str) -> bool:
    return any(ph in value for ph in _PLACEHOLDERS)


def fill_slot(value: str, instruction: str) -> str:
    """Render the only two Jinja constructs that appear: drop {% raw %} / {% endraw %}
    markers and substitute the prompt slot. Only the scaffold is normalised; the inserted
    text is literal data (its braces, escapes and whitespace are not interpreted)."""
    text = value.replace("{% raw %}", "").replace("{% endraw %}", "").strip()
    return _SLOT_PATTERN.sub(lambda match: instruction, text)


def load_jailbreak_wrappers(corpus_dir: str) -> list[dict]:
    import yaml
    wrappers = []
    for dirpath, _, filenames in os.walk(corpus_dir):
        for fn in filenames:
            if not fn.endswith(".yaml"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8") as stream:
                    doc = yaml.safe_load(stream)
            except Exception as e:
                print(f"[skip unparseable] {path}: {e}")
                continue
            if not isinstance(doc, dict):
                continue
            value = doc.get("value")
            params = doc.get("parameters") or []
            if not isinstance(value, str) or params != ["prompt"] or not _has_slot(value):
                continue
            rel = os.path.relpath(path, corpus_dir)
            # Skip wrappers whose filled sentinel still contains {{ or {% (constructs our
            # renderer does not handle); they would yield a malformed prompt.
            probe = fill_slot(value, "\x00PAYLOAD\x00")
            if "\x00PAYLOAD\x00" not in probe or "{{" in probe or "{%" in probe:
                continue
            wrappers.append({"name": doc.get("name") or rel, "rel_path": rel, "value": value})
    wrappers.sort(key=lambda w: w["rel_path"])
    return wrappers


def load_jailbreak_samples(corpus_dir: str, n=None, seed=111) -> list[dict]:
    wrappers = load_jailbreak_wrappers(corpus_dir)
    random.Random(seed).shuffle(wrappers)
    return wrappers if n is None else wrappers[:n]
