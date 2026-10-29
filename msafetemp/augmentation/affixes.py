"""Attack prefix / suffix lists for already-generated HarmBench templates.

The WildGuard track generated affixes inside the main run (generate.py --prefixes 5
--suffixes 5). The HarmBench harmful, benign and post-hoc runs did not, so this tool runs
only the (unjudged) affix generators against every unique translated template of a
language across the three outputs. Affixes are template-level and released as list
columns (5 each) so a consumer chooses whether and how to wrap a prompt; they are not
used by the evaluation clients.

    python -m msafetemp.augmentation.affixes \\
        --harmful_dir $D/harmbench/raw --benign_dir $D/harmbench/raw_benign \\
        --posthoc_dir $D/harmbench/posthoc \\
        --target_url $GEN_URL --target_model google/gemma-4-31B-it \\
        --output $D/harmbench/affixes

Writes <output>/<language>/affixes.jsonl, one row per unique template:
{language, template, hb_ids, sources, prefixes, suffixes}. A template whose generator
failed both tries is not written, so a rerun retries it. build_dataset.py joins these back
by (language, template).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import read_jsonl
from msafetemp.common.llm import make_client
from msafetemp.augmentation._pool import run_language_pool
from msafetemp.augmentation.stages import (
    run_affix_generator, run_prefix_generator, run_suffix_generator,
)


def _source_files(args, language: str) -> list[tuple[str, str, str]]:
    return [
        ("harmful", os.path.join(args.harmful_dir, f"standard_{language}", "regen.jsonl"), "translated_template"),
        ("benign", os.path.join(args.benign_dir, f"standard_{language}", "regen.jsonl"), "translated_template"),
        ("posthoc", os.path.join(args.posthoc_dir, language, "posthoc.jsonl"), "template"),
    ]


def discover_languages(args) -> list[str]:
    langs = set()
    for base in (args.harmful_dir, args.benign_dir):
        if base and os.path.isdir(base):
            for entry in os.listdir(base):
                if entry.startswith("standard_") and os.path.exists(os.path.join(base, entry, "regen.jsonl")):
                    langs.add(entry[len("standard_"):])
    if args.posthoc_dir and os.path.isdir(args.posthoc_dir):
        for entry in os.listdir(args.posthoc_dir):
            if os.path.exists(os.path.join(args.posthoc_dir, entry, "posthoc.jsonl")):
                langs.add(entry)
    return sorted(langs)


def load_templates(args, language: str) -> list[dict]:
    """Unique translated templates across harmful / benign / post-hoc, first-seen order
    (harmful first). Regen records must have status == "pass"."""
    by_template: dict[str, dict] = {}
    for source, path, field in _source_files(args, language):
        if not os.path.exists(path):
            continue
        for r in read_jsonl(path):
            if source != "posthoc" and r.get("status") != "pass":
                continue
            template = r.get(field)
            if not template:
                continue
            entry = by_template.setdefault(template, {"language": language, "template": template,
                                                      "hb_ids": [], "sources": []})
            hb_id = r.get("hb_id")
            if hb_id is not None and hb_id not in entry["hb_ids"]:
                entry["hb_ids"].append(hb_id)
            if source not in entry["sources"]:
                entry["sources"].append(source)
    templates = list(by_template.values())
    return templates if args.n is None else templates[:args.n]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harmful_dir", required=True)
    p.add_argument("--benign_dir", default=None)
    p.add_argument("--posthoc_dir", default=None)
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--n", type=int, default=None, help="first N templates per language (debug)")
    p.add_argument("--prefixes", type=int, default=5)
    p.add_argument("--suffixes", type=int, default=5)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument("--target_url", required=True)
    p.add_argument("--target_model", required=True)
    p.add_argument("--api_key", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    args.benign_dir = args.benign_dir or ""
    args.posthoc_dir = args.posthoc_dir or ""

    client = make_client(args.target_url, resolve_api_key(args.api_key), args.workers)
    languages = args.languages or discover_languages(args)

    def process(language, sample):
        prefixes = run_affix_generator(client, args.target_model, sample["template"], args.prefixes,
                                       run_prefix_generator) if args.prefixes > 0 else []
        suffixes = run_affix_generator(client, args.target_model, sample["template"], args.suffixes,
                                       run_suffix_generator) if args.suffixes > 0 else []
        return {**sample, "prefixes": prefixes, "suffixes": suffixes}

    def complete(rec):
        return ((args.prefixes == 0 or bool(rec.get("prefixes")))
                and (args.suffixes == 0 or bool(rec.get("suffixes"))))

    def new_counts():
        return {"processed": 0, "failed": 0, "prefixes_total": 0, "suffixes_total": 0}

    def tally(rec, counts):
        if not complete(rec):
            counts["failed"] += 1
            return
        counts["prefixes_total"] += len(rec["prefixes"])
        counts["suffixes_total"] += len(rec["suffixes"])

    def summary(language, counts, elapsed):
        print(f"[{language}] templates={counts['processed']} failed={counts['failed']} "
              f"prefixes={counts['prefixes_total']} suffixes={counts['suffixes_total']}")

    run_language_pool(languages, lambda lang: load_templates(args, lang),
                      lambda r: r["template"], process, Path(args.output), "affixes.jsonl",
                      args.workers, new_counts, tally, summary, args.checkpoint_every, keep=complete)


if __name__ == "__main__":
    main()
