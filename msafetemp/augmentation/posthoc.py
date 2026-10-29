"""Post-hoc cultural localization, the PH arm.

Localization happens only after template translation and validation. For each validated
translated template (context_source=none), a fit judge reads the language's cultural card
and decides slot by slot whether any card entity genuinely fits (Gate 1 type and
qualifier, Gate 2 existence / meaning / coherence incl. dates and fixed wording, Gate 3
gravity, Gate 4 victim polarity). Default is NO. For fitting templates it returns up to
--max_recipes concrete `slot -> entity` recipes; recipes whose slot names are not
copy-checkable against the template are dropped. The contextual generator writes one
variant per recipe (inflecting the entity and rephrasing around it, optionally
reframing the surface device), and every variant passes the same four-axis judge2 as TR.

    python -m msafetemp.augmentation.posthoc --track wildguard \\
        --raw_dir $MSAFETEMP_DATA_DIR/wildguard/raw \\
        --canonical $MSAFETEMP_DATA_DIR/wildguard/canonical_2500.json \\
        --target_url $GEN_URL --target_model google/gemma-4-31B-it \\
        --judge_url $JUDGE_URL --judge_model openai/gpt-oss-120b \\
        --output $MSAFETEMP_DATA_DIR/wildguard/posthoc

Writes <output>/<language>/posthoc.jsonl. `baseline_variants` (the uncontextualized TR
fills of the same template) are kept for side-by-side comparison and are not part of the
released PH arm.
"""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from msafetemp.common.config import api_key as resolve_api_key
from msafetemp.common.io import read_jsonl
from msafetemp.common.llm import chat, make_client, parse_json_object, parse_string_list
from msafetemp.augmentation._pool import run_language_pool
from msafetemp.augmentation.multilingual import TRACKS
from msafetemp.augmentation.prompts.posthoc import (
    CONTEXTUAL_GENERATOR_SYSTEM, CONTEXTUAL_GENERATOR_USER, FIT_JUDGE_SYSTEM, FIT_JUDGE_USER,
    REFRAMING_BLOCK, REFRAMING_RULE,
)
from msafetemp.augmentation.select_canonical import load_canonical
from msafetemp.augmentation.stages import build_slots_block, run_judge2
from msafetemp.cultural_cards import available_cards, load_card

_KNOWN_JUDGE2_CRITERIA = {"HARMFUL", "LANGUAGE_CONSISTENCY", "COHERENT", "FRAMING_INTACT"}


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------

def build_entity_index(taxonomy_supplements: dict) -> dict:
    """entity name -> full record, so the generator receives the grammatical_form /
    example_usage of exactly the entities a recipe uses."""
    index = {}
    for entities in taxonomy_supplements.values():
        for e in entities:
            if isinstance(e, dict) and e.get("entity"):
                index[e["entity"]] = e
    return index


def format_taxonomy_block(taxonomy_supplements: dict) -> str:
    """Judge-facing render: entity + meaning + fit-relevant metadata (grammatical form,
    usable_as and, most important, NOT usable_as). example_usage is routed to the
    generator instead."""
    lines = []
    for category, entities in taxonomy_supplements.items():
        lines.append(f"  {category}:")
        for e in entities:
            if not isinstance(e, dict):
                lines.append(f"    - {e}")
                continue
            ent = e.get("entity", "")
            ctx = e.get("societal_context", "")
            lines.append(f"    - {ent} — {ctx}" if ctx else f"    - {ent}")
            form = e.get("grammatical_form")
            if form:
                lines.append(f"        form: {form}")
            usable = e.get("usable_as")
            if usable:
                lines.append(f"        usable_as: {'; '.join(usable)}")
            not_usable = e.get("not_usable_as")
            if not_usable:
                lines.append(f"        NOT usable_as: {'; '.join(not_usable)}")
    return "\n".join(lines)


def format_recipe(recipe: dict, entity_index: dict | None = None) -> str:
    """Generator-facing render of one recipe, with the grammatical form and a
    correctly-inflected reference example for each assigned entity when available."""
    lines = []
    for slot, entity in recipe.items():
        lines.append(f"  {slot} = {entity}")
        rec = (entity_index or {}).get(entity)
        if rec:
            form = rec.get("grammatical_form")
            if form:
                lines.append(f"      grammatical form: {form}")
            example = rec.get("example_usage")
            if example:
                lines.append(f"      reference example (correctly inflected, do NOT copy): {example}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sample loading (validated translated templates, context_source=none)
# ---------------------------------------------------------------------------

def _regen_path(raw_dir: str, track: str, language: str) -> str:
    sub = f"standard_{language}" if track == "harmbench" else language
    return os.path.join(raw_dir, sub, "regen.jsonl")


def load_samples(raw_dir: str, track: str, language: str, n: int | None = None, seed: int = 0,
                 canonical: set[str] | None = None) -> list[dict]:
    """Every passing, templateable sample with at least one accepted TR variant; n=None
    returns all in file order, otherwise a deterministic reservoir sample."""
    rng = random.Random(seed)
    samples, reservoir, seen = [], [], 0
    for r in read_jsonl(_regen_path(raw_dir, track, language)):
        if r.get("status") != "pass":
            continue
        if canonical is not None and r.get("original_prompt") not in canonical:
            continue
        slots = (r.get("extraction") or {}).get("slots") or []
        translated_template = r.get("translated_template")
        if not slots or not r.get("variants") or not translated_template:
            continue
        record = {"language": language}
        if track == "harmbench":
            record.update({"behavior": r.get("behavior"), "template": translated_template,
                           "slots": slots, "hb_category": r.get("hb_category"),
                           "hb_id": r.get("hb_id")})
        else:
            record.update({"original_prompt": r.get("original_prompt"),
                           "template": translated_template, "slots": slots,
                           "wg_subcategory": r.get("wg_subcategory")})
        record["baseline_variants"] = [v.get("text") for v in r["variants"] if v.get("text")]
        if n is None:
            samples.append(record)
            continue
        seen += 1
        if len(reservoir) < n:
            reservoir.append(record)
        else:
            j = rng.randint(0, seen - 1)
            if j < n:
                reservoir[j] = record
    return samples if n is None else reservoir


def sample_key(rec: dict):
    return (rec.get("original_prompt") or rec.get("behavior"), rec.get("template"))


# ---------------------------------------------------------------------------
# Fit judge, contextual generator, judge2
# ---------------------------------------------------------------------------

def run_fit_judge(judge_client, judge_model, language, template, slots, taxonomy_block,
                  cultural_guidance, max_recipes):
    user_msg = FIT_JUDGE_USER.format(
        language=language.capitalize(),
        template=template,
        slots_block=build_slots_block(slots),
        taxonomy_block=taxonomy_block,
        cultural_guidance=cultural_guidance,
        max_recipes=max_recipes,
    )
    raw = chat(judge_client, judge_model, [
        {"role": "system", "content": FIT_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=24576, temperature=0.0)
    return parse_json_object(raw), raw


def fit_judge_with_retry(judge_client, judge_model, language, template, slots, taxonomy_block,
                         cultural_guidance, max_recipes, attempts=3):
    """Plain resend on an unparseable (e.g. bang-corrupted) reply."""
    fit, raw = None, ""
    for _ in range(attempts):
        fit, raw = run_fit_judge(judge_client, judge_model, language, template, slots,
                                 taxonomy_block, cultural_guidance, max_recipes)
        if fit is not None:
            return fit, raw
    return None, raw


def judge2_with_retry(judge_client, judge_model, template, variant, attempts=2):
    j2, raw = None, ""
    for _ in range(attempts):
        j2, raw = run_judge2(judge_client, judge_model, template, variant)
        if j2 is not None:
            return j2, raw
    return None, raw


def validate_recipes(recipes: list[dict], slot_names: set[str]) -> list[dict]:
    """Drop any recipe whose slot keys are not copy-checkable against the template."""
    valid = []
    for recipe in recipes:
        bad_keys = [k for k in recipe if k not in slot_names]
        if bad_keys:
            print(f"\n  [dropped recipe] unknown slot name(s) {bad_keys} not in {sorted(slot_names)}")
            continue
        valid.append(recipe)
    return valid


def run_contextual_generator(client, model, language, template, slots, recipe,
                             reframing_note="", entity_index=None):
    if reframing_note:
        reframing_block = REFRAMING_BLOCK.format(reframing_note=reframing_note)
        reframing_rule = REFRAMING_RULE
    else:
        reframing_block = ""
        reframing_rule = ""
    user_msg = CONTEXTUAL_GENERATOR_USER.format(
        language=language.capitalize(),
        template=template,
        slots_block=build_slots_block(slots),
        recipe_block=format_recipe(recipe, entity_index),
        reframing_block=reframing_block,
        reframing_rule=reframing_rule,
    )
    raw = chat(client, model, [
        {"role": "system", "content": CONTEXTUAL_GENERATOR_SYSTEM},
        {"role": "user", "content": user_msg},
    ], max_tokens=4096, temperature=0.9)
    variants = parse_string_list(raw)
    return (variants[0] if variants else None), raw


def process_sample(sample, language, client, model, judge_client, judge_model, taxonomy_block,
                   cultural_guidance, max_recipes, entity_index=None):
    template = sample["template"]
    slots = sample["slots"]
    slot_names = {s["name"] for s in slots}

    fit, fit_raw = fit_judge_with_retry(judge_client, judge_model, language, template, slots,
                                        taxonomy_block, cultural_guidance, max_recipes)
    result = {**sample, "fit_judge": fit, "fit_judge_raw_on_failure": None if fit else fit_raw,
              "contextual_variants": []}
    if not fit or not fit.get("fits"):
        return result

    recipes = validate_recipes(fit.get("entity_recipes") or [], slot_names)
    if len(recipes) > max_recipes:
        recipes = recipes[:max_recipes]
    if not recipes:
        return result

    reframing_note = fit.get("reframing_note", "") if fit.get("needs_reframing") else ""

    def _gen_and_judge(recipe):
        variant, gen_raw = run_contextual_generator(client, model, language, template, slots,
                                                    recipe, reframing_note, entity_index)
        if not variant:
            return {"recipe": recipe, "text": None, "generator_raw_on_failure": gen_raw}
        j2, _ = judge2_with_retry(judge_client, judge_model, template, variant)
        return {"recipe": recipe, "text": variant, "judge2": j2, "reframing_note": reframing_note}

    # Parallel across recipes, bounded by max_recipes.
    if len(recipes) == 1:
        result["contextual_variants"] = [_gen_and_judge(recipes[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(recipes)) as ex:
            result["contextual_variants"] = list(ex.map(_gen_and_judge, recipes))
    return result


def _new_counts() -> dict:
    return {"processed": 0, "fits": 0, "no_fit": 0, "fit_judge_failed": 0,
            "contextual_variants_total": 0, "contextual_variants_passing": 0,
            "contextual_variants_rejected": 0, "generation_failed": 0, "failed_criteria": {}}


def tally(result: dict, counts: dict) -> None:
    fit = result.get("fit_judge")
    if fit is None:
        counts["fit_judge_failed"] += 1
    elif fit.get("fits"):
        counts["fits"] += 1
    else:
        counts["no_fit"] += 1
    for cv in result.get("contextual_variants") or []:
        if not cv.get("text"):
            counts["generation_failed"] += 1
            continue
        counts["contextual_variants_total"] += 1
        j2 = cv.get("judge2")
        if j2 is None or j2.get("verdict") == "pass":
            counts["contextual_variants_passing"] += 1
        else:
            counts["contextual_variants_rejected"] += 1
            for crit in j2.get("failed_criteria", []):
                if crit in _KNOWN_JUDGE2_CRITERIA:
                    counts["failed_criteria"][crit] = counts["failed_criteria"].get(crit, 0) + 1


def summary(language: str, counts: dict, elapsed: float) -> None:
    keys = ("fits", "no_fit", "fit_judge_failed", "contextual_variants_total",
            "contextual_variants_passing", "contextual_variants_rejected", "generation_failed")
    print(f"[{language}] processed={counts['processed']}  " +
          "  ".join(f"{k}={counts.get(k, 0)}" for k in keys) + f"  {counts.get('failed_criteria')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", choices=TRACKS, required=True)
    p.add_argument("--raw_dir", required=True, help="generate.py output dir of the track")
    p.add_argument("--canonical", default=None, help="wildguard: canonical_2500.json")
    p.add_argument("--cards_dir", default=None, help="default: $MSAFETEMP_CARDS_DIR or data/cultural_cards")
    p.add_argument("--languages", nargs="+", default=None,
                   help="default: every language with both a cultural card and a regen.jsonl")
    p.add_argument("--n", type=int, default=None, help="cap samples per language (reservoir)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_recipes", type=int, default=5, help="maximum number of contextual variants per template")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument("--target_url", required=True)
    p.add_argument("--target_model", required=True)
    p.add_argument("--judge_url", required=True)
    p.add_argument("--judge_model", required=True)
    p.add_argument("--api_key", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    languages = args.languages or [l for l in available_cards(args.cards_dir)
                                   if os.path.exists(_regen_path(args.raw_dir, args.track, l))]
    if not languages:
        raise SystemExit("no languages with both a cultural card and a regen.jsonl")
    canonical = set(load_canonical(args.canonical)) if args.canonical else None
    cards = {l: load_card(l, args.cards_dir) for l in languages}
    rendered = {l: {"taxonomy_block": format_taxonomy_block(c["taxonomy_supplements"]),
                    "cultural_guidance": c["cultural_guidance"],
                    "entity_index": build_entity_index(c["taxonomy_supplements"])}
                for l, c in cards.items()}
    key = resolve_api_key(args.api_key)
    client = make_client(args.target_url, key, args.workers)
    judge_client = make_client(args.judge_url, key, args.workers)
    print(f"post-hoc localization: {len(languages)} language(s): {', '.join(languages)}")

    def process(language, sample):
        r = rendered[language]
        return process_sample(sample, language, client, args.target_model, judge_client,
                              args.judge_model, r["taxonomy_block"], r["cultural_guidance"],
                              args.max_recipes, r["entity_index"])

    run_language_pool(
        languages,
        lambda lang: load_samples(args.raw_dir, args.track, lang, args.n, args.seed, canonical),
        sample_key, process, Path(args.output), "posthoc.jsonl", args.workers,
        _new_counts, tally, summary, args.checkpoint_every)


if __name__ == "__main__":
    main()
