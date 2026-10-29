"""Build the HarmBench track of the dataset as a Hugging Face DatasetDict.

Same row schema as the WildGuardMix track (build_dataset_wildguard.py), from the 200
HarmBench `standard` behaviors:

    harmful  judge2-accepted TR variants, main + paraphrase branches
    benign   exact-fill minimal pairs of class direct | bounded (separate benign run)
    posthoc  culturally localized PH variants

There is NO train/test split: HarmBench is evaluation-only, so every behavior lands in a
single `test` split (plus its nested `grouped_test` view); rows carry split="test" for
schema parity. Prefixes/suffixes are joined from affixes.py output by (language,
TEMPLATE TEXT), because the benign run re-translated its templates. The affix generator
is unjudged, so blanks, duplicates and template echoes are dropped at join time.

Provenance: original_id (hb_NNNN, stable across languages), group_id
(<language>_hb_NNNN), hb_id, the English `behavior`, HarmBench's `hb_category`, and the
LinguaSafe `ls_*` label of the English behavior. A build_manifest.json is written next to
the data.

    python -m msafetemp.augmentation.build_dataset_harmbench \\
        --harmful_dir $D/harmbench/raw --benign_dir $D/harmbench/raw_benign \\
        --posthoc_dir $D/harmbench/posthoc --affix_dir $D/harmbench/affixes \\
        --classification $D/harmbench/classification/harmbench_200_tags.json \\
        --output $D/harmbench/hf
"""

import argparse
import json
import os

from datasets import Dataset, DatasetDict, Features, Value, Sequence

CONFIG = "standard"   # only HarmBench functional category in the data (see dataset_utils_harmbench)
SPLIT_NAME = "test"   # eval-only dataset: one split, no train side


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harmful_dir", required=True,
                   help="generate.py --track harmbench output: standard_<lang>/regen.jsonl")
    p.add_argument("--benign_dir", required=True,
                   help="output of the separate benign run of generate.py (standard_<lang>/regen.jsonl). "
                        "NOT --harmful_dir: its benign_variants are never used")
    p.add_argument("--posthoc_dir", required=True, help="posthoc.py output: <lang>/posthoc.jsonl")
    p.add_argument("--affix_dir", default=None,
                   help="affixes.py output (<lang>/affixes.jsonl), joined by (language, template). "
                        "Record-level prefixes/suffixes win when present")
    p.add_argument("--no_affixes", action="store_true", default=False,
                   help="do not join affixes (prefixes/suffixes then stay empty)")
    p.add_argument("--languages", nargs="+", default=None,
                   help="bare language names to include (default: all found under --harmful_dir)")
    p.add_argument("--output", required=True, help="output dir for the HF dataset (save_to_disk)")
    p.add_argument("--cache_dir", default=None,
                   help="where from_generator spills Arrow shards while building")
    p.add_argument("--classification", default=None,
                   help="harmbench_200_tags.json from msafetemp.taxonomy.classify --track harmbench; "
                        "if missing, ls_* fields are empty")
    p.add_argument("--no_benign", action="store_true", default=False)
    p.add_argument("--no_posthoc", action="store_true", default=False)
    p.add_argument("--no_grouped", action="store_true", default=False,
                   help="skip the nested grouped_test split")
    p.add_argument("--push_to_hub", default=None, help="optional hub repo id")
    p.add_argument("--private", action="store_true", default=False)
    return p.parse_args()


# Explicit schema so from_generator never buffers rows to infer features, and so a
# language with 0 rows in a split still gets the right column types.
FEATURES = Features({
    "id": Value("string"),
    "group_id": Value("string"),            # <language>_hb_NNNN: one template's whole output
    "original_id": Value("string"),         # hb_NNNN: the source behavior, same in every language
    "original_index": Value("int64"),       # hb_id as an int
    "split": Value("string"),               # always "test" (eval-only); kept for schema parity
    "language": Value("string"),
    "category": Value("string"),            # harmful / benign / posthoc
    "text": Value("string"),
    "template": Value("string"),            # the template this text was filled from
    "behavior": Value("string"),            # the English HarmBench behavior it all came from
    "branch": Value("string"),              # main / paraphrase / contextual
    "paraphrase_index": Value("int64"),     # -1 when not a paraphrase
    "variant_index": Value("int64"),
    "hb_id": Value("int64"),
    "hb_category": Value("string"),         # HarmBench semantic taxonomy (6 categories)
    "hb_functional_category": Value("string"),
    "ls_primary": Value("string"),          # LinguaSafe primary subcategory (disjoint, sums to n)
    "ls_high_level": Value("string"),       # its parent category
    "ls_labels": Sequence(Value("string")), # all LinguaSafe labels, ranked
    "ls_non_consensual_sexual": Value("bool"),
    "context_source": Value("string"),
    "benign_class": Value("string"),        # direct / bounded (benign only), else ""
    "recipe": Value("string"),              # JSON slot->entity (posthoc only), else ""
    "prefixes": Sequence(Value("string")),  # 5 attack prefixes for this template
    "suffixes": Sequence(Value("string")),  # 5 attack suffixes for this template
    "judge_reasoning": Value("string"),
})


# ─────────────────────────────────────────────────────────────────────────────
# Behaviors: identity and provenance metadata
# ─────────────────────────────────────────────────────────────────────────────

def bare_lang(dirname):
    return dirname[len(CONFIG) + 1:] if dirname.startswith(CONFIG + "_") else dirname


def find_languages(harmful_dir, languages):
    if languages:
        return languages
    return sorted(bare_lang(d) for d in os.listdir(harmful_dir)
                  if os.path.exists(os.path.join(harmful_dir, d, "regen.jsonl")))


def load_behaviors(harmful_dir, languages):
    """hb_id -> {behavior, hb_category, hb_functional_category}, the universe of source
    behaviors and the provenance fields describing the ENGLISH behavior.

    Scanned across every language rather than english alone: a behavior that english
    happened to drop (judge1 reject) but other languages kept would otherwise be missing
    from the behavior universe entirely. english is read first so its wording wins."""
    ordered = (["english"] if "english" in languages else []) + [l for l in languages if l != "english"]
    meta = {}
    for language in ordered:
        path = os.path.join(harmful_dir, f"{CONFIG}_{language}", "regen.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") != "pass":
                    continue
                hb_id = r.get("hb_id")
                if hb_id is None or hb_id in meta:
                    continue
                meta[hb_id] = {
                    "behavior": r.get("behavior") or r.get("original_prompt") or "",
                    "hb_category": r.get("hb_category") or "",
                    "hb_functional_category": r.get("hb_functional_category") or CONFIG,
                }
    if not meta:
        raise SystemExit(f"no standard_<lang>/regen.jsonl under {harmful_dir} to read behaviors from")
    print(f"behaviors: {len(meta)} distinct hb_ids across {len(ordered)} language(s)", flush=True)
    return meta


CLASSIFICATION_EMPTY = {"ls_primary": "", "ls_high_level": "", "ls_labels": [],
                        "ls_non_consensual_sexual": False}


def load_classification(path):
    """original_id (hb_NNNN) -> LinguaSafe tag, same shape as the WildGuard track's.
    `reasoning` is dropped here and stays in classification/tags.jsonl for audits."""
    if not path or not os.path.exists(path):
        print(f"classification: {path} not found -- ls_* fields will be empty", flush=True)
        return {}
    tags = json.load(open(path, encoding="utf-8"))["tags"]
    out = {oid: {"ls_primary": t.get("primary", ""),
                 "ls_high_level": t.get("primary_high_level", ""),
                 "ls_labels": list(t.get("labels") or []),
                 "ls_non_consensual_sexual": bool(t.get("is_non_consensual_sexual"))}
           for oid, t in tags.items()}
    print(f"classification: {len(out)} behaviors tagged", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Affix join (affixes.py output)
# ─────────────────────────────────────────────────────────────────────────────

AFFIX_CAP = 5    # affixes.py asks for 5; the generator occasionally returns 1-8


def _clean_affixes(items, template):
    """The affix generator is unjudged (as in the production pipeline), so the few
    malformed items it returns are dropped here rather than shipped in the dataset:
    blank strings, duplicates within a record, and items that echo the template back
    instead of being a standalone prefix/suffix (rule 1 of the generator prompt).
    Over-long lists are capped so every row carries at most AFFIX_CAP of each."""
    head = (template or "").strip()[:40]
    cleaned = []
    for a in items or []:
        a = a.strip()
        if not a or a in cleaned or (head and head in a):
            continue
        cleaned.append(a)
        if len(cleaned) == AFFIX_CAP:
            break
    return cleaned


def load_affixes(affix_dir, language):
    """{template: (prefixes, suffixes)} for one language; {} when absent."""
    lookup = {}
    if not affix_dir:
        return lookup
    path = os.path.join(affix_dir, language, "affixes.jsonl")
    if not os.path.exists(path):
        return lookup
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lookup[r["template"]] = (_clean_affixes(r.get("prefixes"), r["template"]),
                                          _clean_affixes(r.get("suffixes"), r["template"]))
    return lookup


def _affixes_for(record, template, affixes):
    """Record-level affixes (generated in-pipeline) win; else the post-hoc join by template."""
    if record.get("prefixes") or record.get("suffixes"):
        return record.get("prefixes") or [], record.get("suffixes") or []
    return affixes.get(template, ([], []))


# ─────────────────────────────────────────────────────────────────────────────
# Row builders -- one per source population
# ─────────────────────────────────────────────────────────────────────────────

def _base(ctx, row_id, category, text, template, branch, para_idx, var_idx,
          benign_class="", recipe="", prefixes=(), suffixes=(), reasoning=""):
    return {
        "id": row_id,
        "group_id": ctx["group_id"],
        "original_id": ctx["original_id"],
        "original_index": ctx["original_index"],
        "split": ctx["split"],
        "language": ctx["language"],
        "category": category,
        "text": text,
        "template": template,
        "behavior": ctx["behavior"],
        "branch": branch,
        "paraphrase_index": para_idx,
        "variant_index": var_idx,
        "hb_id": ctx["original_index"],
        "hb_category": ctx["hb_category"],
        "hb_functional_category": ctx["hb_functional_category"],
        "ls_primary": ctx["ls_primary"],
        "ls_high_level": ctx["ls_high_level"],
        "ls_labels": list(ctx["ls_labels"]),
        "ls_non_consensual_sexual": ctx["ls_non_consensual_sexual"],
        "context_source": ctx["context_source"],
        "benign_class": benign_class,
        "recipe": recipe,
        "prefixes": list(prefixes),
        "suffixes": list(suffixes),
        "judge_reasoning": reasoning,
    }


def harmful_rows(record, ctx, affixes):
    """judge2-accepted variants only (`variants`/`paraphrases[].variants`; the rejected
    ones live in separate fields and never enter the dataset). Paraphrase rows carry the
    record's main-template affixes, as in the pipeline."""
    rows = []
    template = record.get("translated_template") or (record.get("extraction") or {}).get("template") or ""
    prefixes, suffixes = _affixes_for(record, template, affixes)

    for var_idx, v in enumerate(record.get("variants") or []):
        rows.append(_base(
            ctx, f"{ctx['group_id']}_harmful_main_{var_idx}", "harmful",
            v.get("text", ""), template, "main", -1, var_idx,
            prefixes=prefixes, suffixes=suffixes,
            reasoning=(v.get("judge2") or {}).get("reasoning", "")))

    for para_idx, para in enumerate(record.get("paraphrases") or []):
        para_template = para.get("template") or ""
        for var_idx, v in enumerate(para.get("variants") or []):
            rows.append(_base(
                ctx, f"{ctx['group_id']}_harmful_para{para_idx}_{var_idx}", "harmful",
                v.get("text", ""), para_template, "paraphrase", para_idx, var_idx,
                prefixes=prefixes, suffixes=suffixes,
                reasoning=(v.get("judge2") or {}).get("reasoning", "")))
    return rows


def benign_rows(record, ctx, affixes):
    """Accepted exact-fill minimal pairs. `direct` and `bounded` are both kept and
    labelled: direct is eval+train (the over-refusal eval scores only these), bounded is
    dual-use and train-only. `toxic`/`reject_quality` are generator discards and live in
    rejected_benign_variants, which is not read. A variant with no class is
    pre-exact-fill data and is skipped."""
    rows = []
    template = record.get("translated_template") or (record.get("extraction") or {}).get("template") or ""
    prefixes, suffixes = _affixes_for(record, template, affixes)
    for var_idx, v in enumerate(record.get("benign_variants") or []):
        bj = v.get("benign_judge") or {}
        cls = bj.get("class") or ""
        if cls not in ("direct", "bounded"):
            continue
        rows.append(_base(
            ctx, f"{ctx['group_id']}_benign_{var_idx}", "benign",
            v.get("text", ""), template, "main", -1, var_idx,
            benign_class=cls, prefixes=prefixes, suffixes=suffixes,
            reasoning=bj.get("reasoning", "")))
    return rows


def posthoc_rows(record, ctx, affixes):
    """Culturally-contextualized variants. A variant with no judge2 result counts as
    passing (same convention as the eval's post-hoc arm builder); explicit rejects are
    dropped. `baseline_variants` are the uncontextualized control arm -- plain strings,
    never judged -- and are not part of the dataset."""
    rows = []
    template = record.get("template") or ""
    prefixes, suffixes = _affixes_for(record, template, affixes)
    for var_idx, v in enumerate(record.get("contextual_variants") or []):
        j2 = v.get("judge2")
        if j2 is not None and j2.get("verdict") != "pass":
            continue
        text = v.get("text")
        if not text:
            continue
        rows.append(_base(
            ctx, f"{ctx['group_id']}_posthoc_{var_idx}", "posthoc",
            text, template, "contextual", -1, var_idx,
            recipe=json.dumps(v.get("recipe") or {}, ensure_ascii=False),
            prefixes=prefixes, suffixes=suffixes,
            reasoning=(j2 or {}).get("reasoning", "")))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Streaming assembly
# ─────────────────────────────────────────────────────────────────────────────

SOURCES = {
    "harmful": (lambda d, lang: os.path.join(d, f"{CONFIG}_{lang}", "regen.jsonl"), harmful_rows, True),
    "benign": (lambda d, lang: os.path.join(d, f"{CONFIG}_{lang}", "regen.jsonl"), benign_rows, True),
    "posthoc": (lambda d, lang: os.path.join(d, lang, "posthoc.jsonl"), posthoc_rows, False),
}


def iter_rows(languages, dirs, categories, behaviors, affix_dir, classification=None, quiet=False):
    """Stream every row, one language and one category at a time, so from_generator
    writes Arrow shards incrementally and peak memory stays flat."""
    grand = 0
    for language in languages:
        affixes = load_affixes(affix_dir, language)
        if affix_dir and not affixes and not quiet:
            print(f"  [{SPLIT_NAME}] WARNING: no affixes for {language}", flush=True)
        for category in categories:
            path_fn, row_fn, require_pass = SOURCES[category]
            path = path_fn(dirs[category], language)
            if not os.path.exists(path):
                if not quiet:
                    print(f"  [{SPLIT_NAME}/{category}] no source for {language}, skipping", flush=True)
                continue
            n = 0
            seen_ids = set()
            dropped = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if require_pass and record.get("status") != "pass":
                        continue
                    hb_id = record.get("hb_id")
                    if hb_id is None or hb_id not in behaviors:
                        continue
                    original_id = f"hb_{hb_id:04d}"
                    # One record per (language, category, behavior). A couple of source
                    # files carry the same hb_id twice -- a resumed run that re-processed
                    # a behavior (measured: 24 in data/standard_zulu, 23 in
                    # posthoc_context/amharic). Those second records are a different
                    # template for a group_id that is supposed to identify exactly one,
                    # and their rows collide on `id` (the id is group_id + branch +
                    # index). First occurrence in file order wins, deterministically for
                    # both the flat and grouped passes.
                    if hb_id in seen_ids:
                        dropped += 1
                        continue
                    seen_ids.add(hb_id)
                    info = behaviors[hb_id]
                    ctx = {
                        "group_id": f"{language}_{original_id}",
                        "original_id": original_id,
                        "original_index": hb_id,
                        "split": SPLIT_NAME,
                        "language": language,
                        "behavior": info["behavior"],
                        "hb_category": record.get("hb_category") or info["hb_category"],
                        "hb_functional_category": info["hb_functional_category"],
                        "context_source": record.get("context_source") or "",
                        **((classification or {}).get(original_id) or CLASSIFICATION_EMPTY),
                    }
                    for row in row_fn(record, ctx, affixes):
                        n += 1
                        yield row
            grand += n
            if not quiet:
                dup_note = f"  [{dropped} duplicate-hb_id record(s) skipped]" if dropped else ""
                print(f"  [{SPLIT_NAME}/{category}] {language}: {n} rows  "
                      f"(running total {grand}){dup_note}", flush=True)


def build_flat(languages, dirs, categories, behaviors, affix_dir, classification, cache_dir):
    # `categories` must be a tuple, not a list: datasets treats every list in gen_kwargs
    # as a shardable data source and refuses to guess which one to parallelize over.
    # `languages` stays a list so it is the one it shards on.
    return Dataset.from_generator(
        iter_rows,
        gen_kwargs={"languages": list(languages), "dirs": dirs, "categories": tuple(categories),
                    "behaviors": behaviors, "affix_dir": affix_dir,
                    "classification": classification},
        features=FEATURES,
        cache_dir=cache_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Nested view: one row per (language, behavior)
# ─────────────────────────────────────────────────────────────────────────────

GROUPED_FEATURES = Features({
    "group_id": Value("string"),
    "original_id": Value("string"),
    "original_index": Value("int64"),
    "split": Value("string"),               # always "test" (eval-only)
    "language": Value("string"),
    "behavior": Value("string"),
    "template": Value("string"),            # the translated template the main variants fill
    "hb_id": Value("int64"),
    "hb_category": Value("string"),
    "hb_functional_category": Value("string"),
    "ls_primary": Value("string"),
    "ls_high_level": Value("string"),
    "ls_labels": Sequence(Value("string")),
    "ls_non_consensual_sexual": Value("bool"),
    "num_harmful_main": Value("int64"),
    "num_harmful_paraphrase": Value("int64"),
    "num_benign": Value("int64"),
    "num_posthoc": Value("int64"),
    "prefixes": Sequence(Value("string")),
    "suffixes": Sequence(Value("string")),
    "harmful_main": [{
        "text": Value("string"),
        "variant_index": Value("int64"),
        "judge_reasoning": Value("string"),
    }],
    "harmful_paraphrases": [{
        "text": Value("string"),
        "template": Value("string"),
        "paraphrase_index": Value("int64"),
        "variant_index": Value("int64"),
        "judge_reasoning": Value("string"),
    }],
    "benign": [{
        "text": Value("string"),
        "benign_class": Value("string"),
        "variant_index": Value("int64"),
        "judge_reasoning": Value("string"),
    }],
    "posthoc": [{
        "text": Value("string"),
        "recipe": Value("string"),
        "variant_index": Value("int64"),
        "judge_reasoning": Value("string"),
    }],
})


def _new_group():
    return {"original_id": "", "original_index": -1, "split": "", "language": "",
            "behavior": "", "template": "", "hb_category": "", "hb_functional_category": "",
            "ls_primary": "", "ls_high_level": "", "ls_labels": [], "ls_non_consensual_sexual": False,
            "prefixes": [], "suffixes": [],
            "harmful_main": [], "harmful_paraphrases": [], "benign": [], "posthoc": []}


def build_grouped(languages, dirs, categories, behaviors, affix_dir, classification=None):
    """Fold the flat rows (same builders, so the two views can never disagree) into one
    group per (language, behavior)."""
    groups = {}
    for row in iter_rows(languages, dirs, categories, behaviors, affix_dir, classification,
                         quiet=True):
        g = groups.setdefault(row["group_id"], _new_group())
        g["original_id"] = row["original_id"]
        g["original_index"] = row["original_index"]
        g["split"] = row["split"]
        g["language"] = row["language"]
        g["behavior"] = row["behavior"] or g["behavior"]
        g["hb_category"] = row["hb_category"] or g["hb_category"]
        g["hb_functional_category"] = row["hb_functional_category"] or g["hb_functional_category"]
        for k in ("ls_primary", "ls_high_level", "ls_labels", "ls_non_consensual_sexual"):
            g[k] = row[k] or g[k]
        if row["category"] == "harmful" and row["branch"] == "main":
            g["template"] = row["template"] or g["template"]
            g["prefixes"] = row["prefixes"] or g["prefixes"]
            g["suffixes"] = row["suffixes"] or g["suffixes"]
            g["harmful_main"].append({
                "text": row["text"], "variant_index": row["variant_index"],
                "judge_reasoning": row["judge_reasoning"]})
        elif row["category"] == "harmful":
            g["harmful_paraphrases"].append({
                "text": row["text"], "template": row["template"],
                "paraphrase_index": row["paraphrase_index"],
                "variant_index": row["variant_index"], "judge_reasoning": row["judge_reasoning"]})
        elif row["category"] == "benign":
            g["benign"].append({
                "text": row["text"], "benign_class": row["benign_class"],
                "variant_index": row["variant_index"], "judge_reasoning": row["judge_reasoning"]})
        elif row["category"] == "posthoc":
            g["posthoc"].append({
                "text": row["text"], "recipe": row["recipe"],
                "variant_index": row["variant_index"], "judge_reasoning": row["judge_reasoning"]})
        # A group whose harmful main branch is absent (benign/posthoc only) still needs
        # affixes: they are template-level, and every category of a group shares them.
        if not g["prefixes"] and row["prefixes"]:
            g["prefixes"] = row["prefixes"]
        if not g["suffixes"] and row["suffixes"]:
            g["suffixes"] = row["suffixes"]
    print(f"  [grouped_{SPLIT_NAME}] {len(groups)} groups", flush=True)

    def gen():
        for gid in sorted(groups):
            g = groups[gid]
            yield {
                "group_id": gid,
                "original_id": g["original_id"],
                "original_index": g["original_index"],
                "split": g["split"],
                "language": g["language"],
                "behavior": g["behavior"],
                "template": g["template"],
                "hb_id": g["original_index"],
                "hb_category": g["hb_category"],
                "hb_functional_category": g["hb_functional_category"],
                "ls_primary": g["ls_primary"],
                "ls_high_level": g["ls_high_level"],
                "ls_labels": g["ls_labels"],
                "ls_non_consensual_sexual": g["ls_non_consensual_sexual"],
                "num_harmful_main": len(g["harmful_main"]),
                "num_harmful_paraphrase": len(g["harmful_paraphrases"]),
                "num_benign": len(g["benign"]),
                "num_posthoc": len(g["posthoc"]),
                "prefixes": g["prefixes"],
                "suffixes": g["suffixes"],
                "harmful_main": g["harmful_main"],
                "harmful_paraphrases": g["harmful_paraphrases"],
                "benign": g["benign"],
                "posthoc": g["posthoc"],
            }

    return Dataset.from_generator(gen, features=GROUPED_FEATURES)


def main():
    args = parse_args()
    languages = find_languages(args.harmful_dir, args.languages)
    behaviors = load_behaviors(args.harmful_dir, languages)
    affix_dir = None if (args.no_affixes or not args.affix_dir) else os.path.abspath(args.affix_dir)
    classification = load_classification(args.classification)

    categories = ["harmful"]
    if not args.no_benign:
        categories.append("benign")
    if not args.no_posthoc:
        categories.append("posthoc")
    dirs = {"harmful": args.harmful_dir, "benign": args.benign_dir, "posthoc": args.posthoc_dir}

    print(f"languages ({len(languages)}): {languages}", flush=True)
    print(f"categories: {categories}", flush=True)
    print(f"behaviors: {len(behaviors)}  ->  all in the single '{SPLIT_NAME}' split "
          f"(eval-only, no train side)", flush=True)
    print(f"affix join: {affix_dir or 'DISABLED'}", flush=True)
    if args.cache_dir:
        print(f"build cache_dir: {args.cache_dir}", flush=True)

    splits = {}
    print(f"\nbuilding {SPLIT_NAME} split...", flush=True)
    splits[SPLIT_NAME] = build_flat(languages, dirs, categories, behaviors, affix_dir,
                                    classification, args.cache_dir)
    if not args.no_grouped:
        print(f"\nbuilding grouped_{SPLIT_NAME} split...", flush=True)
        splits[f"grouped_{SPLIT_NAME}"] = build_grouped(languages, dirs, categories,
                                                        behaviors, affix_dir, classification)

    ds = DatasetDict(splits)

    print("\ncounts:", flush=True)
    for name, split in ds.items():
        print(f"  {name}: {len(split):,} rows", flush=True)
        if name.startswith("grouped"):
            continue
        by_cat, by_lang, by_hb = {}, {}, {}
        for cat in split["category"]:
            by_cat[cat] = by_cat.get(cat, 0) + 1
        for lang in split["language"]:
            by_lang[lang] = by_lang.get(lang, 0) + 1
        for cat in split["hb_category"]:
            by_hb[cat or "<none>"] = by_hb.get(cat or "<none>", 0) + 1
        print(f"    by category: {by_cat}", flush=True)
        print(f"    languages: {len(by_lang)}  min/max rows per language: "
              f"{min(by_lang.values()):,}/{max(by_lang.values()):,}", flush=True)
        print(f"    by hb_category: {dict(sorted(by_hb.items(), key=lambda x: -x[1]))}", flush=True)
        n_affixed = sum(1 for p in split["prefixes"] if p)
        print(f"    rows with prefixes: {n_affixed:,}/{len(split):,}", flush=True)

    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)

    # What went into the build, written next to the data. There is no split assignment to
    # record (eval-only, single split) -- this is the source/behavior inventory instead.
    manifest = {
        "split": SPLIT_NAME,
        "train_test_split": False,
        "languages": languages,
        "categories": categories,
        "affix_dir": affix_dir,
        "harmful_dir": os.path.abspath(args.harmful_dir),
        "benign_dir": os.path.abspath(args.benign_dir),
        "posthoc_dir": os.path.abspath(args.posthoc_dir),
        "classification": os.path.abspath(args.classification)
                          if args.classification and os.path.exists(args.classification) else None,
        "behaviors": sorted(f"hb_{i:04d}" for i in behaviors),
    }
    with open(os.path.join(args.output, "build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nsaved to {args.output}  (+ build_manifest.json)")

    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub, private=args.private)
        print(f"pushed to hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
