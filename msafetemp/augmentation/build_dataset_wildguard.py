"""Build the WildGuardMix track of the dataset as a Hugging Face DatasetDict.

Three generated populations, all derived from the SAME canonical 2,500 English WildGuard
prompts:

    harmful  judge2-accepted TR variants, main + paraphrase branches (generate.py)
    benign   exact-fill minimal pairs of class direct | bounded (benign.py)
    posthoc  culturally localized PH variants (posthoc.py); a variant without a judge2
             result counts as passing, explicit rejects are dropped

Splits are TRAIN/TEST over the 2,000/500 ORIGINAL PROMPTS (seeded shuffle, seed 111), not
over rows: every variant of one original lands in the same split in every language and
category, so no paraphrase or entity substitution of one seed crosses the boundary.

Every row carries the LinguaSafe taxonomy label of its ENGLISH source prompt (`ls_*`,
msafetemp.taxonomy.classify), inherited through `original_id` (wg_NNNN = index in
canonical_2500.json), plus `group_id` (<language>_wg_NNNN), `original_index` and the
English `original_prompt`. The optional grouped_* splits nest one row per
(language, original). A split_assignment.json is written next to the data.

The benign audit (audit_benign.py -> audit_select.py -> audit_reclassify.py) is applied
afterwards to produce the audited dataset used for training and evaluation.

    python -m msafetemp.augmentation.build_dataset_wildguard \\
        --harmful_dir $D/wildguard/raw --benign_dir $D/wildguard/benign \\
        --posthoc_dir $D/wildguard/posthoc --canonical $D/wildguard/canonical_2500.json \\
        --classification $D/wildguard/classification/canonical_2500_tags.json \\
        --output $D/wildguard/hf --cache_dir $SCRATCH/hf_build_cache
"""

import argparse
import json
import os
import random

from datasets import Dataset, DatasetDict, Features, Value, Sequence

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harmful_dir", required=True,
                   help="generate.py --track wildguard output: <lang>/regen.jsonl (main + paraphrase "
                        "harmful variants). Its benign_variants are NOT read")
    p.add_argument("--benign_dir", required=True,
                   help="benign.py output: <lang>/benign.jsonl (exact-fill benign minimal pairs)")
    p.add_argument("--posthoc_dir", required=True,
                   help="posthoc.py output: <lang>/posthoc.jsonl (localized variants)")
    p.add_argument("--canonical", required=True,
                   help="canonical_2500.json from select_canonical.py")
    p.add_argument("--classification", default=None,
                   help="canonical_2500_tags.json from msafetemp.taxonomy.classify; if missing, "
                        "ls_* fields are emitted empty")
    p.add_argument("--languages", nargs="+", default=None,
                   help="languages to include (default: all found under --harmful_dir)")
    p.add_argument("--output", required=True, help="output dir for the HF dataset (save_to_disk)")
    p.add_argument("--cache_dir", default=None,
                   help="where from_generator spills Arrow shards while building (~2M rows); "
                        "point it at a scratch filesystem")
    p.add_argument("--n_test", type=int, default=500,
                   help="how many of the 2500 original prompts go to test (rest go to train)")
    p.add_argument("--split_seed", type=int, default=111,
                   help="seed for the train/test assignment over original prompts")
    p.add_argument("--no_benign", action="store_true", default=False)
    p.add_argument("--no_posthoc", action="store_true", default=False)
    p.add_argument("--no_grouped", action="store_true", default=False,
                   help="skip the nested grouped_train/grouped_test splits")
    p.add_argument("--push_to_hub", default=None, help="optional hub repo id")
    p.add_argument("--private", action="store_true", default=False)
    return p.parse_args()


# Explicit schema so from_generator never buffers rows to infer features, and so a
# language with 0 rows in a split still gets the right column types.
FEATURES = Features({
    "id": Value("string"),
    "group_id": Value("string"),           # <language>_wg_NNNN: one template's whole output
    "original_id": Value("string"),        # wg_NNNN: the source prompt, same in every language
    "original_index": Value("int64"),      # its position in canonical_2500.json
    "split": Value("string"),              # train / test (redundant, survives concatenation)
    "language": Value("string"),
    "category": Value("string"),           # harmful / benign / posthoc
    "text": Value("string"),
    "template": Value("string"),           # the template this text was filled from
    "original_prompt": Value("string"),    # the English WildGuard prompt it all came from
    "branch": Value("string"),             # main / paraphrase / contextual
    "paraphrase_index": Value("int64"),    # -1 when not a paraphrase
    "variant_index": Value("int64"),
    "wg_subcategory": Value("string"),
    "wg_harm_label": Value("string"),
    "wg_adversarial": Value("bool"),
    # LinguaSafe taxonomy of the ENGLISH source prompt -- same for every language/category
    # of a group. wg_subcategory is "others" for all 2500 canonical originals, so these are
    # the only usable harm-type fields in the dataset.
    "ls_primary": Value("string"),         # single most central subcategory (disjoint, sums to n)
    "ls_high_level": Value("string"),      # its parent category
    "ls_labels": Sequence(Value("string")),# all labels, ranked; [] when it fits no subcategory
    "ls_non_consensual_sexual": Value("bool"),
    "context_source": Value("string"),
    "benign_class": Value("string"),       # direct / bounded (benign only), else ""
    "recipe": Value("string"),             # JSON slot->entity (posthoc only), else ""
    "prefixes": Sequence(Value("string")),
    "suffixes": Sequence(Value("string")),
    "judge_reasoning": Value("string"),
})


# ─────────────────────────────────────────────────────────────────────────────
# Canonical originals: identity, provenance metadata, and the train/test split
# ─────────────────────────────────────────────────────────────────────────────

def load_canonical(path):
    """original_prompt -> (index, wg_NNNN id), in canonical_2500.json order."""
    prompts = json.load(open(path, encoding="utf-8"))["original_prompts"]
    return {prompt: (i, f"wg_{i:04d}") for i, prompt in enumerate(prompts)}


def load_prompt_metadata(harmful_dir, canonical):
    """wg_* provenance fields (subcategory / harm label / adversarial) per original prompt.

    Read once from english/regen.jsonl and reused for every language: the benign and
    posthoc files only carry wg_subcategory, and the fields describe the ENGLISH source
    prompt anyway, so one shared map keeps them identical across all rows of a group.
    Falls back to whatever language file exists if english is missing."""
    candidates = ["english"] + sorted(os.listdir(harmful_dir))
    for lang in candidates:
        path = os.path.join(harmful_dir, lang, "regen.jsonl")
        if not os.path.exists(path):
            continue
        meta = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                prompt = r.get("original_prompt")
                if prompt in canonical and prompt not in meta:
                    meta[prompt] = {
                        "wg_subcategory": r.get("wg_subcategory") or "",
                        "wg_harm_label": r.get("wg_harm_label") or "",
                        "wg_adversarial": bool(r.get("wg_adversarial")),
                    }
        print(f"prompt metadata: {len(meta)}/{len(canonical)} originals from {lang}", flush=True)
        return meta
    raise SystemExit(f"no <lang>/regen.jsonl under {harmful_dir} to read prompt metadata from")


CLASSIFICATION_EMPTY = {"ls_primary": "", "ls_high_level": "", "ls_labels": [],
                        "ls_non_consensual_sexual": False}


def load_classification(path, canonical):
    """original_id -> LinguaSafe tag. Keyed by original_id (wg_NNNN), which
    msafetemp.taxonomy.classify derives from the same canonical_2500.json ordering used here, so
    the join needs no prompt-text matching.

    `reasoning` is deliberately dropped: it would be duplicated onto every one of the
    ~1.8M rows that descend from the 2500 originals. It stays in tags.jsonl for audits."""
    if not path or not os.path.exists(path):
        print(f"classification: {path} not found -- ls_* fields will be empty", flush=True)
        return {}
    tags = json.load(open(path, encoding="utf-8"))["tags"]
    out = {}
    for oid, t in tags.items():
        out[oid] = {
            "ls_primary": t.get("primary", ""),
            "ls_high_level": t.get("primary_high_level", ""),
            "ls_labels": list(t.get("labels") or []),
            "ls_non_consensual_sexual": bool(t.get("is_non_consensual_sexual")),
        }
    n_unclassified = sum(1 for v in out.values() if not v["ls_labels"])
    print(f"classification: {len(out)}/{len(canonical)} originals tagged "
          f"({n_unclassified} fit no subcategory)", flush=True)
    return out


def assign_splits(canonical, n_test, seed):
    """train/test over ORIGINAL PROMPTS, so every variation of one prompt -- across all
    languages and all three categories -- lands on the same side. Seeded shuffle rather
    than a head/tail cut of the canonical order, so the split doesn't inherit whatever
    ordering the WildGuardMix stream had."""
    ids = sorted(oid for _, oid in canonical.values())
    random.Random(seed).shuffle(ids)
    test = set(ids[:n_test])
    return {oid: ("test" if oid in test else "train") for oid in ids}


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
        "original_prompt": ctx["original_prompt"],
        "branch": branch,
        "paraphrase_index": para_idx,
        "variant_index": var_idx,
        "wg_subcategory": ctx["wg_subcategory"],
        "wg_harm_label": ctx["wg_harm_label"],
        "wg_adversarial": ctx["wg_adversarial"],
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


def harmful_rows(record, ctx):
    """judge2-accepted variants only (`variants`/`paraphrases[].variants`; the rejected
    ones live in separate fields and never enter the dataset)."""
    rows = []
    template = record.get("translated_template") or (record.get("extraction") or {}).get("template") or ""
    prefixes = record.get("prefixes") or []
    suffixes = record.get("suffixes") or []

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


def benign_rows(record, ctx):
    """Accepted exact-fill minimal pairs. `direct` and `bounded` are both kept and
    labelled: direct is eval+train (over-refusal eval scores only these), bounded is
    dual-use and train-only. `toxic`/`reject_quality` are generator discards and live
    in rejected_benign_variants, which is not read."""
    rows = []
    template = record.get("template") or ""
    for var_idx, v in enumerate(record.get("benign_variants") or []):
        bj = v.get("benign_judge") or {}
        cls = bj.get("class") or ""
        if cls not in ("direct", "bounded"):
            continue
        rows.append(_base(
            ctx, f"{ctx['group_id']}_benign_{var_idx}", "benign",
            v.get("text", ""), template, "main", -1, var_idx,
            benign_class=cls, reasoning=bj.get("reasoning", "")))
    return rows


def posthoc_rows(record, ctx):
    """Culturally-contextualized variants. A variant with no judge2 result counts as
    passing (same convention as the evaluation post-hoc arm builder); explicit rejects
    are dropped. `baseline_variants` are the uncontextualized control arm -- plain
    strings, never judged -- and are not part of the dataset."""
    rows = []
    template = record.get("template") or ""
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
            reasoning=(j2 or {}).get("reasoning", "")))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Streaming assembly
# ─────────────────────────────────────────────────────────────────────────────

SOURCES = {
    "harmful": (lambda d, lang: os.path.join(d, lang, "regen.jsonl"), harmful_rows, True),
    "benign": (lambda d, lang: os.path.join(d, lang, "benign.jsonl"), benign_rows, False),
    "posthoc": (lambda d, lang: os.path.join(d, lang, "posthoc.jsonl"), posthoc_rows, False),
}


def iter_rows(languages, dirs, categories, canonical, metadata, classification, split_of,
              want_split, quiet=False):
    """Stream every row of one train/test side, one language and one category at a time,
    so from_generator writes Arrow shards incrementally and peak memory stays flat."""
    grand = 0
    for language in languages:
        for category in categories:
            path_fn, row_fn, require_pass = SOURCES[category]
            path = path_fn(dirs[category], language)
            if not os.path.exists(path):
                if not quiet:
                    print(f"  [{want_split}/{category}] no source for {language}, skipping", flush=True)
                continue
            n = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if require_pass and record.get("status") != "pass":
                        continue
                    prompt = record.get("original_prompt")
                    hit = canonical.get(prompt)
                    if hit is None:          # outside the canonical 2500
                        continue
                    original_index, original_id = hit
                    if split_of[original_id] != want_split:
                        continue
                    meta = metadata.get(prompt, {})
                    tag = classification.get(original_id) or CLASSIFICATION_EMPTY
                    ctx = {
                        "group_id": f"{language}_{original_id}",
                        "original_id": original_id,
                        "original_index": original_index,
                        "split": want_split,
                        "language": language,
                        "original_prompt": prompt,
                        "wg_subcategory": meta.get("wg_subcategory", record.get("wg_subcategory") or ""),
                        "wg_harm_label": meta.get("wg_harm_label", ""),
                        "wg_adversarial": meta.get("wg_adversarial", False),
                        **tag,
                        "context_source": record.get("context_source") or "",
                    }
                    for row in row_fn(record, ctx):
                        n += 1
                        yield row
            grand += n
            if not quiet:
                print(f"  [{want_split}/{category}] {language}: {n} rows  (running total {grand})", flush=True)


def build_flat(languages, dirs, categories, canonical, metadata, classification, split_of,
               want_split, cache_dir):
    # `categories` must be a tuple, not a list: datasets treats every list in gen_kwargs
    # as a shardable data source and refuses to guess which one to parallelize over.
    # `languages` stays a list so it is the one it shards on.
    return Dataset.from_generator(
        iter_rows,
        gen_kwargs={"languages": list(languages), "dirs": dirs, "categories": tuple(categories),
                    "canonical": canonical, "metadata": metadata,
                    "classification": classification, "split_of": split_of,
                    "want_split": want_split},
        features=FEATURES,
        cache_dir=cache_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Nested view: one row per (language, original prompt)
# ─────────────────────────────────────────────────────────────────────────────

GROUPED_FEATURES = Features({
    "group_id": Value("string"),
    "original_id": Value("string"),
    "original_index": Value("int64"),
    "split": Value("string"),
    "language": Value("string"),
    "original_prompt": Value("string"),
    "template": Value("string"),            # the translated template the main variants fill
    "wg_subcategory": Value("string"),
    "wg_harm_label": Value("string"),
    "wg_adversarial": Value("bool"),
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
            "original_prompt": "", "template": "", "wg_subcategory": "", "wg_harm_label": "",
            "wg_adversarial": False, "ls_primary": "", "ls_high_level": "",
            "ls_labels": [], "ls_non_consensual_sexual": False,
            "prefixes": [], "suffixes": [],
            "harmful_main": [], "harmful_paraphrases": [], "benign": [], "posthoc": []}


def build_grouped(languages, dirs, categories, canonical, metadata, classification,
                  split_of, want_split):
    """Fold the flat rows (same builders, so the two views can never disagree) into one
    group per (language, original prompt)."""
    groups = {}
    for row in iter_rows(languages, dirs, categories, canonical, metadata, classification,
                         split_of, want_split, quiet=True):
        g = groups.setdefault(row["group_id"], _new_group())
        g["original_id"] = row["original_id"]
        g["original_index"] = row["original_index"]
        g["split"] = row["split"]
        g["language"] = row["language"]
        g["original_prompt"] = row["original_prompt"]
        g["wg_subcategory"] = row["wg_subcategory"] or g["wg_subcategory"]
        g["wg_harm_label"] = row["wg_harm_label"] or g["wg_harm_label"]
        g["wg_adversarial"] = row["wg_adversarial"] or g["wg_adversarial"]
        g["ls_primary"] = row["ls_primary"] or g["ls_primary"]
        g["ls_high_level"] = row["ls_high_level"] or g["ls_high_level"]
        g["ls_labels"] = row["ls_labels"] or g["ls_labels"]
        g["ls_non_consensual_sexual"] = row["ls_non_consensual_sexual"] or g["ls_non_consensual_sexual"]
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
    print(f"  [grouped_{want_split}] {len(groups)} groups", flush=True)

    def gen():
        for gid in sorted(groups):
            g = groups[gid]
            yield {
                "group_id": gid,
                "original_id": g["original_id"],
                "original_index": g["original_index"],
                "split": g["split"],
                "language": g["language"],
                "original_prompt": g["original_prompt"],
                "template": g["template"],
                "wg_subcategory": g["wg_subcategory"],
                "wg_harm_label": g["wg_harm_label"],
                "wg_adversarial": g["wg_adversarial"],
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


def find_languages(harmful_dir, languages):
    if languages:
        return languages
    return sorted(d for d in os.listdir(harmful_dir)
                  if os.path.exists(os.path.join(harmful_dir, d, "regen.jsonl")))


def main():
    args = parse_args()
    languages = find_languages(args.harmful_dir, args.languages)
    canonical = load_canonical(args.canonical)
    metadata = load_prompt_metadata(args.harmful_dir, canonical)
    classification = load_classification(args.classification, canonical)
    split_of = assign_splits(canonical, args.n_test, args.split_seed)
    n_train = sum(1 for v in split_of.values() if v == "train")

    categories = ["harmful"]
    if not args.no_benign:
        categories.append("benign")
    if not args.no_posthoc:
        categories.append("posthoc")
    dirs = {"harmful": args.harmful_dir, "benign": args.benign_dir, "posthoc": args.posthoc_dir}

    print(f"languages ({len(languages)}): {languages}", flush=True)
    print(f"categories: {categories}", flush=True)
    print(f"originals: {len(canonical)}  ->  train {n_train} / test {args.n_test}  "
          f"(seed {args.split_seed})", flush=True)
    if args.cache_dir:
        print(f"build cache_dir: {args.cache_dir}", flush=True)

    splits = {}
    for want_split in ("train", "test"):
        print(f"\nbuilding {want_split} split...", flush=True)
        splits[want_split] = build_flat(languages, dirs, categories, canonical, metadata,
                                        classification, split_of, want_split, args.cache_dir)
    if not args.no_grouped:
        for want_split in ("train", "test"):
            print(f"\nbuilding grouped_{want_split} split...", flush=True)
            splits[f"grouped_{want_split}"] = build_grouped(
                languages, dirs, categories, canonical, metadata, classification,
                split_of, want_split)

    ds = DatasetDict(splits)

    print("\ncounts:", flush=True)
    for name, split in ds.items():
        print(f"  {name}: {len(split):,} rows", flush=True)
        if name.startswith("grouped"):
            continue
        by_cat, by_lang = {}, {}
        for cat in split["category"]:
            by_cat[cat] = by_cat.get(cat, 0) + 1
        for lang in split["language"]:
            by_lang[lang] = by_lang.get(lang, 0) + 1
        print(f"    by category: {by_cat}", flush=True)
        print(f"    languages: {len(by_lang)}  min/max rows per language: "
              f"{min(by_lang.values()):,}/{max(by_lang.values()):,}", flush=True)
        by_hl = {}
        for hl in split["ls_high_level"]:
            by_hl[hl or "<unclassified>"] = by_hl.get(hl or "<unclassified>", 0) + 1
        print(f"    by ls_high_level: {dict(sorted(by_hl.items(), key=lambda x: -x[1]))}", flush=True)

    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)

    # The split assignment is the one thing that cannot be re-derived from the dataset
    # alone if the seed or n_test ever changes -- write it next to the data.
    manifest = {
        "canonical": os.path.abspath(args.canonical),
        "n_test": args.n_test,
        "split_seed": args.split_seed,
        "languages": languages,
        "categories": categories,
        "classification": os.path.abspath(args.classification)
                          if args.classification and os.path.exists(args.classification) else None,
        "split_of_original_id": split_of,
    }
    with open(os.path.join(args.output, "split_assignment.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nsaved to {args.output}  (+ split_assignment.json)")

    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub, private=args.private)
        print(f"pushed to hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
