# msafetemp.augmentation

Template-based multilingual safety data augmentation: everything that turns the English seed
prompts into the released MSafeTemp tracks, plus the generator that writes the SFT responses.

All model calls go to OpenAI-compatible endpoints (for example vLLM servers). Every endpoint
argument takes one base URL or a comma-separated list of per-replica URLs. Paths come from
CLI arguments; the examples use `$D=$MSAFETEMP_DATA_DIR`. See
`scripts/examples/augmentation_wildguard.sh` and `augmentation_harmbench.sh` for complete runs
and `configs/augmentation/models_and_settings.yaml` for the models and per-stage settings.

## Stages

| Stage | Module | Output |
|---|---|---|
| Seed sources (WildGuardMix `wildguardtrain` harmful, HarmBench `standard`) | `sources.py` | |
| Extraction, validity judge, translation + translation judge + targeted patcher, TR fills + four-axis judge, paraphrases + paraphrase judge, benign exact-fill branch, affixes | `multilingual.py` (per seed and language), `stages.py` (single calls), `generate.py` (CLI) | `<lang>/regen.jsonl` |
| Canonical 2,500 WildGuard seeds | `select_canonical.py` | `canonical_2500.json` |
| WildGuard benign (OR) minimal pairs over the canonical seeds | `benign.py` | `<lang>/benign.jsonl` |
| Post-hoc cultural localization (PH): fit judge + contextual generator + judge2 | `posthoc.py` | `<lang>/posthoc.jsonl` |
| HarmBench affix backfill | `affixes.py` | `<lang>/affixes.jsonl` |
| Hugging Face datasets | `build_dataset_wildguard.py`, `build_dataset_harmbench.py` | `save_to_disk` dirs |
| Benign audit (judge -> selection rule -> reclassification) | `audit_benign.py`, `audit_select.py`, `audit_reclassify.py` | audited dataset |
| SFT responses: refusal design v2 + benign answers | `generate_responses.py` | `<lang>/responses.jsonl` |
| Jailbreak wrappers: corpus, native translation with checks, literal wrapping, dual judge | `wrappers/` | wrapper bank, wrapped rows |
| LinguaSafe labels | `msafetemp.taxonomy` | tag files |
| Cultural cards | `msafetemp.cultural_cards`, `data/cultural_cards` | |

All prompts are in `prompts/` and are verbatim copies of the prompts that produced the released
data. Judges answer independent per-criterion booleans and the verdict is computed in code; a
judge that returns no parseable JSON twice gets the `*_CLINICAL` fallback system prompt once.

## Order of a full run

WildGuardMix track (training data, 44 languages):

```
generate.py --track wildguard   (n 10,000 prompts >= 800 chars, seed 111, 5 fills x 3 templates, 5 prefixes / suffixes)
select_canonical.py             (first 2,500 passing English originals)
benign.py                       (5 exact-fill candidates per canonical template)
msafetemp.taxonomy.classify     (--track wildguard)
posthoc.py --track wildguard    (languages with a cultural card; at most 5 recipes per template)
build_dataset_wildguard.py      (2,000 / 500 original prompts train / test, split seed 111)
audit_benign.py -> audit_select.py -> audit_reclassify.py   (-> audited dataset)
generate_responses.py           (SFT responses for the chosen training rows)
```

HarmBench track (evaluation only, 60 languages):

```
generate.py --track harmbench                                   (harmful: 5 fills x 3 templates, --benign 0)
generate.py --track harmbench --num_variants 0 --paraphrasing 0 --benign 5   (benign run, own output dir)
posthoc.py --track harmbench
affixes.py
msafetemp.taxonomy.classify --track harmbench
build_dataset_harmbench.py      (single `test` split)
```

## Models and sampling

Generation: `google/gemma-4-31B-it`. Pipeline judges: `openai/gpt-oss-120b`. Taxonomy:
`openai/gpt-oss-safeguard-120b`. All calls use plain temperature sampling (top_p 1.0, no
top-k). Per-stage temperature / max tokens as implemented:

| Stage | Temp. | Max tokens |
|---|---|---|
| Template extractor | 0.2 | 8,192 |
| Template validity judge (judge1) | 0.0 | 32,768 |
| Template translator | 0.3 | 8,192 |
| Translation judge | 0.0 | 32,768 |
| Translation patcher | 0.2 | 32,768 (WildGuard) / 8,192 (HarmBench) |
| Harmful slot generator | 0.9 | 32,768 (WildGuard) / 8,192 (HarmBench) |
| Four-axis variant judge (judge2) | 0.0 | 32,768 |
| Benign exact-fill generator | 0.8 | 8,192 |
| Benign judge | 0.0 | 32,768 |
| Paraphraser | 0.9 | 32,768 |
| Paraphrase judge | 0.0 | 32,768 |
| Prefix / suffix generators | 0.9 | 8,192 |
| Jailbreak-wrapper extractor | 0.2 | 32,768 |
| Cultural fit judge | 0.0 | 24,576 |
| Cultural localization (contextual generator) | 0.9 | 4,096 |
| Taxonomy classifier | 0.0 | 16,384 |
| Benign audit judge (reasoning effort high) | 0.0 | 16,384 |
| Response generator | 0.8 | 400-1,400 (refusal, by length) / 2,048 (benign) |
| Response judge / evaluation harm judge | 0.0 | 8,192 / 16,384 |

Retries: a rejected extraction is retried with the judge's feedback up to `--max_retries`
(3 in the released runs) extra times; a rejected translation gets up to 3 targeted patch rounds
and is then kept with its validation record; a rejected TR variant with feedback is regenerated
once and re-judged.

## Version choices (what is and is not here)

Where several versions of a stage existed, only the one that produced the released data is
kept:

* Benign branch: the exact-fill design (fill only the slots, class judge with direct /
  bounded / toxic / reject_quality). The earlier free-rewrite benign generator and judge are
  not included; the WildGuard benign pairs were regenerated with the exact-fill branch over
  the canonical seeds (`benign.py`), and the benign variants stored by the main WildGuard run
  are not used by the dataset builder.
* Cultural context: generation and translation use `context_source=none`; localization happens
  only post hoc with the cultural cards (`posthoc.py`). Injecting card text directly into the
  generator was tried and reverted; the hook is kept in `multilingual.py` but not exposed by
  the CLI.
* Translator prompt: two framings, one per track (literal for WildGuard, localization for
  HarmBench), both kept because each produced its track.
* Refusal responses: design v2 only ("decline, then help"). The short-refusal v1 grid is not
  included.
* Benign audit: the final selection rule (`audit_select.py`), not the first 4-criterion rule.
* WildGuard test split for evaluation = the audited dataset's `test` split.
* Jailbreak wrappers: the released wrapped data insert dataset prompts into the 557 original
  single-slot wrappers and their checked translations. The earlier track that templated the
  wrapper scaffolds themselves and generated new payloads is not included; only the
  wrapper-extractor call and prompt are kept (`stages.run_jailbreak_extractor`) because they
  are part of the published prompt set.

## Settings notes

* The released data were produced with `--max_retries 3` for both failed extractions and
  translation patch rounds.
* The benign minimal-pair generator runs at temperature 0.8 and the taxonomy classifier at
  16,384 max tokens. The harmful slot generator uses 32,768 max tokens on the WildGuard track
  (long adversarial scaffolds); the translator runs at 0.3.
* The wrapper translation checks and the assembled-prompt quality judge used a separate
  multilingual quality model rather than GPT-OSS-120B; the model is a CLI argument.
* `prompts/benign_response.py`: BENIGN_BOUNDED_SYSTEM contains a paragraph of design notes
  that was part of the prompt string actually sent; kept verbatim.

## Outputs, schemas

`generate.py` record: original_prompt (+ behavior / hb_id / hb_category on HarmBench),
translated_template, translation / translation_judge / translation_attempts, status,
extraction (English template + slots), judge1, variants / rejected_variants (with judge2),
paraphrases (template, judge1, variants, rejected_variants), prefixes, suffixes,
benign_variants / rejected_benign_variants (with benign_judge incl. `class`).

`posthoc.py` record: the template sample + fit_judge (reasoning, applicable_slots, fits,
entity_recipes, needs_reframing, reframing_note) + contextual_variants (recipe, text, judge2).

Dataset rows (both tracks): id, group_id, original_id, original_index, split, language,
category (harmful / benign / posthoc, plus benign_audit_rejected after the audit), text,
template, branch (main / paraphrase / contextual), paraphrase_index, variant_index,
benign_class, recipe, prefixes, suffixes, judge_reasoning, ls_primary / ls_high_level /
ls_labels / ls_non_consensual_sexual, and track provenance (wg_* or hb_*).

## Jailbreak wrappers

```
wrappers/corpus.py     load the single-slot PyRIT-style wrapper YAMLs (557 eligible), fill_slot()
wrappers/translate.py  translate each wrapper into every target language with the payload slot
                       replaced by <harmful_request>; language check (candidate only), meaning
                       check (vs. English), slot-integrity and mixed-script checks; up to 2
                       repair rounds; -> wrapper_bank.jsonl (original / pass / reject / error)
wrappers/assemble.py   literal insertion of dataset prompts into bank wrappers:
                       --mode per_source   (matched evaluation: 10 wrappers per prompt, seed 20260913)
                       --mode per_language (training rows: 1,000 per language, 5-10 wrappers per
                                            payload, seed 20260922, train-source prompts only)
wrappers/judge.py      quality + safety dual judge of the assembled prompts
```

The wrapper YAML corpus itself is not redistributed; point `--corpus_dir` at a local copy of the
PyRIT jailbreak template directory. Wrapped prompts are never generated: the payload is inserted
verbatim, and a form that failed translation is recorded as blocked rather than replaced.
