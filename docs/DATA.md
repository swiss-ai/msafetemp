# Data layout

All scripts read the released data from one root, `$MSAFETEMP_DATA_DIR`, and write their own
outputs under `$MSAFETEMP_OUT_DIR`. The augmentation examples (`scripts/examples/augmentation_*.sh`)
are the exception: they build the dataset itself and therefore write into `$MSAFETEMP_DATA_DIR`,
producing the same layout.

Legend: **[R]** released data (shipped with the dataset release, not produced by running this
code, or produced by a step that cannot be re-run identically), **[B]** built by the code in
this repository (the released copy is provided for convenience), **[O]** optional.

```
$MSAFETEMP_DATA_DIR/
├── harmbench/
│   ├── hf/                              [B] HarmBench track (60 languages), save_to_disk dataset with a
│   │                                        single `test` split and its `grouped_test` view
│   │                                        (build_dataset_harmbench; evaluation only)
│   ├── classification/harmbench_200_tags.json   [B] LinguaSafe labels of the 200 behaviors
│   └── prompts/
│       ├── standard_english.jsonl       [R] English HarmBench prompts ({"prompt": ...} per line), used by
│       └── benign_standard_english.jsonl    the evaluation-overlap guard of training/data/short_contrast
├── wildguard/
│   ├── canonical_2500.json              [B] the 2,500 canonical WildGuardMix seeds
│   ├── classification/canonical_2500_tags.json  [B] LinguaSafe labels of the seeds
│   ├── hf_audited/                      [B] WildGuardMix track (44 languages) after the benign audit:
│   │                                        `train` / `test` (2,000 / 500 source prompts, seed 111) and
│   │                                        `grouped_train` / `grouped_test`; the `test` split is the
│   │                                        WildGuard evaluation set
│   └── responses_manifest/ids.txt       [R] ids of the training prompts that received an SFT response
│                                            (training/data/select_response_rows; the released run
│                                            preferred prompts an unreleased reference model failed)
├── release/                             [R] flat parquet export of the public dataset (e.g.
│                                            test_harmful.parquet, test_posthoc.parquet)
├── sft/
│   └── train_balanced.jsonl             [R] balanced safety SFT set: 261,872 rows, 44 languages,
│                                            refuse : comply 1 : 1 per language (harmful 91,642,
│                                            posthoc 39,168, benign_direct 120,349, benign_bounded 10,713);
│                                            recipe in training/data/build_safety_sft
├── sft_mix/
│   └── id_order.txt                     [R][O] generated-row ids (`<bin>:<id>`) in the order of the reference
│                                            SFT files; `build_sft_mix --id-order` restores the exact
│                                            row order (and hence the exact seeded 400k TRL sample)
├── wrappers/
│   └── wrapper_bank.jsonl               [B] the 557 single-slot wrappers (English originals) and their
│                                            checked translations (augmentation/wrappers/translate)
├── wrapped_train/
│   ├── harmful/rows.jsonl               [B] 1,000 wrapped train-source prompts per language and subset
│   ├── benign/rows.jsonl                    (augmentation/wrappers/assemble --mode per_language)
│   └── split/
│       ├── train_ids.txt                [R] authoritative family-level train / held-out split of the
│       ├── test_harmful.jsonl               wrapped rows (32% of source families held out); the
│       └── test_benign.jsonl                original split hash is not recoverable
├── wrapped_judged/
│   ├── full/rows.jsonl                  [R] earlier dual-judged wrapped set (localized harmful payloads,
│   ├── english_harmful/rows.jsonl           all languages; English harmful and benign payloads)
│   ├── english_benign/rows.jsonl
│   └── judgments/<name>_judgments.jsonl [R] quality + safety judgments (augmentation/wrappers/judge)
├── wrapped_eval/
│   ├── posthoc_wrapped_test.jsonl       [R] PH-JB candidates (10 wrappers per post-hoc test prompt, both
│   │                                        tracks), e.g. for multi-wrapper evaluations
│   ├── harmbench_posthoc_wrapped_test.jsonl  [R] HB-JB (wrapped_hb300), harmful candidates
│   └── harmbench_benign_wrapped.jsonl   [R] HB-JB (wrapped_hb300), benign candidates
└── dpo/                                 [R][O] reference copies: DPO prompt pool (126,458 prompts,
                                             training/dpo/build_pool) and the Apertus-8B pairs
                                             (58,272, training/dpo/build_pairs)
```

The cultural cards are small and ship inside the repository (`data/cultural_cards/<language>.json`,
override with `MSAFETEMP_CARDS_DIR`). The PyRIT jailbreak templates are not redistributed; point
`PYRIT_TEMPLATES` at a local copy.

## What is released data and why

* `sft/train_balanced.jsonl`: the safety SFT rows used in every training run. It depends on model
  sampling (Gemma 4 31B IT answers, GPT-OSS-120B judge), on the response manifest and on the benign
  answers of an earlier response round, so re-running `select_response_rows`, `generate_responses`
  and `build_safety_sft` gives a set of the same design but not the same rows.
* `wildguard/responses_manifest/ids.txt`: the manifest preferred harmful prompts that an earlier,
  unreleased checkpoint answered unsafely; without that evaluation the tool falls back to a seeded
  random choice with the same quotas.
* `wrapped_train/split/train_ids.txt`: the held-out wrapped test families were drawn ad hoc; the
  released id list is authoritative (`training/data/wrapped_prompts split` applies the same procedure
  to new data).
* `sft_mix/id_order.txt`: optional; only needed for a row-for-row copy of the reference SFT files.
* `wrapped_judged/`, `wrapped_eval/`, `release/`, `harmbench/prompts/`: frozen inputs of the
  reference training mix and evaluations.

Everything else can be rebuilt with the example scripts (model outputs will differ between runs).

## Outputs (`$MSAFETEMP_OUT_DIR`)

```
$MSAFETEMP_OUT_DIR/
├── training/data/      scripts/examples/training_sft_data.sh (prompt sets, responses/, mix/)
├── training/runs/      scripts/examples/training_sft_dpo.sh (checkpoints, DPO pool, samples, pairs)
├── endpoints/          <job-name>.endpoint files written by scripts/slurm/eval_vllm_server.sbatch
├── hb/, wg/            scripts/examples/evaluation_tracks.sh (one dir per model)
├── wrapped_hb300/      scripts/examples/evaluation_wrapped.sh
└── lm_eval/            scripts/examples/evaluation_lm_eval.sh
```
