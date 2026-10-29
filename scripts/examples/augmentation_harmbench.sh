#!/bin/bash
# HarmBench track (evaluation only, 60 languages): harmful run, separate benign run, post-hoc
# localization, affixes, taxonomy, HF dataset. Same endpoints as augmentation_wildguard.sh.
set -euo pipefail
: "${MSAFETEMP_DATA_DIR:?}" "${GEN_URL:?}" "${JUDGE_URL:?}" "${TAXONOMY_URL:?}"
D=$MSAFETEMP_DATA_DIR/harmbench
GEN_MODEL=${GEN_MODEL:-google/gemma-4-31B-it}
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
TAX_MODEL=${TAX_MODEL:-openai/gpt-oss-safeguard-120b}
WORKERS=${WORKERS:-600}

# 1. Harmful variants: 200 standard behaviors x all languages (english skips translation)
python -m msafetemp.augmentation.generate --track harmbench \
  --base_url "$GEN_URL" --model "$GEN_MODEL" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" \
  --seed 111 --max_retries 3 --num_variants 5 --paraphrasing 2 --benign 0 \
  --workers "$WORKERS" --output "$D/raw"

# 2. Benign minimal pairs: a separate run (re-extracts and re-translates the templates)
python -m msafetemp.augmentation.generate --track harmbench \
  --base_url "$GEN_URL" --model "$GEN_MODEL" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" \
  --seed 111 --max_retries 3 --num_variants 0 --paraphrasing 0 --benign 5 \
  --workers "$WORKERS" --output "$D/raw_benign"

# 3. Post-hoc cultural localization
python -m msafetemp.augmentation.posthoc --track harmbench --raw_dir "$D/raw" --max_recipes 5 \
  --target_url "$GEN_URL" --target_model "$GEN_MODEL" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" \
  --workers 200 --output "$D/posthoc"

# 4. Attack prefixes / suffixes per unique template (released list columns)
python -m msafetemp.augmentation.affixes --harmful_dir "$D/raw" --benign_dir "$D/raw_benign" \
  --posthoc_dir "$D/posthoc" --target_url "$GEN_URL" --target_model "$GEN_MODEL" --output "$D/affixes"

# 5. LinguaSafe labels of the 200 behaviors
python -m msafetemp.taxonomy.classify --track harmbench --harmbench_raw "$D/raw" \
  --judge_url "$TAXONOMY_URL" --judge_model "$TAX_MODEL" --output "$D/classification"

# 6. Hugging Face dataset (single test split)
python -m msafetemp.augmentation.build_dataset_harmbench \
  --harmful_dir "$D/raw" --benign_dir "$D/raw_benign" --posthoc_dir "$D/posthoc" \
  --affix_dir "$D/affixes" --classification "$D/classification/harmbench_200_tags.json" \
  --output "$D/hf"
