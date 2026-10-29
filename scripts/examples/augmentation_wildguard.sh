#!/bin/bash
# WildGuardMix track end to end: generation -> canonical seeds ->
# benign pairs -> taxonomy -> post-hoc localization -> HF dataset -> benign audit.
# Run each step on a CPU node (e.g. wrap it with scripts/slurm/augment_client.sbatch); the steps
# only call OpenAI-compatible endpoints:
#   vllm serve google/gemma-4-31B-it         --tensor-parallel-size 2 --max-model-len 65536
#   vllm serve openai/gpt-oss-120b           --max-model-len 65536
#   vllm serve openai/gpt-oss-safeguard-120b --max-model-len 32768
# Every step resumes from its own output if interrupted.
set -euo pipefail
: "${MSAFETEMP_DATA_DIR:?}" "${GEN_URL:?}" "${JUDGE_URL:?}" "${TAXONOMY_URL:?}"
D=$MSAFETEMP_DATA_DIR/wildguard
GEN_MODEL=${GEN_MODEL:-google/gemma-4-31B-it}
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
TAX_MODEL=${TAX_MODEL:-openai/gpt-oss-safeguard-120b}
WORKERS=${WORKERS:-150}
LANGS=${LANGS:-"english albanian amharic arabic bengali bulgarian chinese_simplified croatian czech danish dutch estonian finnish french german greek hindi hungarian indonesian italian japanese korean latvian lithuanian macedonian norwegian persian polish portuguese_brazilian romanian russian serbian slovak slovenian spanish swahili swedish swiss_german tagalog thai turkish ukrainian vietnamese yoruba"}

# 1. Templates, translation, TR fills (+2 paraphrases), affixes. The benign branch of this run is
#    not used by the dataset (step 3 regenerates it over the canonical seeds), so it is disabled.
python -m msafetemp.augmentation.generate --track wildguard \
  --base_url "$GEN_URL" --model "$GEN_MODEL" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" \
  --languages $LANGS --n 10000 --min_prompt_length 800 --seed 111 --max_retries 3 \
  --num_variants 5 --paraphrasing 2 --prefixes 5 --suffixes 5 --benign 0 \
  --workers "$WORKERS" --output "$D/raw"

# 2. The 2,500 canonical seeds
python -m msafetemp.augmentation.select_canonical --raw_dir "$D/raw" --out "$D/canonical_2500.json"

# 3. Exact-fill benign minimal pairs over the canonical templates
python -m msafetemp.augmentation.benign --raw_dir "$D/raw" --canonical "$D/canonical_2500.json" \
  --target_url "$GEN_URL" --target_model "$GEN_MODEL" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" \
  --benign 5 --workers "$WORKERS" --output "$D/benign"

# 4. LinguaSafe labels of the English originals
python -m msafetemp.taxonomy.classify --track wildguard --canonical "$D/canonical_2500.json" \
  --judge_url "$TAXONOMY_URL" --judge_model "$TAX_MODEL" --output "$D/classification"

# 5. Post-hoc cultural localization (languages with a card in data/cultural_cards)
python -m msafetemp.augmentation.posthoc --track wildguard --raw_dir "$D/raw" \
  --canonical "$D/canonical_2500.json" --max_recipes 5 \
  --target_url "$GEN_URL" --target_model "$GEN_MODEL" --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" \
  --workers "$WORKERS" --output "$D/posthoc"

# 6. Hugging Face dataset, train/test over original prompts (2,000 / 500, seed 111)
python -m msafetemp.augmentation.build_dataset_wildguard \
  --harmful_dir "$D/raw" --benign_dir "$D/benign" --posthoc_dir "$D/posthoc" \
  --canonical "$D/canonical_2500.json" --classification "$D/classification/canonical_2500_tags.json" \
  --output "$D/hf" --cache_dir "${HF_BUILD_CACHE:-$D/.hf_build_cache}"

# 7. Benign audit -> audited dataset (used for training and for the WildGuard test split)
python -m msafetemp.augmentation.audit_benign --export_from_dataset "$D/hf" \
  --infile "$D/audit/benign_all.jsonl" --outfile "$D/audit/audit_full.jsonl" \
  --judge_url "$JUDGE_URL" --judge_model "$JUDGE_MODEL" --reasoning_effort high
python -m msafetemp.augmentation.audit_select --audit "$D/audit/audit_full.jsonl" --out_dir "$D/audit"
python -m msafetemp.augmentation.audit_reclassify --in "$D/hf" --out "$D/hf_audited" \
  --reclassify "$D/audit/reclassify.json"

# 8. SFT responses for the training rows and the balanced safety SFT set (released as
#    sft/train_balanced.jsonl): see step 0 of scripts/examples/training_sft_data.sh
#    (select_response_rows -> generate_responses -> build_safety_sft).
