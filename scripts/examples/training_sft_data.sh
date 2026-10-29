#!/bin/bash
# Build the final SFT mix from scratch. Run the steps on a CPU node (the
# generation steps only talk to OpenAI-compatible endpoints). Every step refuses to overwrite its output.
#
# Endpoints (one URL per line in each file; any OpenAI-compatible server works):
#   vllm serve google/gemma-4-31B-it --port 8080 --tensor-parallel-size 2 --max-model-len 16384
#   vllm serve openai/gpt-oss-120b  --port 8080 --max-model-len 32768
#   echo http://<gemma-host>:8080/v1 > $URLS/gemma.txt ; echo http://<judge-host>:8080/v1 > $URLS/judge.txt
set -euo pipefail
: "${MSAFETEMP_DATA_DIR:?}"                 # released MSafeTemp data, layout in docs/DATA.md
OUT=${MSAFETEMP_OUT_DIR:?}/training/data     # everything this script writes
URLS=${URLS:-$OUT/urls}
GEN_MODEL=${GEN_MODEL:-google/gemma-4-31B-it}
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
GEN=$URLS/gemma.txt; JUDGE=$URLS/judge.txt
mkdir -p "$OUT/responses"
respond() {  # respond <prompt dataset> <output dir> [categories...]: answers from the response generator
  local ds=$1 out=$2; shift 2
  local cats=(); [[ $# -gt 0 ]] && cats=(--categories "$@")
  python -m msafetemp.augmentation.generate_responses --dataset "$ds" --output "$out" --splits train \
    "${cats[@]}" --gen_url "$(paste -sd, "$GEN")" --gen_model "$GEN_MODEL" \
    --judge_url "$(paste -sd, "$JUDGE")" --judge_model "$JUDGE_MODEL" --attempts 2 --workers 800 --checkpoint_every 200
}

# 0. (optional, REBUILD_SAFETY_SFT=1) Rebuild the balanced safety SFT set instead of using the released
#    sft/train_balanced.jsonl: response manifest -> v2 refusals + benign answers -> 1:1 per language.
#    Model sampling differs between runs, so the result is a new set of the same design.
SAFETY_SFT=$MSAFETEMP_DATA_DIR/sft/train_balanced.jsonl
if [[ ${REBUILD_SAFETY_SFT:-0} == 1 ]]; then
  python -m msafetemp.training.data.select_response_rows \
    --dataset "$MSAFETEMP_DATA_DIR/wildguard/hf_audited" --out "$OUT/responses_manifest"
  python -m msafetemp.augmentation.generate_responses --dataset "$MSAFETEMP_DATA_DIR/wildguard/hf_audited" \
    --splits train --ids "$OUT/responses_manifest/ids.txt" --output "$OUT/responses_safety" \
    --gen_url "$(paste -sd, "$GEN")" --gen_model "$GEN_MODEL" \
    --judge_url "$(paste -sd, "$JUDGE")" --judge_model "$JUDGE_MODEL" --attempts 2 --seed 111 --workers 1200
  python -m msafetemp.training.data.build_safety_sft --responses "$OUT/responses_safety" \
    --benign_responses "$OUT/responses_safety" --output "$OUT/safety_sft"
  SAFETY_SFT=$OUT/safety_sft/train_balanced.jsonl
fi

# 1. MSafeTemp safety rows (harmful / localized harmful / benign direct / benign bounded; refuse:comply 1:1)
python -m msafetemp.training.data.adapt_safety_rows --source "$SAFETY_SFT" --output "$OUT/safety_rows.jsonl"

# 2. General instruction data: Dolci-Instruct-SFT (no tools), minus its WildGuardMix subset and decline openers
huggingface-cli download allenai/Dolci-Instruct-SFT-No-Tools --repo-type dataset --local-dir "$OUT/dolci"
python -m msafetemp.training.data.dolci_general extract --dolci-dir "$OUT/dolci/data" --output "$OUT/dolci_general_full.jsonl"
python -m msafetemp.training.data.dolci_general clean --input "$OUT/dolci_general_full.jsonl" --output "$OUT/dolci_general_clean.jsonl"

# 3. Unwrapped short prompts: English WildGuardMix-train vanilla prompts + 300 benign / 200 harmful per language
python -m msafetemp.training.data.short_prompts vanilla --out "$OUT/wg_vanilla_hf"
python -m msafetemp.training.data.short_prompts translate --src "$OUT/wg_vanilla_hf" --out "$OUT/wg_vanilla_ml_hf" \
  --urls "$GEN" --model "$GEN_MODEL"
for cat in harmful benign; do
  respond "$OUT/wg_vanilla_hf" "$OUT/responses/vanilla/$cat" "$cat"
  respond "$OUT/wg_vanilla_ml_hf" "$OUT/responses/vanilla_ml/$cat" "$cat"
done

# 4. Short benign contrast requests: 2 twins per short harmful prompt, judge-filtered, eval-overlap guarded,
#    then 500 per language translated
python -m msafetemp.training.data.short_contrast --src "$OUT/wg_vanilla_hf" --out "$OUT/short_contrast_en_hf" \
  --gen-urls "$GEN" --gen-model "$GEN_MODEL" --judge-urls "$JUDGE" --judge-model "$JUDGE_MODEL" \
  --harmbench-prompts "$MSAFETEMP_DATA_DIR/harmbench/prompts/standard_english.jsonl" \
                      "$MSAFETEMP_DATA_DIR/harmbench/prompts/benign_standard_english.jsonl" --per-prompt 2 --workers 2000
python -m msafetemp.training.data.short_prompts translate --src "$OUT/short_contrast_en_hf" --out "$OUT/short_contrast_ml_hf" \
  --urls "$GEN" --model "$GEN_MODEL" --benign-per-lang 500 --harmful-per-lang 0 --workers 1500
respond "$OUT/short_contrast_en_hf" "$OUT/responses/short_contrast/en/benign" benign
respond "$OUT/short_contrast_ml_hf" "$OUT/responses/short_contrast/ml/benign" benign

# 5. Jailbreak-wrapped train prompts: 1,000 per language and subset, 5-10 wrappers per payload, train-source
#    payloads only (scripts/examples/augmentation_wrappers_and_cards.sh step 3). The released
#    wrapped_train/{harmful,benign}/rows.jsonl and split/train_ids.txt reproduce the reference mix.
W=$MSAFETEMP_DATA_DIR/wrapped_train
python -m msafetemp.training.data.wrapped_prompts wrap1k --rows-dir "$W" --train-ids "$W/split/train_ids.txt" \
  --out "$OUT/wrapped_train_1k_hf"
respond "$OUT/wrapped_train_1k_hf" "$OUT/responses/wrap1k/harmful" harmful
respond "$OUT/wrapped_train_1k_hf" "$OUT/responses/wrap1k/benign" benign
#    Earlier judged wrapped set (mostly localized harmful payloads): accepted train-source rows only
J=$MSAFETEMP_DATA_DIR/wrapped_judged
python -m msafetemp.training.data.wrapped_prompts judged --out "$OUT/pyrit_wrapped_train_hf" \
  $(for s in "$J"/*/rows.jsonl; do n=$(basename "$(dirname "$s")"); echo --set "$s:$J/judgments/${n}_judgments.jsonl"; done)
for cat in posthoc harmful benign; do respond "$OUT/pyrit_wrapped_train_hf" "$OUT/responses/pyrit/$cat" "$cat"; done

# 6. The final mix: Megatron source + blend weights, and the 400k TRL sample
python -m msafetemp.training.data.build_sft_mix --safety-rows "$OUT/safety_rows.jsonl" \
  --general "$OUT/dolci_general_clean.jsonl" --responses-dir "$OUT/responses" \
  --wrap-train-ids "$W/split/train_ids.txt" \
  --short-contrast-prompts "$OUT/short_contrast_en_hf" "$OUT/short_contrast_ml_hf" --out-dir "$OUT/mix" \
  $( [[ -f "$MSAFETEMP_DATA_DIR/sft_mix/id_order.txt" ]] && echo --id-order "$MSAFETEMP_DATA_DIR/sft_mix/id_order.txt" )
# -> $OUT/mix/sft_source.jsonl (2,260,005 rows), sft_trl_400k.jsonl, mix_manifest.json
#    (sft_mix/id_order.txt, if present, restores the exact row order of the reference files)
