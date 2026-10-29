#!/bin/bash
# HarmBench and WildGuard track evaluation of one target, then aggregation (summary and
# per-language rates). PROTOCOL=full: full test sets, 8192 tokens; PROTOCOL=light: fast checkpoint
# evaluation (4096 tokens, seeded WildGuard subsample). See configs/evaluation/protocols.yaml.
# Assumes a target and a GPT-OSS-Safeguard-120B judge are served on OpenAI-compatible endpoints, e.g.
#   sbatch --job-name=judge-safeguard scripts/slurm/eval_vllm_server.sbatch openai/gpt-oss-safeguard-120b 4 32768
#   sbatch --job-name=target-apertus8b scripts/slurm/eval_vllm_server.sbatch swiss-ai/Apertus-8B-Instruct-2509 4 32768
set -euo pipefail
: "${MSAFETEMP_OUT_DIR:?output root}"
# Evaluation datasets (docs/DATA.md); override with MSAFETEMP_HB_DATASET / MSAFETEMP_WG_DATASET.
export MSAFETEMP_HB_DATASET=${MSAFETEMP_HB_DATASET:-${MSAFETEMP_DATA_DIR:?}/harmbench/hf}
export MSAFETEMP_WG_DATASET=${MSAFETEMP_WG_DATASET:-$MSAFETEMP_DATA_DIR/wildguard/hf_audited}
NAME=${NAME:-Apertus-8B-Instruct-2509}
TARGET_MODEL=${TARGET_MODEL:-swiss-ai/Apertus-8B-Instruct-2509}
TARGET_URL=${TARGET_URL:-$MSAFETEMP_OUT_DIR/endpoints/target-apertus8b.endpoint}
JUDGE_URL=${JUDGE_URL:-$MSAFETEMP_OUT_DIR/endpoints/judge-safeguard.endpoint}
PROTOCOL=${PROTOCOL:-full}   # full | light
AGG=$( [[ $PROTOCOL == full ]] && echo breakdown || echo summary )
export TARGET_MODEL TARGET_URL JUDGE_URL

if [[ $PROTOCOL == full ]]; then
  # HarmBench: TR + PH + benign, 8192 tokens (add --score_degenerate to judge degenerate replies)
  TRACK=hb OUT=$MSAFETEMP_OUT_DIR/hb/$NAME EXTRA_ARGS="--posthoc --benign --max_tokens 8192" \
    sbatch --job-name=hb-$NAME --export=ALL scripts/slurm/eval_track_client.sbatch
  # WildGuard test split, full, 8192 tokens
  TRACK=wg OUT=$MSAFETEMP_OUT_DIR/wg/$NAME EXTRA_ARGS="--splits test --max_tokens 8192" \
    sbatch --job-name=wg-$NAME --export=ALL scripts/slurm/eval_track_client.sbatch
else
  # checkpoints: HarmBench TR + benign, then PH arm into a separate dir; WildGuard seeded subsample
  TRACK=hb OUT=$MSAFETEMP_OUT_DIR/hb/$NAME EXTRA_ARGS="--benign --max_tokens 4096" \
    sbatch --job-name=hb-$NAME --export=ALL scripts/slurm/eval_track_client.sbatch
  TRACK=hb OUT=$MSAFETEMP_OUT_DIR/hb/${NAME}_ph EXTRA_ARGS="--ph_only --no_original --max_tokens 4096" \
    sbatch --job-name=hbph-$NAME --export=ALL scripts/slurm/eval_track_client.sbatch
  TRACK=wg OUT=$MSAFETEMP_OUT_DIR/wg/$NAME \
    EXTRA_ARGS="--splits test --max_groups_per_arm 150 --max_variants_per_behavior 2 --max_tokens 4096" \
    sbatch --job-name=wg-$NAME --export=ALL scripts/slurm/eval_track_client.sbatch
fi

# After the clients finish (one --run per model; repeat --run NAME=DIR to merge a separate PH dir):
cat <<EOC
python -m msafetemp.evaluation.aggregate --track hb --protocol $AGG \\
    --run $NAME=$MSAFETEMP_OUT_DIR/hb/$NAME $( [[ $PROTOCOL == light ]] && echo "--run $NAME=$MSAFETEMP_OUT_DIR/hb/${NAME}_ph" ) \\
    --out_csv hb_$PROTOCOL.csv --per_language_csv hb_${PROTOCOL}_perlang.csv
python -m msafetemp.evaluation.aggregate --track wg --protocol $AGG \\
    --run $NAME=$MSAFETEMP_OUT_DIR/wg/$NAME --out_csv wg_$PROTOCOL.csv --per_language_csv wg_${PROTOCOL}_perlang.csv
EOC
