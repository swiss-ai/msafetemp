#!/bin/bash
# Apertus-8B safety SFT with Megatron-LM: render the run config, then submit
#   prepare (tokenizer view + tokenized bins) -> convert (HF -> TP4 Megatron) -> pack (plan + cache)
#   -> smoke (1 node) -> full run (32 nodes) -> HF export (afterany: see train_megatron_train.sbatch)
#
#   KIND=instruct MIX_MANIFEST=$MIX/mix_manifest.json WEIGHT_MODEL=$MODELS/Apertus-8B-Instruct-2509 \
#   INSTRUCT_TOKENIZER=$MODELS/Apertus-8B-Instruct-2509 WORK_ROOT=$RUNS/apertus_8b_instruct_sft \
#   EXPORT_ROOT=$MODELS/Apertus-8B-Instruct-2509-MSafeTemp-SFT MEGATRON_ROOT=... TRANSFORMERS_ROOT=... \
#   HF_SAVER=<Apertus HF saver name> bash scripts/slurm/train_megatron_pipeline.sh
#
# KIND=base starts from swiss-ai/Apertus-8B-2509 (WEIGHT_MODEL) with the same Instruct tokenizer view.
# Optional env: DEP, NODES (32), SBATCH_ARGS, ENV_SETUP, SRUN_ARGS, MSAFETEMP_ROOT, PYTHON (for rendering).
set -euo pipefail
: "${KIND:?base or instruct}" "${MIX_MANIFEST:?}" "${WEIGHT_MODEL:?}" "${INSTRUCT_TOKENIZER:?}" "${WORK_ROOT:?}"
: "${EXPORT_ROOT:?}" "${MEGATRON_ROOT:?}" "${TRANSFORMERS_ROOT:?}" "${HF_SAVER:?}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$(cd "$HERE/../.." && pwd)}
export KIND WEIGHT_MODEL INSTRUCT_TOKENIZER WORK_ROOT EXPORT_ROOT MEGATRON_ROOT TRANSFORMERS_ROOT HF_SAVER
SB="sbatch --parsable ${SBATCH_ARGS:-} --export=ALL"
mkdir -p "$WORK_ROOT"
(cd "$MSAFETEMP_ROOT" && ${PYTHON:-python} -m msafetemp.training.megatron_sft.render_config --kind "$KIND" \
   --mix-manifest "$MIX_MANIFEST" --weight-checkpoint "$WEIGHT_MODEL" --tokenizer-template "$INSTRUCT_TOKENIZER" \
   --work-root "$WORK_ROOT" --output "$WORK_ROOT/run_config.json")
PR=$($SB ${DEP:+--dependency=afterok:$DEP} "$HERE/train_megatron_prepare.sbatch")
CV=$($SB --dependency=afterok:$PR "$HERE/train_megatron_convert.sbatch")
PK=$($SB --dependency=afterok:$CV "$HERE/train_megatron_pack.sbatch")
SM=$(SMOKE=1 $SB --dependency=afterok:$PK --nodes=1 --time=00:45:00 --job-name=megatron-sft-smoke "$HERE/train_megatron_train.sbatch")
TR=$(SMOKE=0 $SB --dependency=afterok:$SM --nodes=${NODES:-32} "$HERE/train_megatron_train.sbatch")
XP=$($SB --dependency=afterany:$TR "$HERE/train_megatron_export.sbatch")
echo "prepare=$PR convert=$CV pack=$PK smoke=$SM train=$TR export=$XP -> $EXPORT_ROOT"
