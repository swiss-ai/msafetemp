#!/bin/bash
# Submit a TRL safety SFT run: 1-node warm job (datasets cache) -> NODES-node training -> optional
# vendor config restore (Olmo 3; see msafetemp/training/hf_utils.py restore-config).
#
#   MODEL=... DATA=$MIX/sft_trl_400k.jsonl OUTPUT=$OUT/Llama-3.1-8B-Instruct-MSafeTemp-SFT \
#   ACC_CFG=configs/accelerate/fsdp2_llama.yaml bash scripts/slurm/train_trl_sft_pipeline.sh
#
# Optional env: NODES (32), DEP (job id the run waits for), ORIGCFG (vendor config.json to restore
#   after training; set for Olmo 3), TEMPLATE_MODEL, SBATCH_ARGS (cluster flags for every job, e.g.
#   "--account=X --partition=Y"), plus everything train_trl_sft.sbatch reads.
# Accelerate configs per model: Llama fsdp2_llama.yaml, Olmo 3 fsdp2_olmo3.yaml,
#   Qwen 3.5 fsdp2_qwen35_ac.yaml (activation checkpointing needed for its large vocabulary).
set -euo pipefail
: "${MODEL:?}" "${DATA:?}" "${OUTPUT:?}" "${ACC_CFG:?}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$(cd "$HERE/../.." && pwd)}
NODES=${NODES:-32}
SB="sbatch --parsable ${SBATCH_ARGS:-}"
export MODEL DATA ACC_CFG TEMPLATE_MODEL=${TEMPLATE_MODEL:-} SAVE_STRATEGY=${SAVE_STRATEGY:-no}

WU=$(OUTPUT=${OUTPUT}_warm GRAD_ACCUM=32 WARM_ONLY=1 $SB ${DEP:+--dependency=afterok:$DEP} --nodes=1 \
     --job-name=trl-sft-warm --export=ALL "$HERE/train_trl_sft.sbatch")
TR=$(OUTPUT=$OUTPUT GRAD_ACCUM=1 WARM_ONLY=0 $SB --dependency=afterok:$WU --nodes=$NODES \
     --job-name=trl-sft --export=ALL "$HERE/train_trl_sft.sbatch")
POST=$TR
if [[ -n "${ORIGCFG:-}" ]]; then
  POST=$($SB --dependency=afterok:$TR --nodes=1 --ntasks=1 --time=00:10:00 --job-name=trl-sft-restore-config \
         --wrap "cd $MSAFETEMP_ROOT && ${ENV_SETUP:-true} && python -m msafetemp.training.hf_utils restore-config $OUTPUT $ORIGCFG")
fi
echo "warm=$WU train=$TR post=$POST output=$OUTPUT"
