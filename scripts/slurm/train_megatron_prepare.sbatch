#!/bin/bash
# Apertus-8B Megatron SFT, step 1/6: tokenizer view + tokenized dataset bins.
# Generic template: pass cluster options on the command line (sbatch --account=... --partition=...).
#
# Required env:
#   MSAFETEMP_ROOT     repository root (default: submit dir)
#   WORK_ROOT          run directory (holds run_config.json from render_config.py; created outputs go here)
#   MEGATRON_ROOT      Megatron-LM checkout with Apertus SFT support (commit 512aae23559aff142f1488c4a3dee2192cfbba1a)
#   TRANSFORMERS_ROOT  transformers checkout with Apertus support (commit b9f90dc388fd415a2ba2a6a31a372f451d4a4eed)
#   INSTRUCT_TOKENIZER local swiss-ai/Apertus-8B-Instruct-2509 directory (tokenizer + chat_template.jinja)
# Optional env:
#   ENV_SETUP          shell snippet run before python (e.g. "source /path/venv/bin/activate")
#   SRUN_ARGS          extra srun flags (e.g. a container: --container-image=...)
# Reference runtime: NGC PyTorch 25.12 container + huggingface-hub==1.0.0rc6, on GH200 nodes.
#SBATCH --job-name=megatron-sft-prepare
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
set -euo pipefail
MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$SLURM_SUBMIT_DIR}
: "${WORK_ROOT:?}" "${MEGATRON_ROOT:?}" "${TRANSFORMERS_ROOT:?}" "${INSTRUCT_TOKENIZER:?}"
CONFIG=${WORK_ROOT}/run_config.json
DATASET_ROOT=${WORK_ROOT}/dataset
TOKENIZER_VIEW=${WORK_ROOT}/tokenizer_sft_view
[[ -f "${CONFIG}" && -f "${INSTRUCT_TOKENIZER}/chat_template.jinja" ]]
[[ ! -e "${DATASET_ROOT}" && ! -e "${TOKENIZER_VIEW}" ]]
export MSAFETEMP_ROOT MEGATRON_ROOT TRANSFORMERS_ROOT CONFIG DATASET_ROOT TOKENIZER_VIEW INSTRUCT_TOKENIZER ENV_SETUP="${ENV_SETUP:-}"
# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -lc '
  set -euo pipefail
  eval "${ENV_SETUP}"
  export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${MEGATRON_ROOT}:${MSAFETEMP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  python -m msafetemp.training.megatron_sft.tokenizer_view --source "${INSTRUCT_TOKENIZER}" --output "${TOKENIZER_VIEW}"
  python -m msafetemp.training.megatron_sft.prepare_dataset --config "${CONFIG}" --output-root "${DATASET_ROOT}" --workers 16
'
[[ -f "${DATASET_ROOT}/dataset_manifest.json" && -f "${DATASET_ROOT}/per_split_data_args.json" ]]
echo "PREPARE_COMPLETE work_root=${WORK_ROOT}"
