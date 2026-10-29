#!/bin/bash
# Apertus-8B Megatron SFT, step 2/6: convert the HF checkpoint to a TP=4 Megatron init checkpoint.
# Env: WORK_ROOT, MEGATRON_ROOT, TRANSFORMERS_ROOT, WEIGHT_MODEL (local HF checkpoint: Apertus-8B-2509 or
#      Apertus-8B-Instruct-2509); optional ENV_SETUP, SRUN_ARGS (see train_megatron_prepare.sbatch).
#SBATCH --job-name=megatron-sft-convert
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --no-requeue
set -euo pipefail
: "${WORK_ROOT:?}" "${MEGATRON_ROOT:?}" "${TRANSFORMERS_ROOT:?}" "${WEIGHT_MODEL:?}"
TOKENIZER_VIEW=${WORK_ROOT}/tokenizer_sft_view
INIT_CHECKPOINT=${WORK_ROOT}/megatron_init_tp4
TEXT_VOCAB_SIZE=131072
[[ -f "${WEIGHT_MODEL}/config.json" && -f "${WEIGHT_MODEL}/model.safetensors.index.json" ]]
[[ -f "${TOKENIZER_VIEW}/sft_tokenizer_view_manifest.json" && -f "${MEGATRON_ROOT}/tools/checkpoint/loader_apertus.py" ]]
[[ ! -e "${INIT_CHECKPOINT}" ]]
export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON_ROOT TRANSFORMERS_ROOT WEIGHT_MODEL TOKENIZER_VIEW INIT_CHECKPOINT TEXT_VOCAB_SIZE ENV_SETUP="${ENV_SETUP:-}"
# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -lc '
  set -euo pipefail
  eval "${ENV_SETUP}"
  export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  python - <<"PY"
import json, os
from pathlib import Path
config = json.loads((Path(os.environ["WEIGHT_MODEL"]) / "config.json").read_text())
expected = {"model_type": "apertus", "vocab_size": 131072, "hidden_size": 4096, "intermediate_size": 21504,
            "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
            "max_position_embeddings": 65536, "rope_theta": 12000000}
assert {key: config.get(key) for key in expected} == expected
PY
  cd "${MEGATRON_ROOT}"
  python tools/checkpoint/convert.py --model-type GPT --loader apertus --saver core --loader-saver-mode process \
    --load-dir "${WEIGHT_MODEL}" --save-dir "${INIT_CHECKPOINT}" --tokenizer-model "${TOKENIZER_VIEW}" \
    --true-vocab-size "${TEXT_VOCAB_SIZE}" --target-tensor-parallel-size 4 --target-pipeline-parallel-size 1 \
    --make-vocab-size-divisible-by 128 --bf16
'
[[ -d "${INIT_CHECKPOINT}" && -n "$(find "${INIT_CHECKPOINT}" -type f -print -quit)" ]]
echo "CONVERT_COMPLETE init_checkpoint=${INIT_CHECKPOINT}"
