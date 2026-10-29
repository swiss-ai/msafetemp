#!/bin/bash
# Apertus-8B Megatron SFT, step 6/6: export the final iteration to a standalone Hugging Face directory.
# torch_dist -> torch (Megatron scripts/conversion/torchdist_2_torch.py), then torch -> HF with the
# Apertus HF checkpoint saver of the Megatron-LM checkout (tools/checkpoint/saver_<HF_SAVER>.py),
# chat model files included; validated with transformers before the final move.
# Env: WORK_ROOT, MEGATRON_ROOT, TRANSFORMERS_ROOT, EXPORT_ROOT (new output dir), HF_SAVER (name of the
#      Apertus HF saver module in the Megatron-LM checkout); optional ENV_SETUP, SRUN_ARGS.
#SBATCH --job-name=megatron-sft-export
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --no-requeue
set -euo pipefail
: "${WORK_ROOT:?}" "${MEGATRON_ROOT:?}" "${TRANSFORMERS_ROOT:?}" "${EXPORT_ROOT:?}" "${HF_SAVER:?}"
[[ -f "${MEGATRON_ROOT}/tools/checkpoint/saver_${HF_SAVER}.py" ]]
TEXT_VOCAB_SIZE=131072
PLAN_ENV=${WORK_ROOT}/training_plan.env
source "${PLAN_ENV}"
(( TRAIN_ITERS > 0 ))
ITER_DIR=$(printf 'iter_%07d' "${TRAIN_ITERS}")
HF_TOKENIZER=${WORK_ROOT}/tokenizer_sft_view
TRAIN_CHECKPOINT=${WORK_ROOT}/runs/full_tp4_gbs128/checkpoints
TORCH_CONVERT_ROOT=${WORK_ROOT}/conversion/hf_export_${ITER_DIR}_${SLURM_JOB_ID}
TORCH_CHECKPOINT=${TORCH_CONVERT_ROOT}/torch
EXPORT_STAGE=$(dirname "${EXPORT_ROOT}")/.$(basename "${EXPORT_ROOT}").incomplete-${SLURM_JOB_ID}
[[ -f "${TRANSFORMERS_ROOT}/src/transformers/models/apertus/modeling_apertus.py" ]]
[[ -f "${TRAIN_CHECKPOINT}/latest_checkpointed_iteration.txt" ]]
[[ "$(tr -d '[:space:]' < "${TRAIN_CHECKPOINT}/latest_checkpointed_iteration.txt")" == "${TRAIN_ITERS}" ]]
[[ -d "${TRAIN_CHECKPOINT}/${ITER_DIR}" ]]
[[ ! -e "${EXPORT_ROOT}" && ! -e "${EXPORT_STAGE}" && ! -e "${TORCH_CONVERT_ROOT}" ]]
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONUNBUFFERED=1
export HF_SAVER MEGATRON_ROOT TRANSFORMERS_ROOT HF_TOKENIZER TRAIN_CHECKPOINT EXPORT_STAGE TORCH_CONVERT_ROOT TORCH_CHECKPOINT TEXT_VOCAB_SIZE TRAIN_ITERS ENV_SETUP="${ENV_SETUP:-}"
# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -lc '
  set -euo pipefail
  eval "${ENV_SETUP}"
  export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  cd "${MEGATRON_ROOT}"
  python -m torch.distributed.run --standalone --nproc-per-node=4 scripts/conversion/torchdist_2_torch.py \
    --bf16 --load "${TRAIN_CHECKPOINT}" --ckpt-convert-save "${TORCH_CONVERT_ROOT}" --ckpt-step "${TRAIN_ITERS}"
  test -d "${TORCH_CHECKPOINT}"
  python tools/checkpoint/convert.py --model-type GPT --loader core --saver "${HF_SAVER}" \
    --load-dir "${TORCH_CHECKPOINT}" --save-dir "${EXPORT_STAGE}" --hf-tokenizer "${HF_TOKENIZER}" \
    --true-vocab-size "${TEXT_VOCAB_SIZE}" --position-embedding-type rope --save-chat-model
  python - <<"PY"
import json, os
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
root = Path(os.environ["EXPORT_STAGE"])
config = AutoConfig.from_pretrained(root, local_files_only=True)
assert config.model_type == "apertus"
assert int(config.vocab_size) == int(os.environ["TEXT_VOCAB_SIZE"])
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
assert tokenizer.vocab_size > 0
weight_map = json.loads((root / "model.safetensors.index.json").read_text()).get("weight_map", {})
assert "model.embed_tokens.weight" in weight_map and "lm_head.weight" in weight_map
for shard in set(weight_map.values()):
    assert (root / shard).is_file(), shard
print(f"hf_export_validation=pass vocab={config.vocab_size} tensors={len(weight_map)}")
PY
'
for f in config.json model.safetensors.index.json tokenizer_config.json generation_config.json; do
  [[ -f "${EXPORT_STAGE}/$f" ]]
done
mv "${EXPORT_STAGE}" "${EXPORT_ROOT}"
echo "HF_EXPORT_COMPLETE export_root=${EXPORT_ROOT}"
