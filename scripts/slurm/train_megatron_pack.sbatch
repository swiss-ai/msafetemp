#!/bin/bash
# Apertus-8B Megatron SFT, step 3/6: one-epoch packing plan + packed-data cache.
# Packing: best-fit decreasing, sequence length 4096, at most 8 documents per packed sequence.
# Budget: one packed epoch of the unweighted pool rounded down to a full global batch of 128
# (reference mix: 453,504 sequences = 3,543 steps); the blend weights decide which bins fill it.
# Env: MSAFETEMP_ROOT, WORK_ROOT, MEGATRON_ROOT, TRANSFORMERS_ROOT; optional ENV_SETUP, SRUN_ARGS.
#SBATCH --job-name=megatron-sft-pack
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
set -euo pipefail
MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$SLURM_SUBMIT_DIR}
: "${WORK_ROOT:?}" "${MEGATRON_ROOT:?}" "${TRANSFORMERS_ROOT:?}"
CONFIG=${WORK_ROOT}/run_config.json
DATASET_ROOT=${WORK_ROOT}/dataset
DATA_CACHE=${WORK_ROOT}/dataset_cache
PLAN=${WORK_ROOT}/training_plan.json
TOKENIZER_VIEW=${WORK_ROOT}/tokenizer_sft_view
TEXT_VOCAB_SIZE=131072
[[ -f "${DATASET_ROOT}/dataset_manifest.json" && -f "${DATASET_ROOT}/per_split_data_args.json" ]]
[[ -f "${MEGATRON_ROOT}/tools/initialize_sft_dataset.py" && ! -e "${PLAN}" && ! -e "${DATA_CACHE}" ]]
mkdir -p "${DATA_CACHE}"
export MSAFETEMP_ROOT MEGATRON_ROOT TRANSFORMERS_ROOT CONFIG DATASET_ROOT PLAN DATA_CACHE TOKENIZER_VIEW TEXT_VOCAB_SIZE ENV_SETUP="${ENV_SETUP:-}"
# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -lc '
  set -euo pipefail
  eval "${ENV_SETUP}"
  export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${MEGATRON_ROOT}:${MSAFETEMP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  python -m msafetemp.training.megatron_sft.plan --config "${CONFIG}" --dataset-root "${DATASET_ROOT}" --output "${PLAN}"
  source "${PLAN%.json}.env"
  cd "${MEGATRON_ROOT}"
  exec python -m torch.distributed.run --nproc-per-node=1 tools/initialize_sft_dataset.py \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --num-layers 32 --hidden-size 4096 --ffn-hidden-size 21504 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --max-position-embeddings 65536 --position-embedding-type rope --rotary-base 12000000 \
    --make-vocab-size-divisible-by 128 --vocab-size "${TEXT_VOCAB_SIZE}" --padded-vocab-size "${TEXT_VOCAB_SIZE}" \
    --normalization RMSNorm --xielu --qk-layernorm --disable-bias-linear --untie-embeddings-and-output-weights \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${TOKENIZER_VIEW}" \
    --per-split-data-args-path "${DATASET_ROOT}/per_split_data_args.json" --data-cache-path "${DATA_CACHE}" \
    --seq-length 4096 --num-workers 1 --num-dataset-builder-threads 1 --dataloader-type single \
    --ap-sft --ap-sft-pack-samples --ap-sft-packing-strategy bfd --max-docs-per-bin-sft 8 \
    --ap-sft-plw 0 --ap-sft-mask-special-tokens --ap-sft-equalize-sample-loss --calculate-per-token-loss \
    --micro-batch-size 1 --global-batch-size "${GLOBAL_BATCH_SIZE}" --train-samples "${TRAIN_SAMPLES}" \
    --eval-interval "${EVAL_INTERVAL}" --eval-iters "${EVAL_ITERS}" --bf16 --seed 1234 --trigger-path /tmp
'
[[ -f "${PLAN}" && -n "$(find "${DATA_CACHE}" -type f -print -quit)" ]]
echo "PACK_COMPLETE plan=${PLAN}"
