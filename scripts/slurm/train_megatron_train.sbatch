#!/bin/bash
# Apertus-8B Megatron SFT, steps 4-5/6: smoke run (SMOKE=1, 1 node, GBS 4, 40 samples) and the full run.
# Reference configuration: 32 nodes x 4 GH200 = 128 GPUs, TP 4, micro batch 1, global batch 128 packed sequences,
# one epoch (training_plan.env), lr 1e-5 -> 1e-6 cosine, 3% warmup, weight decay 0, clip 1.0, bf16,
# seed 1234, assistant-only loss (--ap-sft-plw 0), final checkpoint reported.
#   sbatch --nodes=1 --export=ALL,SMOKE=1 train_megatron_train.sbatch     # smoke
#   sbatch --nodes=32 train_megatron_train.sbatch                          # full run
# The full run can exit non-zero on a harmless post-training StopIteration after the final save, so
# chain the export with --dependency=afterany and let the export check the final iteration.
# Env: WORK_ROOT, MEGATRON_ROOT, TRANSFORMERS_ROOT; optional ENV_SETUP, SRUN_ARGS, MASTER_PORT.
#SBATCH --job-name=megatron-sft-train
#SBATCH --time=12:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
set -euo pipefail
: "${WORK_ROOT:?}" "${MEGATRON_ROOT:?}" "${TRANSFORMERS_ROOT:?}"
DATASET_ROOT=${WORK_ROOT}/dataset
DATA_CACHE=${WORK_ROOT}/dataset_cache
PLAN_ENV=${WORK_ROOT}/training_plan.env
INIT_CHECKPOINT=${WORK_ROOT}/megatron_init_tp4
TOKENIZER_VIEW=${WORK_ROOT}/tokenizer_sft_view
TEXT_VOCAB_SIZE=131072
[[ -f "${PLAN_ENV}" && -f "${DATASET_ROOT}/per_split_data_args.json" && -d "${DATA_CACHE}" && -d "${INIT_CHECKPOINT}" ]]
source "${PLAN_ENV}"
if [[ "${SMOKE:-0}" == 1 ]]; then
  RUN_ROOT=${WORK_ROOT}/runs/smoke_tp4_gbs4
  GBS=4; SAMPLES=40; EVAL_INTERVAL=5; EVAL_ITERS=1; SAVE_INTERVAL=10
else
  RUN_ROOT=${WORK_ROOT}/runs/full_tp4_gbs128
  [[ -f "${WORK_ROOT}/runs/smoke_tp4_gbs4/checkpoints/latest_checkpointed_iteration.txt" ]]
  (( TRAIN_SAMPLES > 0 && TRAIN_ITERS > 0 && GLOBAL_BATCH_SIZE == 128 ))
  GBS=${GLOBAL_BATCH_SIZE}; SAMPLES=${TRAIN_SAMPLES}
  SAVE_INTERVAL=$((TRAIN_ITERS / 5)); (( SAVE_INTERVAL < 10 )) && SAVE_INTERVAL=10; (( SAVE_INTERVAL > 1000 )) && SAVE_INTERVAL=1000
fi
[[ ! -e "${RUN_ROOT}" ]]
mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/tensorboard" "${RUN_ROOT}/triggers" "${WORK_ROOT}/triton_cache"
export PYTHONUNBUFFERED=1 TRITON_CACHE_DIR=${WORK_ROOT}/triton_cache TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-60221} WORLD_SIZE=${SLURM_NTASKS}
export MEGATRON_ROOT TRANSFORMERS_ROOT DATASET_ROOT DATA_CACHE INIT_CHECKPOINT RUN_ROOT TOKENIZER_VIEW TEXT_VOCAB_SIZE
export GBS SAMPLES EVAL_INTERVAL EVAL_ITERS SAVE_INTERVAL ENV_SETUP="${ENV_SETUP:-}"
# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --cpu-bind=none --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -lc '
  set -euo pipefail
  eval "${ENV_SETUP}"
  export PYTHONPATH="${TRANSFORMERS_ROOT}/src:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export RANK=${SLURM_PROCID} LOCAL_RANK=${SLURM_LOCALID}
  cd "${MEGATRON_ROOT}"
  exec python pretrain_gpt.py \
    --transformer-impl transformer_engine --use-precision-aware-optimizer --main-grads-dtype bf16 \
    --num-layers 32 --hidden-size 4096 --ffn-hidden-size 21504 --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --max-position-embeddings 65536 --position-embedding-type rope --rotary-base 12000000 \
    --make-vocab-size-divisible-by 128 --vocab-size "${TEXT_VOCAB_SIZE}" --padded-vocab-size "${TEXT_VOCAB_SIZE}" \
    --normalization RMSNorm --xielu --qk-layernorm --disable-bias-linear --untie-embeddings-and-output-weights \
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${TOKENIZER_VIEW}" \
    --per-split-data-args-path "${DATASET_ROOT}/per_split_data_args.json" --data-cache-path "${DATA_CACHE}" --seq-length 4096 --num-workers 1 --num-dataset-builder-threads 1 --dataloader-type single \
    --ap-sft --ap-sft-pack-samples --ap-sft-packing-strategy bfd --max-docs-per-bin-sft 8 --ap-sft-plw 0 --calculate-per-token-loss \
    --micro-batch-size 1 --global-batch-size "${GBS}" --train-samples "${SAMPLES}" --eval-interval "${EVAL_INTERVAL}" --eval-iters "${EVAL_ITERS}" \
    --lr 1.0e-5 --min-lr 1.0e-6 --lr-decay-style cosine --lr-warmup-fraction 0.03 --weight-decay 0.0 --clip-grad 1.0 \
    --bf16 --seed 1234 --finetune --auto-detect-ckpt-format --exit-on-missing-checkpoint --load "${INIT_CHECKPOINT}" \
    --save "${RUN_ROOT}/checkpoints" --save-interval "${SAVE_INTERVAL}" --ckpt-format torch_dist --tensorboard-dir "${RUN_ROOT}/tensorboard" \
    --log-interval 1 --log-throughput --log-progress --trigger-path "${RUN_ROOT}/triggers"
'
[[ -f "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" ]]
echo "TRAIN_COMPLETE run_root=${RUN_ROOT}"
