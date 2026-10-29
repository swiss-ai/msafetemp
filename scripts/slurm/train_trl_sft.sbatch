#!/bin/bash
# TRL safety SFT (msafetemp/training/sft/trl_sft.py), FSDP2 over N nodes x GPUS_PER_NODE GPUs via accelerate.
# Generic template: pass cluster options on the command line (sbatch --account=... --partition=...).
#
# Reference configuration (Llama 3.1 8B, Olmo 3 7B): 32 nodes x 4 GPUs = 128 GPUs, per-device batch 1, GRAD_ACCUM=1
# (effective batch 128), 400k-row TRL sample, one epoch. Launch a 1-node warm job first so the datasets
# cache is built once instead of by every rank:
#   WARM_ONLY=1 sbatch --nodes=1 train_trl_sft.sbatch
#   WARM_ONLY=0 sbatch --nodes=32 --dependency=afterok:<warm> train_trl_sft.sbatch
# (scripts/slurm/train_trl_sft_pipeline.sh chains both and the optional config restore.)
#
# Required env: MODEL (HF dir), DATA (TRL jsonl), OUTPUT, ACC_CFG (configs/accelerate/fsdp2_*.yaml)
# Optional env: TEMPLATE_MODEL (chat template source for base checkpoints), GRAD_ACCUM (1), MAX_LEN (4096),
#   LR (1e-5), EPOCHS (1), PER_DEVICE_BS (1), GPUS_PER_NODE (4), WARM_ONLY (0), SAVE_STRATEGY (no),
#   TRL_DATA_CACHE, MSAFETEMP_ROOT (default: submit dir), ENV_SETUP (e.g. "source venv/bin/activate"),
#   SRUN_ARGS (e.g. container flags), MASTER_PORT.
# Reference software: NGC PyTorch 25.12 container (its torch and NCCL), trl 1.12.0,
# transformers 5.16.1, accelerate 1.14.0, datasets 5.0.1. On multi-node GH200 the container's
# libfabric NCCL plugin matters (TCP sockets were ~15x slower).
#SBATCH --job-name=trl-sft
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
set -euo pipefail
MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$SLURM_SUBMIT_DIR}
: "${MODEL:?}" "${DATA:?}" "${OUTPUT:?}" "${ACC_CFG:?}"
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WORLD=$((NNODES * GPUS_PER_NODE))
MAX_LEN=${MAX_LEN:-4096}; LR=${LR:-1e-5}; EPOCHS=${EPOCHS:-1}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}; GRAD_ACCUM=${GRAD_ACCUM:-1}
echo "START $(date) model=$MODEL output=$OUTPUT nodes=$NNODES world=$WORLD warm_only=${WARM_ONLY:-0}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export WARM_ONLY=${WARM_ONLY:-0} SAVE_STRATEGY=${SAVE_STRATEGY:-no}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29501}
mkdir -p "$OUTPUT"
# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --ntasks="$NNODES" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  ${ENV_SETUP:-true}
  export PYTHONPATH=$MSAFETEMP_ROOT\${PYTHONPATH:+:\$PYTHONPATH}
  accelerate launch --config_file $ACC_CFG \
    --num_machines $NNODES --num_processes $WORLD --machine_rank \$SLURM_NODEID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    $MSAFETEMP_ROOT/msafetemp/training/sft/trl_sft.py \
      --model '$MODEL' ${TEMPLATE_MODEL:+--template-model '$TEMPLATE_MODEL'} \
      --data '$DATA' --output '$OUTPUT' --max-len $MAX_LEN --lr $LR --epochs $EPOCHS \
      --per-device-bs $PER_DEVICE_BS --grad-accum $GRAD_ACCUM
"
echo "END $(date) -> $OUTPUT"
