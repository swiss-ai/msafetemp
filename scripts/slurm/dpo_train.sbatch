#!/bin/bash
# TRL DPO (msafetemp/training/dpo/trl_dpo.py), FSDP2 + activation checkpointing via accelerate.
#   PHASE=warm  : 1 node, build the datasets cache and exit (chain the wide run afterok on it)
#   PHASE=train : the real run. Reference configuration: 32 nodes x 4 GPUs, per-device batch 1, GRAD_ACCUM=1 (effective
#                 batch 128 pairs), lr 5e-6, 3 epochs, beta 0.1, NLL weight 0.2, maximum length 4096.
#
# Required env: MODEL (safety-SFT checkpoint), DATA (pairs jsonl), OUTPUT, ACC_CFG (configs/accelerate/*_ac.yaml)
# Optional env: LR (5e-6), EPOCHS (3), BETA (0.1), SFT_WEIGHT (0.2), MAX_LEN (4096), PER_DEVICE_BS (1),
#   GRAD_ACCUM (1), GPUS_PER_NODE (4), LOG_STEPS (10), MSAFETEMP_ROOT, ENV_SETUP, SRUN_ARGS, MASTER_PORT.
# Same software as train_trl_sft.sbatch.
#SBATCH --job-name=trl-dpo
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
set -euo pipefail
MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$SLURM_SUBMIT_DIR}
: "${MODEL:?}" "${DATA:?}" "${OUTPUT:?}" "${ACC_CFG:?}"
PHASE=${PHASE:?warm or train}
NNODES=${SLURM_NNODES:-1}
WORLD=$((NNODES * ${GPUS_PER_NODE:-4}))
LR=${LR:-5e-6}; EPOCHS=${EPOCHS:-3}; BETA=${BETA:-0.1}; SFT_WEIGHT=${SFT_WEIGHT:-0.2}; MAX_LEN=${MAX_LEN:-4096}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}; GRAD_ACCUM=${GRAD_ACCUM:-1}
echo "START $(date) phase=$PHASE model=$MODEL output=$OUTPUT nodes=$NNODES world=$WORLD"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOG_STEPS=${LOG_STEPS:-10}
export WARM_ONLY=$([[ $PHASE == warm ]] && echo 1 || echo 0)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29502}
mkdir -p "$OUTPUT"

# Preflight: memory left on a GPU by a previous job makes the run die during model setup.
# Wait up to 5 min for every GPU to be empty, then fail fast naming the node.
for attempt in 1 2 3 4 5 6; do
  busy=$(srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c \
    'u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -gt 2000 ] && echo "$(hostname):${u}MiB"' 2>/dev/null || true)
  [[ -z "$busy" ]] && break
  echo "preflight attempt $attempt: GPUs in use -> $busy"
  (( attempt == 6 )) && { echo "PREFLIGHT_FAILED: $busy" >&2; exit 3; }
  sleep 60
done

# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --ntasks="$NNODES" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  ${ENV_SETUP:-true}
  export PYTHONPATH=$MSAFETEMP_ROOT\${PYTHONPATH:+:\$PYTHONPATH}
  accelerate launch --config_file $ACC_CFG \
    --num_machines $NNODES --num_processes $WORLD --machine_rank \$SLURM_NODEID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    $MSAFETEMP_ROOT/msafetemp/training/dpo/trl_dpo.py \
      --model '$MODEL' --data '$DATA' --output '$OUTPUT' \
      --max-len $MAX_LEN --lr $LR --beta $BETA --sft-weight $SFT_WEIGHT \
      --per-device-bs $PER_DEVICE_BS --grad-accum $GRAD_ACCUM --epochs $EPOCHS
"
echo "END $(date) -> $OUTPUT"
