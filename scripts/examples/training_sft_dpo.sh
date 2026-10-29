#!/bin/bash
# SFT -> on-policy DPO for the reference models, submitted as Slurm job chains (after training_sft_data.sh).
# Cluster flags go into SBATCH_ARGS; a container or virtualenv into SRUN_ARGS / ENV_SETUP.
set -euo pipefail
: "${MSAFETEMP_DATA_DIR:?}" "${MSAFETEMP_OUT_DIR:?}" "${MODELS:?}"   # MODELS: directory with the local HF checkpoints
export SBATCH_ARGS=${SBATCH_ARGS:-}         # e.g. "--account=X --partition=Y"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=$MSAFETEMP_OUT_DIR/training/data
RUNS=$MSAFETEMP_OUT_DIR/training/runs
ACC=$REPO/configs/accelerate
cd "$REPO"

# --- SFT, Apertus-8B-Instruct-2509 with Megatron-LM (TP 4, 32 nodes, packed 4096, 3,543 steps) ---
out=$(KIND=instruct MIX_MANIFEST=$DATA/mix/mix_manifest.json \
      WEIGHT_MODEL=$MODELS/Apertus-8B-Instruct-2509 INSTRUCT_TOKENIZER=$MODELS/Apertus-8B-Instruct-2509 \
      WORK_ROOT=$RUNS/megatron_apertus_8b_instruct EXPORT_ROOT=$RUNS/Apertus-8B-Instruct-2509-MSafeTemp-SFT \
      MEGATRON_ROOT=${MEGATRON_ROOT:?} TRANSFORMERS_ROOT=${TRANSFORMERS_ROOT:?} HF_SAVER=${HF_SAVER:?} \
      bash scripts/slurm/train_megatron_pipeline.sh | tee /dev/stderr | tail -1)
APERTUS_EXPORT=$(sed -E 's/.*export=([0-9]+).*/\1/' <<< "$out")
# (KIND=base with WEIGHT_MODEL=$MODELS/Apertus-8B-2509 trains the base checkpoint on the same mix.)

# --- SFT, Llama-3.1-8B-Instruct and Olmo-3-7B-Instruct with TRL (400k rows, 128 GPUs, GA 1) ---
MODEL=$MODELS/Llama-3.1-8B-Instruct DATA=$DATA/mix/sft_trl_400k.jsonl ACC_CFG=$ACC/fsdp2_llama.yaml \
  OUTPUT=$RUNS/Llama-3.1-8B-Instruct-MSafeTemp-SFT bash scripts/slurm/train_trl_sft_pipeline.sh
MODEL=$MODELS/Olmo-3-7B-Instruct DATA=$DATA/mix/sft_trl_400k.jsonl ACC_CFG=$ACC/fsdp2_olmo3.yaml \
  OUTPUT=$RUNS/Olmo-3-7B-Instruct-MSafeTemp-SFT ORIGCFG=$MODELS/Olmo-3-7B-Instruct/config.json \
  bash scripts/slurm/train_trl_sft_pipeline.sh
# Base checkpoints without a chat template: add TEMPLATE_MODEL=<instruct sibling>.
# Qwen 3.5: ACC_CFG=$ACC/fsdp2_qwen35_ac.yaml (activation checkpointing).

# --- DPO pool (126,458 train-source prompts incl. the short benign contrast requests) ---
python -m msafetemp.training.dpo.build_pool --safety-sft "$MSAFETEMP_DATA_DIR/sft/train_balanced.jsonl" \
  --vanilla "$DATA/wg_vanilla_hf" --vanilla-ml "$DATA/wg_vanilla_ml_hf" \
  --wrapped "$DATA/wrapped_train_1k_hf" --wrap-train-ids "$MSAFETEMP_DATA_DIR/wrapped_train/split/train_ids.txt" \
  --short-contrast "$DATA/short_contrast_en_hf" "$DATA/short_contrast_ml_hf" --output "$RUNS/dpo_pool.jsonl"

# --- DPO on the Apertus SFT model: serve policy + gpt-oss-120b judge, sample, all pairs, 32-node DPO ---
NAME=Apertus-8B-Instruct-2509-MSafeTemp-SFT CKPT=$RUNS/Apertus-8B-Instruct-2509-MSafeTemp-SFT DEP=$APERTUS_EXPORT \
  POOL=$RUNS/dpo_pool.jsonl ACC_CFG=$ACC/fsdp2_apertus_ac.yaml OUT_ROOT=$RUNS/dpo \
  bash scripts/slurm/dpo_pipeline.sh
# Llama: ACC_CFG=$ACC/fsdp2_llama_ac.yaml; Olmo 3: ACC_CFG=$ACC/fsdp2_olmo3_ac.yaml ORIGCFG=<vendor config.json>;
# Qwen 3.5: ACC_CFG=$ACC/fsdp2_qwen35_ac.yaml POLICY_VLLM_EXTRA="--default-chat-template-kwargs '{\"enable_thinking\": false}'".
# Pass DEP=<train or restore-config job id> to start after the SFT run.
