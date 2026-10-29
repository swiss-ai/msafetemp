#!/bin/bash
# lm-evaluation-harness benchmarks (AyaRT, PolyTox, OR-Bench, MMLU, GSM8K).
# Requires upstream lm-evaluation-harness with vLLM: pip install "lm_eval[vllm]"
set -euo pipefail
: "${MSAFETEMP_OUT_DIR:?}"
EP=$MSAFETEMP_OUT_DIR/endpoints
sbatch --job-name=judge-llama70b scripts/slurm/eval_vllm_server.sbatch meta-llama/Llama-3.3-70B-Instruct 4 8192
sbatch --job-name=judge-llamaguard4 scripts/slurm/eval_vllm_server.sbatch meta-llama/Llama-Guard-4-12B 4 8192
# once both endpoint files exist:
MODEL_PATH=${MODEL_PATH:-swiss-ai/Apertus-8B-Instruct-2509} OUT=$MSAFETEMP_OUT_DIR/lm_eval/$(basename "${MODEL_PATH:-Apertus-8B-Instruct-2509}") \
AYA_JUDGE_EP=$EP/judge-llama70b.endpoint GUARD_EP=$EP/judge-llamaguard4.endpoint \
  sbatch --job-name=lmeval --export=ALL scripts/slurm/eval_lm_eval.sbatch
