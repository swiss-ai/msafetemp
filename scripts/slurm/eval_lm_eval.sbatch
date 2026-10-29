#!/bin/bash
# lm-evaluation-harness suite (AyaRT, PolyTox, OR-Bench, MMLU, GSM8K, IFEval, BBH) for one model.
# Serve the judges first (eval_vllm_server.sbatch with Llama-3.3-70B-Instruct and Llama-Guard-4-12B)
# and pass their endpoint files:
#   MODEL_PATH=<hf id or dir> OUT=<dir> AYA_JUDGE_EP=<file> GUARD_EP=<file> \
#   sbatch [--account=... --partition=...] scripts/slurm/eval_lm_eval.sbatch [extra lm_eval_suite args]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${MODEL_PATH:?}" "${OUT:?}"
PY=${PYTHON:-python}
[[ -n "${AYA_JUDGE_EP:-}" ]] && export AYA_JUDGE_API_URL=$(cat "$AYA_JUDGE_EP") AYA_JUDGE_MODEL=${AYA_JUDGE_MODEL:-meta-llama/Llama-3.3-70B-Instruct}
[[ -n "${GUARD_EP:-}" ]] && export LLAMA_GUARD_API_URL=$(cat "$GUARD_EP") LLAMA_GUARD_MODEL=${LLAMA_GUARD_MODEL:-meta-llama/Llama-Guard-4-12B}
"$PY" -m msafetemp.evaluation.lm_eval_suite --model_path "$MODEL_PATH" --output_dir "$OUT" --tp "${TP:-1}" --dp "${DP:-4}" "$@"
