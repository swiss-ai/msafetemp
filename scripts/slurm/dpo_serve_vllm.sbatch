#!/bin/bash
# Serve a model with vLLM on every node of the allocation (one OpenAI-compatible replica per node) and
# write the replica URLs (http://<host>:<PORT>/v1, one per line) to URL_FILE. Runs until cancelled.
# Used for the DPO policy (the fine-tuned model) and judge (default: openai/gpt-oss-120b); any
# OpenAI-compatible endpoint works instead, e.g. a single `vllm serve <model> --port 8080`.
#
# Required env: MODEL (HF dir or id), SERVED_NAME (name clients send as `model`), URL_FILE
# Optional env: PORT (8080), TP (1), DP (4 = GPUs per node / TP), MAX_LEN (16384; judge 32768),
#   VLLM_EXTRA (e.g. for Qwen 3.5 policies: --default-chat-template-kwargs '{"enable_thinking": false}'),
#   ENV_SETUP, SRUN_ARGS.
# Reference configuration: policy 8 nodes, judge 8 nodes, each node TP 1 x DP 4 on 4 GH200.
#SBATCH --job-name=vllm-serve
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-requeue
set -euo pipefail
: "${MODEL:?}" "${SERVED_NAME:?}" "${URL_FILE:?}"
PORT=${PORT:-8080}; TP=${TP:-1}; DP=${DP:-4}; MAX_LEN=${MAX_LEN:-16384}
mkdir -p "$(dirname "$URL_FILE")"
: > "$URL_FILE.partial"
for h in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do echo "http://$h:$PORT/v1" >> "$URL_FILE.partial"; done
export MODEL SERVED_NAME PORT TP DP MAX_LEN VLLM_EXTRA="${VLLM_EXTRA:-}" ENV_SETUP="${ENV_SETUP:-}"
# the URL file appears only once every replica answers /health (clients wait for it)
(
  for u in $(cat "$URL_FILE.partial"); do
    until curl -sf --noproxy '*' -m 10 "${u%/v1}/health" > /dev/null; do sleep 20; done
  done
  mv "$URL_FILE.partial" "$URL_FILE"; echo "all replicas healthy: $URL_FILE"
) &
# shellcheck disable=SC2086
srun ${SRUN_ARGS:-} --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  eval "${ENV_SETUP}"
  eval exec vllm serve "${MODEL}" --served-model-name "${SERVED_NAME}" --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size "${TP}" --data-parallel-size "${DP}" --max-model-len "${MAX_LEN}" ${VLLM_EXTRA}
'
