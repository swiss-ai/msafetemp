#!/bin/bash
# Serve one model (target, response judge or lm-eval judge) with `vllm serve` on one node and
# write its OpenAI-compatible URL (http://<ip>:<port>/v1) to $ENDPOINT_DIR/$SLURM_JOB_NAME.endpoint
# once /v1/models answers. Clients read that file.
#
#   sbatch --job-name=judge-safeguard [--account=... --partition=...] scripts/slurm/eval_vllm_server.sbatch \
#       openai/gpt-oss-safeguard-120b 4 32768 [extra vllm args...]
# Examples of extra args: --dtype bfloat16 (fp32 checkpoints),
#   --default-chat-template-kwargs '{"enable_thinking": false}' (Qwen3.5 targets).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
MODEL=${1:?model id or path}; TP=${2:-4}; MAXLEN=${3:-32768}; shift $(( $# < 3 ? $# : 3 ))
PORT=${PORT:-8000}
ENDPOINT_DIR=${ENDPOINT_DIR:-${MSAFETEMP_OUT_DIR:-$PWD}/endpoints}
SERVED_NAME=${SERVED_NAME:-$MODEL}
mkdir -p "$ENDPOINT_DIR"
EP_FILE=$ENDPOINT_DIR/$SLURM_JOB_NAME.endpoint
rm -f "$EP_FILE"
IP=$(hostname -i | awk '{print $1}')
echo "node $(hostname) ip $IP model $MODEL tp $TP max_model_len $MAXLEN"

vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --gpu-memory-utilization 0.9 "$@" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 360); do
  if curl -sf -m 5 "http://$IP:$PORT/v1/models" >/dev/null; then
    echo "http://$IP:$PORT/v1" > "$EP_FILE"; echo "healthy after $((i * 10))s -> $(cat "$EP_FILE")"; break
  fi
  kill -0 $SERVER_PID 2>/dev/null || { echo "vllm exited before becoming healthy" >&2; exit 1; }
  sleep 10
done
wait $SERVER_PID
