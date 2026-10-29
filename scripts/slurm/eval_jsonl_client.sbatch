#!/bin/bash
# Fixed-input evaluation (wrapped HB300 round, matched multi-wrapper) for one target.
#   MODEL_KEY=apertus8b TARGET_MODEL=<served name> TARGET_URL=<url or file> JUDGE_URL=<url or file> \
#   INPUTS="inputs/harmful.jsonl inputs/benign.jsonl" RUNS_DIR=<dir> [CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'] \
#   sbatch [--account=... --partition=...] scripts/slurm/eval_jsonl_client.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${MODEL_KEY:?}" "${TARGET_MODEL:?}" "${TARGET_URL:?}" "${JUDGE_URL:?}" "${INPUTS:?}" "${RUNS_DIR:?}"
PY=${PYTHON:-python}
resolve() { if [[ -f "$1" ]]; then cat "$1"; else echo "$1"; fi; }
read -r -a IN <<< "$INPUTS"
CTK=${CHAT_TEMPLATE_KWARGS:-"{}"}
"$PY" -m msafetemp.evaluation.jsonl_eval --inputs "${IN[@]}" --out_dir "$RUNS_DIR/$MODEL_KEY" \
  --model_key "$MODEL_KEY" --model "$TARGET_MODEL" --base_url "$(resolve "$TARGET_URL")" \
  --judge_url "$(resolve "$JUDGE_URL")" --judge_model "${JUDGE_MODEL:-openai/gpt-oss-safeguard-120b}" \
  --kinds "${KINDS:-harmful,benign}" --chat_template_kwargs "$CTK" \
  --target_conc "${TARGET_CONC:-48}" --judge_conc "${JUDGE_CONC:-4}" --done_file "$RUNS_DIR/$MODEL_KEY/DONE"
