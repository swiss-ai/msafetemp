#!/bin/bash
# Jailbreak-wrapped HarmBench evaluation (HB-JB): culturally localized HarmBench requests inside
# jailbreak wrappers (ASR) and wrapped benign prompts (over-refusal), 300 prompts per language.
set -euo pipefail
: "${MSAFETEMP_OUT_DIR:?}" "${MSAFETEMP_DATA_DIR:?}"
W=$MSAFETEMP_OUT_DIR/wrapped_hb300
JUDGE_URL=${JUDGE_URL:-$MSAFETEMP_OUT_DIR/endpoints/judge-safeguard.endpoint}

# 1) freeze 300 prompts per language (released candidate rows, docs/DATA.md: wrapped PH HarmBench
#    test rows and wrapped benign HarmBench rows, JSONL with id, track, language, prompt)
python -m msafetemp.evaluation.wrapped_hb300 select \
  --harmful $MSAFETEMP_DATA_DIR/wrapped_eval/harmbench_posthoc_wrapped_test.jsonl \
  --benign  $MSAFETEMP_DATA_DIR/wrapped_eval/harmbench_benign_wrapped.jsonl --out_dir $W/inputs

# 2) one client per target (see configs/evaluation/targets.yaml; Qwen3.5 with thinking off)
MODEL_KEY=apertus8b TARGET_MODEL=swiss-ai/Apertus-8B-Instruct-2509 \
TARGET_URL=$MSAFETEMP_OUT_DIR/endpoints/target-apertus8b.endpoint JUDGE_URL=$JUDGE_URL \
INPUTS="$W/inputs/harmful.jsonl $W/inputs/benign.jsonl" RUNS_DIR=$W/runs \
  sbatch --job-name=hb300-apertus8b --export=ALL scripts/slurm/eval_jsonl_client.sbatch

# 3) summary (macro over languages), after the clients finish
echo python -m msafetemp.evaluation.wrapped_hb300 summarize --inputs_dir $W/inputs --runs_dir $W/runs \
  --model "apertus8b=Apertus-8B-Instruct-2509" --out_dir $W/report
