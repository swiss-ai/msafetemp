#!/bin/bash
# On-policy contrastive DPO sampling client (msafetemp/training/dpo/sample_contrastive.py) over the full
# prompt pool. Waits for the policy and judge URL files (dpo_serve_vllm.sbatch writes them once every
# replica is healthy), then runs one sampler process per arm group: the client is bound to one CPU
# core, so disjoint arm groups run in parallel into separate output dirs.
#
# Required env: POOL, OUT, POLICY_URLS (URL file), POLICY_MODEL (served name), JUDGE_URLS
# Optional env: JUDGE_MODEL (openai/gpt-oss-120b), ARM_GROUPS (';'-separated groups of arms, default:
#   all 12 arms in 7 groups), K_REFUSE (3), K_BENIGN (2), WORKERS (1000), MSAFETEMP_ROOT, ENV_SETUP.
#SBATCH --job-name=dpo-sample
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --no-requeue
set -euo pipefail
MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$SLURM_SUBMIT_DIR}
: "${POOL:?}" "${OUT:?}" "${POLICY_URLS:?}" "${POLICY_MODEL:?}" "${JUDGE_URLS:?}"
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
ARM_GROUPS=${ARM_GROUPS:-"v2_harmful v2_posthoc;v2_benign_direct v2_benign_bounded;vanilla_harmful vanilla_benign;wrapped_harmful wrapped_benign;vanillaml_harmful vanillaml_benign;short_contrast_en;short_contrast_ml"}
eval "${ENV_SETUP:-true}"
export PYTHONPATH=$MSAFETEMP_ROOT${PYTHONPATH:+:$PYTHONPATH}
export no_proxy="*" NO_PROXY="*"
for f in "$POLICY_URLS" "$JUDGE_URLS"; do
  for i in $(seq 1 240); do [[ -s "$f" ]] && break; sleep 30; done
  [[ -s "$f" ]] || { echo "no URL file $f" >&2; exit 1; }
done
mkdir -p "$OUT"
cp "$POLICY_URLS" "$OUT/urls_policy.txt"; cp "$JUDGE_URLS" "$OUT/urls_judge.txt"
echo "policy $(wc -l < "$OUT/urls_policy.txt") replicas, judge $(wc -l < "$OUT/urls_judge.txt") replicas $(date)"
pids=()
IFS=';' read -ra GROUPS_ARR <<< "$ARM_GROUPS"
for grp in "${GROUPS_ARR[@]}"; do
  tag=${grp%% *}
  # shellcheck disable=SC2086
  python -B -m msafetemp.training.dpo.sample_contrastive --pool "$POOL" --arms $grp --output "$OUT/$tag" \
    --policy_url_file "$OUT/urls_policy.txt" --policy_model "$POLICY_MODEL" \
    --judge_url_file "$OUT/urls_judge.txt" --judge_model "$JUDGE_MODEL" \
    --k_refuse "${K_REFUSE:-3}" --k_benign "${K_BENIGN:-2}" --workers "${WORKERS:-1000}" > "$OUT/log_$tag.txt" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { echo "a sampler failed; see $OUT/log_*.txt" >&2; exit 1; }
echo "SAMPLE_DONE $(date)"
