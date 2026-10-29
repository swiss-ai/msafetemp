#!/bin/bash
# Generic CPU client for one augmentation stage. The stages only talk to OpenAI-compatible
# endpoints (serve the generator and judge first, e.g. with scripts/slurm/eval_vllm_server.sbatch)
# and hold their work queue in memory, so they run on one CPU node, not on a login node.
#
#   GEN_URL=<url, url list or endpoint file> JUDGE_URL=<...> \
#   STAGE=msafetemp.augmentation.posthoc STAGE_ARGS="--track wildguard --raw_dir ... --output ..." \
#   sbatch [--account=... --partition=...] scripts/slurm/augment_client.sbatch
#
# The stage command gets "--target_url/--base_url/--gen_url" style arguments from STAGE_ARGS;
# use the placeholders @GEN_URL@ and @JUDGE_URL@ inside STAGE_ARGS, they are replaced with the
# resolved endpoints (an endpoint file is read; several lines are joined with commas).
# Optional env: PYTHON (python), ENV_SETUP (command run first, e.g. activating an env),
# MSAFETEMP_ROOT (repo root, default: submit dir).
#SBATCH --job-name=augment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${STAGE:?module to run, e.g. msafetemp.augmentation.generate}" "${STAGE_ARGS:?}"
MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$SLURM_SUBMIT_DIR}
eval "${ENV_SETUP:-true}"
export PYTHONPATH=$MSAFETEMP_ROOT${PYTHONPATH:+:$PYTHONPATH}
resolve() { if [[ -f "$1" ]]; then paste -sd, "$1"; else echo "$1"; fi; }
ARGS=${STAGE_ARGS//@GEN_URL@/$(resolve "${GEN_URL:-}")}
ARGS=${ARGS//@JUDGE_URL@/$(resolve "${JUDGE_URL:-}")}
echo "stage $STAGE  started $(date)"
# shellcheck disable=SC2086
srun "${PYTHON:-python}" -u -m "$STAGE" $ARGS
echo "finished $(date)"
