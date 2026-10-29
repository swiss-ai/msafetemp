#!/bin/bash
# On-policy contrastive DPO for one safety-SFT checkpoint, as a chain of Slurm jobs:
#   serve policy (CKPT) + judge with vLLM  ->  sample the full prompt pool (k=3 harmful / k=2 benign +
#   one forced refusal, every answer judged)  ->  release the servers  ->  build every pair  ->
#   1-node warm  ->  NODES-node TRL DPO  ->  optional vendor config restore (Olmo 3).
#
#   NAME=Apertus-8B-Instruct-2509-MSafeTemp-SFT CKPT=$OUT/Apertus-8B-Instruct-2509-MSafeTemp-SFT \
#   POOL=$DPO/pool.jsonl ACC_CFG=configs/accelerate/fsdp2_apertus_ac.yaml OUT_ROOT=$DPO \
#   bash scripts/slurm/dpo_pipeline.sh
#
# Optional env:
#   DEP             job id producing CKPT (servers and sampler start after it)
#   SERVE=0         do not start servers; use existing endpoints POLICY_URLS / JUDGE_URLS (URL files)
#                   and POLICY_MODEL (served name of the policy, default NAME)
#   SAMPLES         existing samples dir: skip serving and sampling, build pairs from it
#   JUDGE_MODEL     judge checkpoint to serve and its served name (default openai/gpt-oss-120b)
#   POLICY_VLLM_EXTRA  extra vllm flags for the policy (Qwen 3.5: --default-chat-template-kwargs '{"enable_thinking": false}')
#   POLICY_NODES (8), JUDGE_NODES (8), NODES (32, DPO training)
#   REF_CHOSEN      SFT source jsonl: variant that uses the SFT answer as chosen where the model never complied
#   ORIGCFG         vendor config.json restored into the DPO output (Olmo 3)
#   SBATCH_ARGS     cluster flags for every job (e.g. "--account=X --partition=Y")
#   ENV_SETUP, SRUN_ARGS, MSAFETEMP_ROOT
set -euo pipefail
: "${NAME:?}" "${CKPT:?}" "${POOL:?}" "${ACC_CFG:?}" "${OUT_ROOT:?}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MSAFETEMP_ROOT=${MSAFETEMP_ROOT:-$(cd "$HERE/../.." && pwd)}
SB="sbatch --parsable ${SBATCH_ARGS:-}"
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
R=$OUT_ROOT/$NAME
mkdir -p "$R"
AFTER=${DEP:+--dependency=afterok:$DEP}

if [[ -z "${SAMPLES:-}" ]]; then
  SAMPLES=$R/samples
  if [[ "${SERVE:-1}" == 1 ]]; then
    POLICY_URLS=$R/urls_policy.txt; JUDGE_URLS=$R/urls_judge.txt
    rm -f "$POLICY_URLS" "$JUDGE_URLS"
    POL=$(MODEL=$CKPT SERVED_NAME=$NAME URL_FILE=$POLICY_URLS MAX_LEN=16384 VLLM_EXTRA="${POLICY_VLLM_EXTRA:-}" \
          $SB $AFTER --nodes=${POLICY_NODES:-8} --job-name=serve-policy --export=ALL "$HERE/dpo_serve_vllm.sbatch")
    JUD=$(MODEL=$JUDGE_MODEL SERVED_NAME=$JUDGE_MODEL URL_FILE=$JUDGE_URLS MAX_LEN=32768 VLLM_EXTRA= \
          $SB $AFTER --nodes=${JUDGE_NODES:-8} --job-name=serve-judge --export=ALL "$HERE/dpo_serve_vllm.sbatch")
  else
    : "${POLICY_URLS:?}" "${JUDGE_URLS:?}"
  fi
  SA=$(POOL=$POOL OUT=$SAMPLES POLICY_URLS=$POLICY_URLS POLICY_MODEL=${POLICY_MODEL:-$NAME} JUDGE_URLS=$JUDGE_URLS JUDGE_MODEL=$JUDGE_MODEL \
       $SB $AFTER --job-name=dpo-sample --export=ALL "$HERE/dpo_sample.sbatch")
  if [[ -n "${POL:-}" ]]; then
    RL=$($SB --dependency=afterany:$SA --nodes=1 --ntasks=1 --time=00:05:00 --job-name=release-servers \
         --wrap "scancel $POL $JUD; echo released")
    echo "policy=$POL judge=$JUD release=$RL"
  fi
  AFTER="--dependency=afterok:$SA"
  echo "sample=$SA -> $SAMPLES"
fi

VARIANT=full; EXTRA=""
if [[ -n "${REF_CHOSEN:-}" ]]; then VARIANT=fullref; EXTRA="--ref-chosen $REF_CHOSEN"; fi
PAIRS=$R/dpo_pairs_$VARIANT.jsonl
OUT=$R/$NAME-DPO
PA=$($SB $AFTER --nodes=1 --ntasks=1 --time=00:30:00 --job-name=dpo-pairs \
     --wrap "cd $MSAFETEMP_ROOT && ${ENV_SETUP:-true} && python -B -m msafetemp.training.dpo.build_pairs --samples $SAMPLES --pool $POOL --output $PAIRS $EXTRA")
export MODEL=$CKPT OUTPUT=$OUT DATA=$PAIRS ACC_CFG
WU=$(PHASE=warm $SB --dependency=afterok:$PA --nodes=1 --job-name=dpo-warm --export=ALL "$HERE/dpo_train.sbatch")
TR=$(PHASE=train $SB --dependency=afterok:$WU --nodes=${NODES:-32} --time=08:00:00 --job-name=dpo-train \
     --export=ALL "$HERE/dpo_train.sbatch")
POST=$TR
if [[ -n "${ORIGCFG:-}" ]]; then
  POST=$($SB --dependency=afterok:$TR --nodes=1 --ntasks=1 --time=00:10:00 --job-name=dpo-restore-config \
         --wrap "cd $MSAFETEMP_ROOT && ${ENV_SETUP:-true} && python -m msafetemp.training.hf_utils restore-config $OUT $ORIGCFG")
fi
echo "pairs=$PA warm=$WU train=$TR post=$POST output=$OUT"
