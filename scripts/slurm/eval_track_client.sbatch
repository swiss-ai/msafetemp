#!/bin/bash
# HarmBench or WildGuard track client for one target, sharded over languages. The client runs
# on one asyncio loop (about one CPU core), so SHARDS processes evaluate disjoint language subsets
# into the same output tree (one dir per arm, resume by row id); a final all-language pass then
# finds nothing pending and writes the combined summary.json.
#
#   TRACK=hb|wg TARGET_MODEL=<served name> TARGET_URL=<url or file> JUDGE_URL=<url or file> \
#   OUT=<output dir> EXTRA_ARGS="--benign --max_tokens 4096" \
#   sbatch [--account=... --partition=...] scripts/slurm/eval_track_client.sbatch
# TARGET_URL / JUDGE_URL may be a URL (comma-separated replicas allowed) or an .endpoint file
# written by eval_vllm_server.sbatch. Protocol flags per table: configs/evaluation/protocols.yaml.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${TRACK:?hb or wg}" "${TARGET_MODEL:?}" "${TARGET_URL:?}" "${JUDGE_URL:?}" "${OUT:?}"
PY=${PYTHON:-python}
SHARDS=${SHARDS:-6}
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-safeguard-120b}
resolve() { if [[ -f "$1" ]]; then cat "$1"; else echo "$1"; fi; }
T=$(resolve "$TARGET_URL"); J=$(resolve "$JUDGE_URL")
if [[ $TRACK == hb ]]; then
  MOD=msafetemp.evaluation.harmbench_eval; DS=${MSAFETEMP_HB_DATASET:?}
else
  MOD=msafetemp.evaluation.wildguard_eval; DS=${MSAFETEMP_WG_DATASET:?}
fi
read -r -a XARGS <<< "${EXTRA_ARGS:-}"
mkdir -p "$OUT/shard_logs"
LANGS=($("$PY" - "$DS" "$TRACK" <<'PYEOF'
import sys
from datasets import load_from_disk
ds = load_from_disk(sys.argv[1])
print(" ".join(sorted(set(ds["grouped_test"]["language"]))))
PYEOF
))
run() {
  "$PY" -m "$MOD" --dataset_path "$DS" --base_url "$T" --model "$TARGET_MODEL" \
    --judge_url "$J" --judge_model "$JUDGE_MODEL" --workers "${WORKERS:-1000}" \
    --checkpoint_every 50 --output "$OUT" "${XARGS[@]}" "$@"
}
echo "track=$TRACK target=$TARGET_MODEL @ $T | judge @ $J | ${#LANGS[@]} languages | out=$OUT"
pids=()
for s in $(seq 0 $((SHARDS - 1))); do
  sub=(); for k in "${!LANGS[@]}"; do (( k % SHARDS == s )) && sub+=("${LANGS[$k]}"); done
  extra=(--languages "${sub[@]}"); (( s == 0 )) || extra+=(--no_original)
  run "${extra[@]}" > "$OUT/shard_logs/shard_$s.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { echo "a shard failed; see $OUT/shard_logs" >&2; exit 1; }
run > "$OUT/shard_logs/final.log" 2>&1
echo "EVAL_DONE $TARGET_MODEL"
