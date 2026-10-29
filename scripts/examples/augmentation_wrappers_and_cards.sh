#!/bin/bash
# Jailbreak-wrapper bank, wrapped training / evaluation prompts, and cultural cards.
# Directory names follow docs/DATA.md.
#   PYRIT_TEMPLATES: local copy of the PyRIT jailbreak template directory (YAML files)
#   QC_URL / QC_MODEL: multilingual quality model used for the wrapper translation checks
set -euo pipefail
: "${MSAFETEMP_DATA_DIR:?}" "${GEN_URL:?}" "${JUDGE_URL:?}" "${QC_URL:?}" "${QC_MODEL:?}" "${PYRIT_TEMPLATES:?}"
D=$MSAFETEMP_DATA_DIR
BANK=$D/wrappers/wrapper_bank.jsonl
GEN_MODEL=${GEN_MODEL:-google/gemma-4-31B-it}
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
LANGS=$(python -c "from msafetemp.cultural_cards import SUPPORTED_LANGUAGES as L; print(' '.join(L))")

# 1. Translate the 557 single-slot wrappers into every language (checked, resumable)
python -m msafetemp.augmentation.wrappers.translate --corpus_dir "$PYRIT_TEMPLATES" --languages $LANGS \
  --gen_url "$GEN_URL" --gen_model "$GEN_MODEL" --quality_url "$QC_URL" --quality_model "$QC_MODEL" \
  --output "$BANK"

# 2. Wrapped evaluation candidates (released as wrapped_eval/, docs/DATA.md): 10 wrappers per post-hoc
#    test prompt of both tracks (matched multi-wrapper evaluation, PH-JB rows of the HB300 round) and
#    per benign HarmBench prompt (HB300 over-refusal half). Blocked combinations are dropped here.
E=$D/wrapped_eval
mkdir -p "$E"
assembled() { python -c "import json,sys; [print(l, end='') for l in open(sys.argv[1]) if json.loads(l).get('status') == 'assembled']" "$1"; }
python -m msafetemp.augmentation.wrappers.assemble --mode per_source --k 10 --bank "$BANK" \
  --dataset "$D/wildguard/hf_audited" --track wildguard --categories posthoc --splits test --out "$E/wg_posthoc.all.jsonl"
python -m msafetemp.augmentation.wrappers.assemble --mode per_source --k 10 --bank "$BANK" \
  --dataset "$D/harmbench/hf" --track harmbench --categories posthoc --splits test --out "$E/hb_posthoc.all.jsonl"
python -m msafetemp.augmentation.wrappers.assemble --mode per_source --k 10 --bank "$BANK" \
  --dataset "$D/harmbench/hf" --track harmbench --categories benign --splits test --out "$E/hb_benign.all.jsonl"
assembled "$E/hb_posthoc.all.jsonl" > "$E/harmbench_posthoc_wrapped_test.jsonl"
assembled "$E/hb_benign.all.jsonl" > "$E/harmbench_benign_wrapped.jsonl"
cat "$E/harmbench_posthoc_wrapped_test.jsonl" <(assembled "$E/wg_posthoc.all.jsonl") > "$E/posthoc_wrapped_test.jsonl"

# 3. Wrapped training prompts: 1,000 per language and subset, 5-10 wrappers per train-source payload
#    (-> wrapped_train/{harmful,benign}/rows.jsonl; the family-level train / test split of these rows
#    is released as wrapped_train/split/train_ids.txt, see msafetemp.training.data.wrapped_prompts)
for sub in harmful benign; do
  mkdir -p "$D/wrapped_train/$sub"
  python -m msafetemp.augmentation.wrappers.assemble --mode per_language --per_language 1000 \
    --bank "$BANK" --dataset "$D/wildguard/hf_audited" \
    --categories $sub --splits train --subset $sub --out "$D/wrapped_train/$sub/rows.jsonl"
done

# 4. Earlier dual-judged wrapped set (released as wrapped_judged/, see docs/DATA.md): localized
#    harmful payloads in every language ("full") plus English-only harmful and benign payloads
#    ("english_harmful", "english_benign", assembled the same way from a payload JSONL restricted
#    to English via --payloads). Only accepted train-source rows enter the SFT mix.
J=$D/wrapped_judged
mkdir -p "$J/full" "$J/judgments"
python -m msafetemp.augmentation.wrappers.assemble --mode per_language --per_language 1000 \
  --bank "$BANK" --dataset "$D/wildguard/hf_audited" --categories posthoc --splits train test \
  --subset posthoc --out "$J/full/rows.jsonl"
for name in full english_harmful english_benign; do
  [[ -f "$J/$name/rows.jsonl" ]] || continue
  arm=harmful; [[ $name == english_benign ]] && arm=benign
  python -m msafetemp.augmentation.wrappers.judge --rows "$J/$name/rows.jsonl" --arm $arm \
    --bank "$BANK" --quality_url "$QC_URL" --quality_model "$QC_MODEL" \
    --safety_url "$JUDGE_URL" --safety_model "$JUDGE_MODEL" --output "$J/judgments/${name}_judgments.jsonl"
done

# 5. Draft a cultural card for a new language (review manually before use)
# python -m msafetemp.cultural_cards.generate --language german --base_url "$GEN_URL" \
#   --model "$GEN_MODEL" --out drafts/german.json
