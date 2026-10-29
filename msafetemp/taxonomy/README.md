# msafetemp.taxonomy

LinguaSafe taxonomy labels for the English seed prompts.

WildGuardMix provides no useful harm breakdown for the canonical seeds (all carry the
subcategory `others`), so each English original is labelled once with the LinguaSafe
hierarchy (5 high-level categories, 23 subcategories) plus a `disability` subcategory under
Fairness & Justice. Labels are multi-label, ranked and capped at 3; `labels[0]` is the primary
label. The parent category is derived in code; an empty list (abstention) is a valid answer.
The non-consent gate (`is_non_consensual_sexual`) is enforced in code: when it fires,
`sex_related_crimes` becomes the primary label and `erotic_content` is removed. Labels describe
the underlying request, not the adversarial wrapper, and are inherited by every descendant row
through `original_id`.

| File | Purpose |
|---|---|
| `prompts.py` | classifier system prompt, clinical fallback, user template (verbatim) |
| `classify.py` | CLI: labels the canonical WildGuard seeds or the HarmBench behaviors |

```bash
python -m msafetemp.taxonomy.classify --track wildguard \
    --canonical $D/wildguard/canonical_2500.json \
    --judge_url $JUDGE_URL --judge_model openai/gpt-oss-safeguard-120b \
    --output $D/wildguard/classification
python -m msafetemp.taxonomy.classify --track harmbench --harmbench_raw $D/harmbench/raw \
    --judge_url $JUDGE_URL --judge_model openai/gpt-oss-safeguard-120b \
    --output $D/harmbench/classification
```

Settings: temperature 0, max_tokens 16,384, main
prompt twice then the clinical fallback. Output: `tags.jsonl` (resumable) and a consolidated
`canonical_2500_tags.json` / `harmbench_200_tags.json` read by the dataset builders.

In the released HarmBench labels, two behaviors were labelled by hand (one unparseable output,
one abstention); this tool leaves those unlabelled.
