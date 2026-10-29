# msafetemp.training

Safety fine-tuning with MSafeTemp data:

```
SFT mix  ->  SFT (Megatron-LM for Apertus, TRL for Llama / Olmo / Qwen)  ->  on-policy sampling
         ->  contrastive pairs  ->  DPO (TRL)
```

End-to-end commands: `scripts/examples/training_sft_data.sh` (data) and
`scripts/examples/training_sft_dpo.sh` (SFT and DPO job chains).

## Layout

| Module | Purpose |
|---|---|
| `data/select_response_rows.py` | response manifest: which audited WildGuard-train prompts get an SFT response (all benign and posthoc rows, harmful capped per language and template) |
| `data/build_safety_sft.py` | generated responses -> balanced safety SFT set `train_balanced.jsonl` (refuse:comply 1:1 per language, seed 111) |
| `data/adapt_safety_rows.py` | MSafeTemp safety SFT rows -> fields the mix / Megatron prepare step read (fail-closed checks) |
| `data/dolci_general.py` | Dolci-Instruct-SFT (no tools): `extract` single-turn pairs, `clean` its safety data |
| `data/short_prompts.py` | `vanilla`: short WildGuardMix-train prompts; `translate`: into the 43 non-English training languages |
| `data/short_contrast.py` | short benign contrast requests (generator + judge + evaluation-overlap guard) |
| `data/wrapped_prompts.py` | jailbreak-wrapped train prompts -> prompt datasets (`wrap1k`, `judged`, `split`) |
| `data/build_sft_mix.py` | **the final mix**: Megatron source + blend weights, and the 400k TRL sample |
| `sft/trl_sft.py` | TRL SFT (assistant-only loss, pre-rendered chat template) |
| `megatron_sft/` | Apertus Megatron-LM path: `tokenizer_view`, `prepare_dataset`, `plan`, `render_config` |
| `dpo/build_pool.py` | DPO prompt pool (126,458 train-source prompts) |
| `dpo/sample_contrastive.py` | on-policy sampling + judging (k=3 harmful / k=2 benign + forced refusal) |
| `dpo/build_pairs.py` | every contrastive pair the samples give (`--ref-chosen` variant optional) |
| `dpo/trl_dpo.py` | TRL DPO (sigmoid + 0.2 NLL) |
| `hf_utils.py` | generation-config sanitiser; `restore-config` for Olmo 3 outputs |

Configs: `configs/training/sft_mix.json` (bins, weights, seeds, expected counts),
`configs/training/apertus_megatron_sft.json` (Megatron run template), `configs/accelerate/fsdp2_*.yaml`
(`*_ac` = FSDP activation checkpointing). Slurm templates: `scripts/slurm/train_*`, `scripts/slurm/dpo_*`
(no cluster names; pass `--account/--partition` via `SBATCH_ARGS`, containers via `SRUN_ARGS`, a
virtualenv via `ENV_SETUP`).

## Endpoints

Every generator, judge and policy is an OpenAI-compatible chat-completions endpoint. Tools take a URL
file with one `http://host:port/v1` per line (one per replica) and balance across them client-side.

```bash
vllm serve google/gemma-4-31B-it --port 8080 --tensor-parallel-size 2 --max-model-len 16384   # generator / translator
vllm serve openai/gpt-oss-120b  --port 8080 --max-model-len 32768                              # judge
vllm serve $CKPT --served-model-name $NAME --port 8080 --max-model-len 16384                    # DPO policy
echo http://$(hostname):8080/v1 > urls_judge.txt
```

`scripts/slurm/dpo_serve_vllm.sbatch` does this on N nodes (one replica per node, TP 1 x DP 4) and writes the
URL file once all replicas are healthy. For Qwen 3.5 policies add
`--default-chat-template-kwargs '{"enable_thinking": false}'`.

## 1. SFT mix

| Data | Bin(s) | Target | SFT rows | Blend | DPO pairs |
|---|---|---|---:|---:|---:|
| Harmful | `harmful` | refuse | 91,642 | 4.2% | 1,758 |
| Culturally localized harmful | `posthoc` | refuse | 39,168 | 1.8% | 354 |
| Unwrapped harmful | `vanilla_harmful` | refuse | 20,724 | 2.5% | 1,278 |
| Jailbreak-wrapped harmful | `wrap_harmful` + `pyrit_harmful` | refuse | 56,258 | 2.0% | 1,698 |
| Benign direct | `benign_direct` | answer | 120,349 | 5.5% | 8,462 |
| Benign bounded | `benign_bounded` | answer | 10,713 | 0.5% | 830 |
| Unwrapped benign | `vanilla_benign` | answer | 22,413 | 4.5% | 21,166 |
| Jailbreak-wrapped benign | `wrap_benign` | answer | 22,446 | 2.5% | 7,817 |
| Short benign requests | `short_contrast` | answer | 25,103 | 3.0% | 14,909 |
| General instruction (Dolci) | `general` | answer | 1,851,189 | 73.5% | - |
| Total | | | 2,260,005 | 100% | 58,272 |

Exact blend weights (`sft_mix.json`): harmful .041979, posthoc .017942, benign_direct .055128,
benign_bounded .004907, vanilla_benign .045, vanilla_harmful .025, wrap_benign .025, wrap_harmful .015,
pyrit_harmful .005, short_contrast .03, general = remainder .735044.

* All safety answers come from the response generator (`msafetemp.augmentation.generate_responses`:
  Gemma-4-31B-it answers natively in the row's language, "decline, then help" refusals, GPT-OSS-120B
  judge, 2 attempts, accepted rows only).
* General rows: Dolci-Instruct-SFT (no-tools split), minus its WildGuardMix-derived rows and every
  reply that opens with a decline.
* Unwrapped prompts: all English vanilla WildGuardMix-train prompts plus a hash-ranked 300 benign /
  200 harmful per language translated (temperature 0) into the 43 other training languages.
* Short benign requests: for every short (<300 characters) English vanilla harmful prompt, 2 benign
  twins (types SWAP / PROTECT / ANALYZE / IDIOM / CREATIVE, generator temperature 0.9); GPT-OSS-120B
  keeps a twin only if it is benign and a direct task request; twins overlapping a HarmBench or OR-Bench
  prompt (character 4-gram Jaccard >= 0.5) are dropped; 500 per language translated. The mix keeps
  every sensitive-sounding twin plus innocuous SWAP / IDIOM / PROTECT twins.
* Wrapped prompts: 1,000 per language and subset, 5-10 wrappers per train-source payload, train
  families only (the held-out wrapped test families are listed in the released split), plus an earlier
  dual-judged wrapped set.

`build_sft_mix.py` writes `sft_source.jsonl` (Megatron: every row once, blend weights per bin) and
`sft_trl_400k.jsonl` (TRL: 400,000 rows with the same class proportions baked in; base bins sampled with
seed 20260924, then 12,000 general rows swapped for short-contrast rows with seed 20260925). It prints a
note if the bin counts differ from the reference mix.

## 2. SFT

Common: one epoch, loss on assistant tokens only, lr 1e-5 cosine, 3% warmup, weight decay 0, gradient
clipping 1.0, bf16, seed 1234, final checkpoint reported.

**Apertus-8B (Megatron-LM)**, `scripts/slurm/train_megatron_pipeline.sh`:
render config -> tokenizer view + tokenized bins (2% family-level validation split, stratified by
language) -> HF to TP-4 Megatron checkpoint -> packing plan (sequence length 4096, best-fit decreasing,
at most 8 documents per sequence; one packed epoch = 453,504 sequences = 3,543 steps at global batch
128) -> 1-node smoke -> 32 nodes x 4 GPUs (TP 4, micro batch 1, global batch 128) -> HF export.
Requires the Megatron-LM version released with Apertus (commit `512aae23559aff142f1488c4a3dee2192cfbba1a`,
provides `--ap-sft*`, `--xielu`, the Apertus checkpoint loader/saver and
`tools/preflight_sft_packing.py`), transformers with Apertus support (commit
`b9f90dc388fd415a2ba2a6a31a372f451d4a4eed`), Transformer Engine, and `huggingface-hub==1.0.0rc6`; the
reference runs used the NGC PyTorch 25.12 container on GH200 nodes. The base checkpoint (`KIND=base`) is
trained with the Instruct chat template.

**Llama 3.1 8B / Olmo 3 7B / Qwen 3.5 (TRL)**, `scripts/slurm/train_trl_sft_pipeline.sh`: the 400k TRL
sample, no packing, maximum length 4096, per-device batch 1 on 128 GPUs, no gradient accumulation (effective
batch 128), FSDP2. A 1-node warm job builds the datasets cache first. Software: trl 1.12.0,
transformers 5.16.1, accelerate 1.14.0, datasets 5.0.1 on the container's torch.

## 3. On-policy sampling

`dpo/build_pool.py` builds the pool (arms `v2_*` 15k + 15k safety prompts, `vanilla_*` all English short
prompts, `wrapped_*` 12k + 12k wrapped train prompts, `vanillaml_*` translated short prompts,
`short_contrast_{en,ml}`). `dpo/sample_contrastive.py` (via `scripts/slurm/dpo_sample.sbatch`) asks the
fine-tuned model for 3 answers per harmful and 2 per benign prompt (temperature 0.8, top-p 0.9, at most
4096 tokens) plus one refusal forced by a system instruction (temperature 0). Every answer passes the
validity gate and is labelled by GPT-OSS-120B with the evaluation's harm / over-refusal judge prompts
(`msafetemp.evaluation.judge_prompts`).

## 4. Pairs

`dpo/build_pairs.py`: harmful prompts give a pair when the model produced both a natural refusal
(chosen; the forced refusal if there is none) and an unsafe completion (rejected); benign prompts give a
pair from a helpful answer (chosen) and an over-refusal (rejected; natural if available, else the forced
refusal). Every pair is kept. Apertus-8B-Instruct + SFT: 58,272 pairs (5,088 harmful / 53,184 benign).
Most harmful prompts give no pair because the fine-tuned model never complies; harmful answers are never
taken from another model.

## 5. DPO

`dpo/trl_dpo.py` via `scripts/slurm/dpo_train.sbatch` (or the whole chain with
`scripts/slurm/dpo_pipeline.sh`): sigmoid DPO + NLL on the chosen answer (weight 0.2), beta 0.1, lr 5e-6
cosine with 10% warmup, 3 epochs, maximum length 4096, effective batch 128 pairs (32 nodes x 4 GPUs, GA 1),
FSDP2 with activation checkpointing, seed 1234. The reference model is the SFT checkpoint.

## Checkpoint hygiene

* `hf_utils.sanitise_generation_config` runs before and after training: vendor generation configs
  that combine `do_sample=False` with sampling fields fail strict validation in `save_pretrained`
  and would crash the final save.
* `python -m msafetemp.training.hf_utils restore-config OUT VENDOR_CONFIG` after TRL runs on Olmo 3:
  newer transformers save a nested `rope_parameters` config that vLLM rejects; the vendor config.json
  is restored (dtype bfloat16) and the trained one kept as `config.tf5.json`.
