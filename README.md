# MSafeTemp: Template-based Multilingual Safety Data Augmentation

[![arXiv](https://img.shields.io/badge/arXiv-2610.40516-b31b1b)](https://arxiv.org/abs/2610.40516)
[![Paper](https://img.shields.io/badge/Paper-HuggingFace-yellow)](https://huggingface.co/papers/2610.40516)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/swiss-ai/MSafeTemp)

This repository contains the official implementation of *"How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates"*.

## Authors

[Max Neuwinger](https://www.maxneuwinger.com/) | [Hanna Yukhymenko](https://www.ayukh.com/) | [Imanol Schlag](https://ischlag.github.io/)

## Abstract

Safety alignment for Large Language Models is heavily focused on English with limited multilingual representation, particularly damaging low-resource language performance. Related open training datasets are often limited in size or do not follow region-localized harmful topics to broaden the scope of alignment beyond cultures. We propose a scalable template-based augmentation pipeline to produce massive multilingual refusal and benign corpora, extracting safe templates from harmful prompts, translating them and then filling in the slots with culturally authentic entities, both harmful and benign. As a result, we release MSafeTemp, a validated corpus spanning 60 languages useful for multilingual safety alignment and evaluation. Using the dataset, we show that English-only evaluation misjudges multilingual safety: for several models, attacks that fail in English succeed once translated, while culturally localized variants expose further failures that translation alone misses. We also confirm that augmenting safety training data through templates allows for higher diversity and quality of alignment, resulting in improved safety performance beyond English. MSafeTemp and the respective pipeline are aimed at facilitating research and development of safer and more responsible models for less-represented cultures.

## Overview

The pipeline turns English harmful seed prompts into multilingual safety data:

![MSafeTemp pipeline](assets/pipeline.png)

Paraphrases, attack affixes and jailbreak wrappers extend each template, and every generation step is checked by an independent judge. The repository covers four parts that can be used separately:

* **Augmentation**: build TR / PH / OR data for new seed prompts, languages or cultural cards.
* **Evaluation**: score any model served behind an OpenAI-compatible endpoint on the HarmBench and WildGuard tracks, on jailbreak-wrapped prompts, and on lm-evaluation-harness safety and capability tasks.
* **Training data**: assemble a safety SFT mix (MSafeTemp refusals and benign answers plus general instruction data) and on-policy DPO pairs.
* **Training**: SFT with TRL (any Hugging Face model) or Megatron-LM (Apertus), followed by on-policy contrastive DPO.

**Languages**: 60, with 57 cultural cards for post-hoc localization

**Seed sources**: WildGuardMix (training), HarmBench (evaluation)

**Model serving**: any OpenAI-compatible endpoint (vLLM, hosted APIs); endpoint arguments accept one base URL or a comma-separated list of replicas

---

## Installation

Python 3.12.

```bash
git clone https://github.com/swiss-ai/msafetemp.git
cd msafetemp
pip install -r requirements.txt          # augmentation, dataset build, evaluation clients
pip install -r requirements-train.txt    # TRL SFT / DPO (separate environment, on a PyTorch container)
pip install -r requirements-eval.txt     # vLLM serving and lm-evaluation-harness tasks
```

Run everything from the repository root (or add it to `PYTHONPATH`). Every module runs as `python -m msafetemp.<module>` and has `--help`. API keys are read from `OPENAI_API_KEY` / `JUDGE_API_KEY` (a local vLLM server accepts any value). Apertus SFT uses Megatron-LM with Apertus support, which is not pip-installable; see `msafetemp/training/README.md`.

---

## Quick Start

### Evaluate a model

Evaluate one model on the HarmBench track (TR, PH and OR) with the released dataset:

```bash
export MSAFETEMP_DATA_DIR=/path/to/msafetemp_data MSAFETEMP_OUT_DIR=/path/to/outputs
vllm serve openai/gpt-oss-safeguard-120b --port 8001 &            # response judge
vllm serve swiss-ai/Apertus-8B-Instruct-2509 --port 8002 &        # target
python -m msafetemp.evaluation.harmbench_eval --dataset_path $MSAFETEMP_DATA_DIR/harmbench/hf \
    --base_url http://localhost:8002/v1 --model swiss-ai/Apertus-8B-Instruct-2509 \
    --judge_url http://localhost:8001/v1 --judge_model openai/gpt-oss-safeguard-120b \
    --posthoc --benign --max_tokens 8192 --output $MSAFETEMP_OUT_DIR/hb/Apertus-8B-Instruct-2509
python -m msafetemp.evaluation.aggregate --track hb --protocol breakdown \
    --run Apertus-8B-Instruct-2509=$MSAFETEMP_OUT_DIR/hb/Apertus-8B-Instruct-2509 --out_csv hb.csv
```

The WildGuard track works the same way with `msafetemp.evaluation.wildguard_eval --splits test`. `msafetemp/evaluation/README.md` describes the arms, judging, the response validity gate and the aggregation protocols.

### Augment new data

Each augmentation stage is a CPU client of OpenAI-compatible endpoints (a generator and a judge). `scripts/examples/augmentation_wildguard.sh` runs the full chain for WildGuardMix seeds: generation, canonical seeds, benign pairs, taxonomy labels, post-hoc localization, dataset build and benign audit. To add a language, add its cultural card (`data/cultural_cards/`, guideline in `msafetemp/cultural_cards/`) and pass the language to the stages.

### Inspect the data

`data/sample.json` holds 232 released rows across all configurations, splits and branches (including jailbreak-wrapped prompts) in 12 languages. The full dataset layout is described in `docs/DATA.md`.

---

## Repository Structure

```
msafetemp/
  augmentation/     template extraction, translation, TR fills, benign pairs, post-hoc localization,
                    paraphrases, affixes, dataset builders, benign audit, SFT response generator,
                    jailbreak wrappers (wrappers/), all prompts (prompts/)
  cultural_cards/   card generation guideline, rendering, draft generation, card loader
  taxonomy/         LinguaSafe labelling of seed prompts
  training/         SFT data (data/), TRL SFT (sft/), Megatron-LM SFT for Apertus (megatron_sft/),
                    on-policy DPO (dpo/)
  evaluation/       HarmBench / WildGuard track clients, response judge, aggregation,
                    jailbreak-wrapped evaluation, lm-evaluation-harness tasks
  common/           OpenAI-compatible client with retries, IO and config helpers
configs/            generation settings, judges, evaluation protocols, SFT mix, accelerate (FSDP2)
data/cultural_cards/  57 cultural cards (JSON)
data/sample.json    small sample of the released data
docs/DATA.md        layout of the released data ($MSAFETEMP_DATA_DIR)
scripts/examples/   end-to-end example commands per stage
scripts/slurm/      generic Slurm templates (no cluster-specific settings)
assets/             README figures
```

Each subpackage has its own README with settings and output schemas.

---

## End-to-End Pipeline

Set `MSAFETEMP_DATA_DIR` (released data) and `MSAFETEMP_OUT_DIR` (outputs). Cluster flags for the Slurm templates go into `SBATCH_ARGS`, containers into `SRUN_ARGS`, environment activation into `ENV_SETUP`.

1. **Augmentation.** `scripts/examples/augmentation_wildguard.sh` (WildGuardMix track), `scripts/examples/augmentation_harmbench.sh` (HarmBench track), `scripts/examples/augmentation_wrappers_and_cards.sh` (wrapper bank, wrapped prompts, cultural cards). `scripts/slurm/augment_client.sbatch` runs one stage on a node.
2. **Training data.** `scripts/examples/training_sft_data.sh`: safety SFT rows (released `train_balanced.jsonl`, or rebuilt with `REBUILD_SAFETY_SFT=1`), general instruction data (Dolci-Instruct-SFT), short unwrapped prompts, short benign contrast requests, wrapped prompts, and the final mix (`sft_source.jsonl` for Megatron-LM, `sft_trl_400k.jsonl` for TRL).
3. **Training.** `scripts/examples/training_sft_dpo.sh`: SFT with Megatron-LM (`scripts/slurm/train_megatron_pipeline.sh`) or TRL (`scripts/slurm/train_trl_sft_pipeline.sh`), then the DPO prompt pool, on-policy sampling, pairs and DPO (`scripts/slurm/dpo_pipeline.sh`).
4. **Evaluation.** `scripts/examples/evaluation_tracks.sh` (`PROTOCOL=full` for the full test sets, `PROTOCOL=light` for fast checkpoint evaluation), `scripts/examples/evaluation_wrapped.sh` (jailbreak-wrapped HarmBench), `scripts/examples/evaluation_lm_eval.sh` (AyaRT, PolyTox, OR-Bench, capability tasks).

---

## Models

Default models per role (configurable on every CLI):

| Role | Model |
|---|---|
| Generator (templates, translation, fills, localization, SFT responses) | `google/gemma-4-31B-it` |
| Generation-side judges, response judge for SFT data, DPO judge | `openai/gpt-oss-120b` |
| Evaluation judge, taxonomy classifier | `openai/gpt-oss-safeguard-120b` |
| AyaRT judge / PolyTox classifier (lm-evaluation-harness) | `meta-llama/Llama-3.3-70B-Instruct` / `meta-llama/Llama-Guard-4-12B` |

`scripts/slurm/eval_vllm_server.sbatch` and `scripts/slurm/dpo_serve_vllm.sbatch` start vLLM servers and write their URLs to files.

---

## Responsible Use

MSafeTemp contains harmful prompts and jailbreak wrappers in 60 languages. It is released to train and evaluate safer models; do not use it to build systems that produce harmful content.

## License

The code is released under the [MIT License](LICENSE).

## Citation

```bibtex
@misc{neuwinger2026ialignedmodelscaling,
      title={How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates}, 
      author={Max Neuwinger and Hanna Yukhymenko and Imanol Schlag},
      year={2026},
      eprint={2610.40516},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.40516}, 
}
```
