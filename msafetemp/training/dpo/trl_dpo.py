#!/usr/bin/env python
"""DPO on the on-policy contrastive pairs with TRL DPOTrainer (TRL 1.12).

Recipe: start from the safety-SFT checkpoint, sigmoid DPO loss plus an NLL term on the chosen
answer (loss_type=["sigmoid", "sft"], weights [1.0, 0.2]; TRL's spelling of RPO), beta 0.1,
lr 5e-6 cosine with 10% warmup, weight decay 0, grad clip 1.0, 3 epochs, maximum length 4096
(truncation keeps the start), bf16, seed 1234, per-device batch 1 on 128 GPUs with no gradient
accumulation (effective batch 128 pairs), FSDP2 with activation checkpointing
(configs/accelerate/*_ac.yaml: chosen + rejected at 4096 tokens run out of memory without it).

Input rows (build_pairs.py): {id, arm, language, user, chosen, rejected}, plain strings. The prompt's
chat template is rendered here and TRL gets the standard (non-conversational) format, for the reason
given in sft/trl_sft.py (the Apertus / Qwen generation prompt is not a clean prefix of prompt +
answer). For plain strings DPOTrainer appends eos_token to chosen and rejected itself, so the
end-of-turn token is scored on both sides. Templates that support `enable_thinking` (Qwen 3.5) are
rendered with thinking off, as in SFT.

The reference model is loaded explicitly and passed in: left to DPOTrainer under FSDP it is built
with device_map="auto", spread across the node's GPUs, and the first forward fails with a device
mismatch.

WARM_ONLY=1 builds the trainer (datasets cache) and exits, for a one-node warm job before a wide run.
Environment knobs: TRL_DATA_CACHE, DATA_MAP_PROC, ATTN_IMPL, USE_FA2, GC_ON, LOG_STEPS, SAVE_STRATEGY,
CUDNN_SDP.
"""
import argparse
import math
import os
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

from msafetemp.training.hf_utils import sanitise_generation_config


def load_pairs(path, tok):
    cache_dir = os.getenv("TRL_DATA_CACHE", os.path.join(os.getenv("SCRATCH", "/tmp"), ".cache", "trl_dpo_datasets"))
    ds = load_dataset("json", data_files=path, split="train", cache_dir=cache_dir)
    procs = int(os.getenv("DATA_MAP_PROC", "4"))

    def render(row):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": row["user"]}], tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}))
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    return ds.map(render, remove_columns=ds.column_names, num_proc=procs, desc="rendering chat template")


class NonFiniteGradStop(TrainerCallback):
    """Abort on the first non-finite grad_norm instead of training on NaN weights."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        g = (logs or {}).get("grad_norm")
        if g is not None and not math.isfinite(float(g)):
            raise RuntimeError(f"non-finite grad_norm {g} at step {state.global_step}; aborting")


def main():
    # Some PyTorch builds route SDPA to cuDNN attention, whose backward returned NaN on padded
    # chosen/rejected batches. Disable it so SDPA falls back to the memory-efficient kernel.
    if os.getenv("CUDNN_SDP", "0") != "1":
        import torch
        torch.backends.cuda.enable_cudnn_sdp(False)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="safety-SFT checkpoint (policy and reference)")
    ap.add_argument("--data", required=True, help="pairs jsonl from build_pairs.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.2)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_pairs(args.data, tok)
    print(f"[trl_dpo] {len(ds)} pairs; model={args.model}", flush=True)

    attn = os.getenv("ATTN_IMPL") or ("flash_attention_2" if os.getenv("USE_FA2", "0") == "1" else "sdpa")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16", attn_implementation=attn)
    model.config.use_cache = False
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16", attn_implementation=attn)
    ref_model.config.use_cache = False

    cfg = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,  # float == ratio in transformers 5
        weight_decay=0.0,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=os.getenv("GC_ON", "0") == "1",
        bf16=True,
        beta=args.beta,
        loss_type=["sigmoid", "sft"],
        loss_weights=[1.0, args.sft_weight],
        max_length=args.max_len,
        truncation_mode="keep_start",
        logging_steps=int(os.getenv("LOG_STEPS", "10")),
        logging_nan_inf_filter=False,  # the default filter can hide a NaN loss behind a finite log line
        save_strategy=os.getenv("SAVE_STRATEGY", "no"),
        ddp_timeout=7200,
        report_to="none",
        seed=1234,
        dataset_num_proc=4,
    )
    trainer = DPOTrainer(model=model, ref_model=ref_model, args=cfg, train_dataset=ds,
                         processing_class=tok, callbacks=[NonFiniteGradStop()])
    if os.getenv("WARM_ONLY", "0") == "1":
        print("[trl_dpo] WARM_ONLY: dataset caches built, exiting before training", flush=True)
        return
    sanitise_generation_config(model)
    trainer.train()
    sanitise_generation_config(trainer.model)
    trainer.save_model(args.output)
    # Keep every rank alive until rank 0 has written the weights: on a slow shared filesystem ranks
    # that exit early make rank 0 fail on a collective mid-save, leaving only tokenizer files.
    acc = trainer.accelerator
    acc.wait_for_everyone()
    if acc.is_main_process:
        tok.save_pretrained(args.output)
        weights = [f for f in os.listdir(args.output) if f.endswith(".safetensors")]
        if not weights or not os.path.exists(os.path.join(args.output, "config.json")):
            raise RuntimeError(f"save incomplete in {args.output}: {sorted(os.listdir(args.output))}")
    acc.wait_for_everyone()
    print(f"[trl_dpo] DONE -> {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        print("[trl_dpo][rank %s] FATAL" % rank, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
