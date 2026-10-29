#!/usr/bin/env python
"""Safety SFT with TRL SFTTrainer (Llama 3.1, Olmo 3, Qwen 3.5; any HF causal LM with a chat template).

Recipe (TRL path): the 400,000-row TRL sample of the SFT mix (build_sft_mix.py), one epoch,
no packing, lr 1e-5 cosine, 3% warmup, weight decay 0, grad clip 1.0, bf16, seed 1234, maximum length
4096, loss on assistant tokens only, per-device batch 1 on 128 GPUs with no gradient accumulation
(effective batch 128), FSDP2 via `accelerate launch` (configs/accelerate/).

Input rows: {messages: [user, assistant]} (or {prompt, response}). The chat template is rendered
here and TRL receives plain prompt/completion strings (completion = answer + eos_token), so the
end-of-turn token is trained. With message lists TRL tokenizes prompt and prompt+completion
separately and checks that one is a prefix of the other; for Qwen 3.5 and Apertus the generation
prompt is not a clean prefix, which floods the log and drops TRL onto a very slow path.
Pre-rendering removes the ambiguity for every family.

WARM_ONLY=1 builds the trainer (every datasets map lands in the fingerprinted cache) and exits, so a
wide run can depend on a one-node warm job instead of racing all ranks to build the same cache on
a shared filesystem.

Environment knobs: TRL_DATA_CACHE (datasets cache dir), DATA_MAP_PROC, DATASET_NUM_PROC, USE_FA2,
GC_ON (gradient checkpointing inside the model; FSDP activation checkpointing is set in the
accelerate config instead), SAVE_STRATEGY, PRERENDER_CHAT, WARM_ONLY.
"""
import argparse
import os
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from msafetemp.training.hf_utils import sanitise_generation_config


def _pair(row):
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) == 2 \
            and messages[0].get("role") == "user" and messages[1].get("role") == "assistant":
        p, r = messages[0].get("content"), messages[1].get("content")
    else:
        p, r = row.get("prompt"), row.get("response")
    return {"prompt": [{"role": "user", "content": p or ""}],
            "completion": [{"role": "assistant", "content": r or ""}]}


def load_pairs(path, tok=None, max_samples=None, prerender=True):
    """Memory-mapped load: `load_dataset("json")` keeps an Arrow cache on disk shared by every rank
    (building the dataset in Python RAM per rank exceeded node memory at ~400k rows)."""
    cache_dir = os.getenv("TRL_DATA_CACHE", os.path.join(os.getenv("SCRATCH", "/tmp"), ".cache", "trl_sft_datasets"))
    ds = load_dataset("json", data_files=path, split="train", cache_dir=cache_dir)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    procs = int(os.getenv("DATA_MAP_PROC", "4"))

    if prerender and tok is not None:
        eos = tok.eos_token or ""

        def render(row):
            pair = _pair(row)
            user = pair["prompt"][0]["content"]
            assistant = pair["completion"][0]["content"]
            try:
                prompt_text = tok.apply_chat_template([{"role": "user", "content": user}],
                                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:  # template takes no enable_thinking kwarg
                prompt_text = tok.apply_chat_template([{"role": "user", "content": user}],
                                                      tokenize=False, add_generation_prompt=True)
            return {"prompt": prompt_text, "completion": assistant + eos}

        ds = ds.map(render, remove_columns=ds.column_names, num_proc=procs, desc="rendering chat template")
        return ds.filter(lambda row: bool(row["prompt"]) and bool(row["completion"]),
                         num_proc=procs, desc="dropping empty rows")

    ds = ds.map(_pair, remove_columns=ds.column_names, num_proc=procs, desc="building prompt/completion pairs")
    return ds.filter(lambda row: bool(row["prompt"][0]["content"]) and bool(row["completion"][0]["content"]),
                     num_proc=procs, desc="dropping empty rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template-model", default="",
                    help="borrow chat_template from here if --model has none (base checkpoints)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-device-bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.chat_template is None:
        src = args.template_model or args.model
        ttok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        if ttok.chat_template is None:
            raise SystemExit(f"no chat_template on {args.model} or template-model {src}")
        tok.chat_template = ttok.chat_template
        print(f"[trl_sft] borrowed chat_template from {src}", flush=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_pairs(args.data, tok=tok, max_samples=args.max_samples or None,
                    prerender=os.getenv("PRERENDER_CHAT", "1") == "1")
    print(f"[trl_sft] {len(ds)} training pairs; model={args.model}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="bfloat16", trust_remote_code=True,
        attn_implementation="flash_attention_2" if os.getenv("USE_FA2", "0") == "1" else "sdpa")
    model.config.use_cache = False

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,   # transformers 5: a float warmup_steps is a ratio
        weight_decay=0.0,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=os.getenv("GC_ON", "0") == "1",
        bf16=True,
        max_length=args.max_len,
        completion_only_loss=True,     # loss on assistant tokens only
        packing=False,
        logging_steps=10,
        save_strategy=os.getenv("SAVE_STRATEGY", "no"),
        ddp_timeout=7200,             # long NCCL timeout so the FSDP save-gather does not trip the watchdog
        report_to="none",
        seed=1234,
        dataset_num_proc=int(os.getenv("DATASET_NUM_PROC", "2")),
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    if os.getenv("WARM_ONLY", "0") == "1":
        print("[trl_sft] WARM_ONLY: dataset caches built, exiting before training", flush=True)
        return
    sanitise_generation_config(model)
    trainer.train()
    sanitise_generation_config(trainer.model)
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"[trl_sft] DONE -> {args.output}", flush=True)


if __name__ == "__main__":
    # torch elastic often reports `error_file: <N/A>`; print the child traceback explicitly
    try:
        main()
    except BaseException:
        import traceback
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        print("[trl_sft][rank %s] FATAL" % rank, file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
