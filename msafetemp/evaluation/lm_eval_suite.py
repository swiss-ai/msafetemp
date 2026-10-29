"""Thin wrapper around upstream lm-evaluation-harness for the external safety, over-refusal and
capability benchmarks.

Safety:        aya_redteaming (AyaRT, judge Llama-3.3-70B-Instruct),
               polyglotoxicitypromptsllama_small (PolyTox, classifier Llama-Guard-4-12B)
Over-refusal:  orbench (OR-Bench-hard-1k, refusal keyterms)
Capability:    mmlu_flan_cot_zeroshot, gsm8k_cot, ifeval, bbh

The three benchmark-specific tasks are shipped in msafetemp/evaluation/lm_eval_tasks and are
added with --include_path; all other tasks are upstream. Task list:
configs/evaluation/lm_eval_tasks.txt.

Default settings: vLLM backend, bf16, chat template applied (--apply_chat_template
--fewshot_as_multiturn), greedy decoding, max_gen_toks 2048 unless the task sets its own,
batch_size auto:20. Judges are served separately (scripts/slurm/eval_vllm_server.sbatch) and
passed through AYA_JUDGE_API_URL / LLAMA_GUARD_API_URL.

  python -m msafetemp.evaluation.lm_eval_suite --model_path swiss-ai/Apertus-8B-Instruct-2509 \
      --output_dir $MSAFETEMP_OUT_DIR/lm_eval/Apertus-8B-Instruct-2509 --tp 4 [--dry_run]
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE / "lm_eval_tasks"
REPO = HERE.parents[1]
DEFAULT_TASKS = REPO / "configs" / "evaluation" / "lm_eval_tasks.txt"


def read_tasks(path):
    tasks = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tasks.append(line)
    return tasks


def build_command(a):
    tasks = a.tasks.split(",") if a.tasks else read_tasks(a.task_file)
    model_args = [f"pretrained={a.model_path}", "dtype=bfloat16", f"tensor_parallel_size={a.tp}",
                  f"data_parallel_size={a.dp}", f"gpu_memory_utilization={a.gpu_memory_utilization}"]
    if a.enable_thinking is not None:
        model_args.append(f"enable_thinking={a.enable_thinking}")
    if a.max_model_len:
        model_args.append(f"max_model_len={a.max_model_len}")
    cmd = ["lm_eval", "--model", "vllm", "--model_args", ",".join(model_args),
           "--tasks", ",".join(tasks), "--include_path", str(TASK_DIR),
           "--batch_size", a.batch_size, "--max_batch_size", "32",
           "--output_path", a.output_dir, "--log_samples", "--trust_remote_code",
           "--gen_kwargs", f"max_gen_toks={a.max_gen_toks}"]
    if not a.no_chat_template:
        cmd += ["--apply_chat_template", "--fewshot_as_multiturn"]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True, help="HF id or local checkpoint")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--task_file", default=str(DEFAULT_TASKS))
    ap.add_argument("--tasks", default=None, help="comma-separated tasks (overrides --task_file)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--max_gen_toks", type=int, default=2048)
    ap.add_argument("--batch_size", default="auto:20")
    ap.add_argument("--enable_thinking", choices=["True", "False"], default=None,
                    help="forwarded to the chat template when set")
    ap.add_argument("--no_chat_template", action="store_true",
                    help="base checkpoints without a chat template")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="print the command only")
    a = ap.parse_args(argv)

    tasks = a.tasks or ",".join(read_tasks(a.task_file))
    if "aya_redteaming" in tasks and not os.environ.get("AYA_JUDGE_API_URL"):
        print("WARNING: AYA_JUDGE_API_URL is not set; aya_redteaming will fail at aggregation.")
    if "polyglotoxicityprompts" in tasks and not os.environ.get("LLAMA_GUARD_API_URL"):
        print("WARNING: LLAMA_GUARD_API_URL is not set; PolyTox will fail at aggregation.")
    cmd = build_command(a)
    print(" ".join(shlex.quote(c) for c in cmd))
    if not a.dry_run:
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
