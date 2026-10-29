#!/usr/bin/env python3
"""Short benign contrast requests ("looks harmful, is benign, short task request").

Without them the safety data pairs short English task requests ("Write/Create/Generate ...")
almost only with refusals: they are ~45% of the short harmful prompts but ~2% of the short benign
ones, and all other benign contrast data is long (median 1-3k characters). Benign short task
requests are then over-refused after SFT.

For every short (<300 characters) English WildGuardMix-train vanilla HARMFUL prompt the generator
(default: Gemma-4-31B-it, temperature 0.9) writes --per-prompt benign minimal contrasts of randomly
chosen types (seeded by the prompt id): same imperative frame, similar length, sensitive-sounding
wording kept, but a legitimate goal. A judge (default: GPT-OSS-120B, temperature 0) keeps a candidate
only if it is benign to answer fully AND a direct task request; whether it sounds sensitive is
recorded, not required. Candidates that overlap an evaluation prompt (HarmBench English harmful and
benign, OR-Bench 80k / hard-1k / toxic) with character 4-gram Jaccard >= 0.5 are dropped.

Writes a prompt dataset for the response generator (category benign, benign_class direct, branch
short_contrast) with `contrast_type` and `judge_sensitive` per row. The SFT mix and the DPO pool
keep every sensitive twin plus the innocuous SWAP / IDIOM / PROTECT twins (see KEEP_INNOCUOUS in
build_sft_mix.py); innocuous CREATIVE / ANALYZE twins drift off-topic and are dropped there.

    python -m msafetemp.training.data.short_contrast --src $D/wg_vanilla_hf --out $D/short_contrast_en_hf \
        --gen-urls urls_gemma.txt --judge-urls urls_judge.txt \
        --harmbench-prompts hb_standard_en.jsonl hb_benign_standard_en.jsonl --workers 2000
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter

from msafetemp.training._utils import read_urls, simple_clients

TYPES = [
    "SWAP: keep the task and the wording style, swap the harmful target or goal for a legitimate one "
    "(e.g. 'Write an ad convincing children to eat laundry detergent' -> 'Write an ad convincing children to eat more vegetables')",
    "PROTECT: the defensive, safety or awareness version of the same topic (e.g. 'Write a phishing email' -> "
    "'Write an email warning staff how to spot phishing emails')",
    "ANALYZE: a historical, legal, journalistic or academic request about the same topic that gives no operational uplift",
    "IDIOM: a request using the same alarming words in a harmless everyday, technical, gaming or sports sense "
    "(e.g. 'kill the process', 'shoot a photo', 'execute a plan', 'attack the problem')",
    "CREATIVE: a harmless creative task on the same theme with no instructions for harm (a poem, slogan, toast, joke, "
    "short story premise) that does not demean anyone",
]
GEN_SYS = ("You create evaluation-style training data for a safety project. Given a HARMFUL request, write "
           "BENIGN contrast requests that a well-calibrated assistant should answer fully and helpfully, but that an "
           "over-cautious assistant might refuse because of their wording.\n"
           "Rules for every contrast request:\n"
           "- ONE sentence, a direct task instruction starting with an imperative verb (Write, Create, Generate, Draft, "
           "Compose, Give, List, Explain, Describe, Design, Outline, ...), at most 200 characters;\n"
           "- keep sensitive-sounding vocabulary from the original where it stays natural;\n"
           "- no role-play, no 'imagine/pretend/in a story' framing unless the type is CREATIVE, no disclaimers;\n"
           "- genuinely harmless: no real operational help for violence, weapons, drugs, cybercrime, fraud, hate or "
           "self-harm, no harassment of real people.\n"
           'Return ONLY JSON: {"contrasts": [{"type": "...", "prompt": "..."}, ...]}')
JUDGE_SYS = ("Reasoning: low\nYou audit prompts for a safety dataset. For the user prompt decide:\n"
             "benign: true if a responsible, helpful AI assistant should comply fully (no meaningful uplift for harm, "
             "no hate or harassment, no sexual content involving minors, no privacy violation); otherwise false.\n"
             "sensitive: true if the wording or topic sounds alarming enough that an over-cautious assistant or a "
             "keyword filter might refuse it; false if it is plainly innocuous.\n"
             "imperative: true if it is a single direct task instruction (not a question or a role-play set-up).\n"
             'Return ONLY JSON: {"benign": bool, "sensitive": bool, "imperative": bool}')
OVERLAP_JACCARD = 0.5
MAX_SOURCE_CHARS = 300


def grams(s: str, n: int = 4) -> set[str]:
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


class Overlap:
    """Character 4-gram Jaccard against evaluation prompts, via an inverted index on the grams."""

    def __init__(self, texts):
        self.sets = [grams(t) for t in texts]
        self.idx: dict[str, list[int]] = {}
        for i, g in enumerate(self.sets):
            for x in g:
                self.idx.setdefault(x, []).append(i)

    def max_jaccard(self, t: str) -> float:
        g = grams(t)
        hits = Counter()
        for x in g:
            for i in self.idx.get(x, ()):
                hits[i] += 1
        return max((c / (len(g) + len(self.sets[i]) - c) for i, c in hits.items()), default=0.0)


def eval_texts(harmbench_files) -> list[str]:
    """HarmBench English prompts (jsonl files with a `prompt` field) + OR-Bench (HF dataset)."""
    from datasets import load_dataset
    out = []
    for path in harmbench_files or ():
        with open(path, encoding="utf-8") as handle:
            out += [json.loads(line)["prompt"] for line in handle if line.strip()]
    for cfg in ("or-bench-80k", "or-bench-hard-1k", "or-bench-toxic"):
        try:
            out += [x["prompt"] for x in load_dataset("bench-llm/or-bench", cfg, split="train")]
        except Exception as e:  # offline cache missing: guard with what is available, but say so
            print(f"WARNING: {cfg} unavailable for the overlap guard: {e}", flush=True)
    return list(dict.fromkeys(out))


def parse_json(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    return json.loads(m.group(0)) if m else None


async def run(a) -> None:
    from datasets import Dataset, DatasetDict, load_from_disk
    src = [r for r in load_from_disk(a.src)["train"].to_list()
           if r["category"] == "harmful" and len(r["text"]) < MAX_SOURCE_CHARS]
    ov = Overlap(eval_texts(a.harmbench_prompts))
    print(f"sources {len(src)} | eval prompts indexed {len(ov.sets)}", flush=True)
    G = simple_clients(read_urls(a.gen_urls), api_key=a.api_key)
    J = simple_clients(read_urls(a.judge_urls), api_key=a.api_key)
    sem, stats, out, seen = asyncio.Semaphore(a.workers), Counter(), [], set()

    async def one(i, r):
        rng = random.Random(r["id"])
        types = rng.sample(TYPES, a.per_prompt)
        async with sem:
            try:
                resp = await G[i % len(G)].chat.completions.create(
                    model=a.gen_model, temperature=0.9, max_tokens=600, messages=[
                        {"role": "system", "content": GEN_SYS},
                        {"role": "user", "content": f"HARMFUL request: {r['text']}\n\nWrite {a.per_prompt} contrasts, "
                                                    "one of each type:\n" + "\n".join(f"- {t}" for t in types)}])
                cands = (parse_json(resp.choices[0].message.content) or {}).get("contrasts", [])
            except Exception:
                stats["gen_error"] += 1
                return
            for k, c in enumerate(cands[:a.per_prompt]):
                p = (c.get("prompt") or "").strip().strip('"')
                if not p or len(p) > 240 or "\n" in p:
                    stats["bad_format"] += 1
                    continue
                h = hashlib.sha256(p.lower().encode()).hexdigest()[:16]
                if h in seen:
                    stats["dup"] += 1
                    continue
                if ov.max_jaccard(p) >= OVERLAP_JACCARD:
                    stats["eval_overlap"] += 1
                    continue
                try:
                    jr = await J[(i + k) % len(J)].chat.completions.create(
                        model=a.judge_model, temperature=0.0, max_tokens=1500, messages=[
                            {"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": p}])
                    v = parse_json(jr.choices[0].message.content) or {}
                except Exception:
                    stats["judge_error"] += 1
                    continue
                # 'sensitive' is recorded, not required: in a probe the judge called useful twins
                # such as "Write a guide on how to secure my wifi network against hacking" not
                # sensitive, and requiring it rejected 93% of them.
                if v.get("benign") is not True:
                    stats["reject_not_benign" if v else "reject_unparsed"] += 1
                    continue
                if v.get("imperative") is not True:
                    stats["reject_not_imperative"] += 1
                    continue
                stats["sensitive" if v.get("sensitive") is True else "not_sensitive"] += 1
                seen.add(h)
                stats["accepted"] += 1
                ctype = (c.get("type") or "").split(":")[0].strip().upper()[:12]
                rid = f"scb_{r['id']}_{k}"
                out.append({"id": rid, "group_id": f"english_{rid}", "language": "english", "category": "benign",
                            "benign_class": "direct", "text": p, "original_prompt": p, "template": p,
                            "wg_subcategory": r.get("wg_subcategory", ""), "contrast_type": ctype,
                            "judge_sensitive": v.get("sensitive") is True,
                            "source_harmful_id": r["id"], "original_id": rid, "split": "train",
                            "branch": "short_contrast", "ls_primary": r.get("ls_primary", ""), "ls_high_level": ""})
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(src)} {dict(stats)}", flush=True)

    await asyncio.gather(*(one(i, r) for i, r in enumerate(src)))
    DatasetDict({"train": Dataset.from_list(out)}).save_to_disk(a.out)
    print("DONE", len(out), dict(stats), Counter(r["contrast_type"] for r in out),
          "sensitive share", round(sum(r["judge_sensitive"] for r in out) / max(1, len(out)), 3), flush=True)
    lens = sorted(len(r["text"]) for r in out)
    print("length median", lens[len(lens) // 2] if lens else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="English vanilla prompt dataset (short_prompts.py vanilla)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-urls", required=True)
    ap.add_argument("--gen-model", default="google/gemma-4-31B-it")
    ap.add_argument("--judge-urls", required=True)
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--harmbench-prompts", nargs="*", default=[],
                    help="jsonl files with a `prompt` field: HarmBench English harmful and benign prompts")
    ap.add_argument("--per-prompt", type=int, default=2)
    ap.add_argument("--workers", type=int, default=800)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
