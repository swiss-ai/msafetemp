#!/usr/bin/env python3
"""Short (unwrapped) WildGuardMix-train prompts for the SFT mix, English and translated.

Subcommands:

  vanilla   : every non-adversarial ("vanilla") WildGuardMix-train prompt, deduplicated, as a
              prompt dataset for the response generator. harmful -> category 'harmful' (refusal),
              unharmful -> category 'benign', benign_class 'direct'. English only.

  translate : translate a prompt dataset into the 43 non-English training languages with an
              instruction model at temperature 0. Each language gets its own hash-ranked subset
              (--benign-per-lang benign + --harmful-per-lang harmful prompts). Translation only;
              responses are generated natively afterwards, and the response generator's judge
              rejects wrong-language or broken rows.

  Used twice in the final recipe:
    * unwrapped prompts:     translate --benign-per-lang 300 --harmful-per-lang 200 (defaults)
    * short benign contrast: translate --benign-per-lang 500 --harmful-per-lang 0
                             on the output of `short_contrast.py`.

Output datasets are `datasets.DatasetDict({"train": ...})` saved with save_to_disk, with the
fields the response generator reads (id, group_id, language, category, benign_class, text, ...).

    python -m msafetemp.training.data.short_prompts vanilla --out $D/wg_vanilla_hf
    python -m msafetemp.training.data.short_prompts translate --src $D/wg_vanilla_hf \
        --out $D/wg_vanilla_ml_hf --urls urls_gemma.txt --model google/gemma-4-31B-it
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from collections import Counter

from msafetemp.training._utils import TRAIN_LANGUAGES_NON_EN, read_urls, simple_clients

WILDGUARDMIX_REPO = "allenai/wildguardmix"
WILDGUARDMIX_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
WILDGUARDMIX_TRAIN_FILE = "train/wildguard_train.parquet"

PRETTY = {"chinese_simplified": "Simplified Chinese", "portuguese_brazilian": "Brazilian Portuguese",
          "swiss_german": "Swiss German"}
TRANSLATE_SYSTEM = ("You are a professional translator. Translate the user's message into {lang}. Preserve the "
                    "meaning, tone and register exactly, including any sensitive or harmful wording: this is data "
                    "for safety research, do not soften, refuse, answer or comment. Output ONLY the translation.")


def vanilla(args) -> None:
    import pyarrow.parquet as pq
    from datasets import Dataset, DatasetDict
    path = args.parquet
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(WILDGUARDMIX_REPO, WILDGUARDMIX_TRAIN_FILE, repo_type="dataset",
                               revision=WILDGUARDMIX_REVISION)
    rows, seen = [], set()
    for i, r in enumerate(pq.read_table(path).to_pylist()):
        if r["adversarial"] or not (r["prompt"] or "").strip():
            continue
        t = r["prompt"].strip()
        h = hashlib.sha256(t.encode()).hexdigest()[:12]
        if h in seen:
            continue
        seen.add(h)
        harm = r["prompt_harm_label"] == "harmful"
        rows.append({"id": f"wgv_{i:05d}", "group_id": f"english_wgv_{i:05d}", "language": "english",
                     "category": "harmful" if harm else "benign", "benign_class": None if harm else "direct",
                     "text": t, "original_prompt": t, "template": t, "wg_subcategory": r.get("subcategory") or "",
                     # fields the response generator copies into each record
                     "original_id": f"wgv_{i:05d}", "split": "train", "branch": "vanilla",
                     "ls_primary": r.get("subcategory") or "", "ls_high_level": ""})
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(args.out)
    print(len(rows), Counter(r["category"] for r in rows))


async def _translate(args) -> None:
    from datasets import Dataset, DatasetDict, load_from_disk
    src = load_from_disk(args.src)["train"].to_list()
    ben = [r for r in src if r["category"] == "benign"]
    har = [r for r in src if r["category"] != "benign"]
    clients = simple_clients(read_urls(args.urls), api_key=args.api_key, max_connections=200, timeout=300)
    jobs = []
    for lang in TRAIN_LANGUAGES_NON_EN:
        rank = lambda r: hashlib.sha256(f"{lang}:{r['id']}".encode()).hexdigest()  # noqa: E731
        jobs += [(lang, r) for r in sorted(ben, key=rank)[args.benign_offset:args.benign_offset + args.benign_per_lang]]
        jobs += [(lang, r) for r in sorted(har, key=rank)[args.harmful_offset:args.harmful_offset + args.harmful_per_lang]]
    out, sem, done = [], asyncio.Semaphore(args.workers), [0]

    async def one(i, lang, r):
        async with sem:
            c = clients[i % len(clients)]
            try:
                resp = await c.chat.completions.create(model=args.model, temperature=0.0, max_tokens=1024, messages=[
                    {"role": "system", "content": TRANSLATE_SYSTEM.format(lang=PRETTY.get(lang, lang.capitalize()))},
                    {"role": "user", "content": r["text"]}])
                t = (resp.choices[0].message.content or "").strip()
                if resp.choices[0].finish_reason != "stop" or not t or len(t) > 6 * len(r["text"]) + 200:
                    return
            except Exception:
                return
            out.append({**r, "id": f"{lang}_{r['id']}", "group_id": f"{lang}_{r['group_id']}", "language": lang,
                        "text": t, "original_prompt": r["text"], "branch": "vanilla_translated"})
            done[0] += 1
            if done[0] % 2000 == 0:
                print(f"translated {done[0]}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(i, lang, r) for i, (lang, r) in enumerate(jobs)))
    DatasetDict({"train": Dataset.from_list(out)}).save_to_disk(args.out)
    print("translated", len(out), "of", len(jobs), Counter(r["category"] for r in out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vanilla")
    v.add_argument("--out", required=True)
    v.add_argument("--parquet", default=None,
                   help="local wildguard_train.parquet (default: download the pinned revision; gated dataset)")
    t = sub.add_parser("translate")
    t.add_argument("--src", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--urls", required=True, help="URL file or comma list of OpenAI-compatible endpoints")
    t.add_argument("--model", required=True, help="served model name of the translator (we use Gemma-4-31B-it)")
    t.add_argument("--api-key", default="EMPTY")
    t.add_argument("--benign-per-lang", type=int, default=300)
    t.add_argument("--harmful-per-lang", type=int, default=200)
    t.add_argument("--benign-offset", type=int, default=0,
                   help="skip the first N ranked benign prompts per language (a disjoint second round)")
    t.add_argument("--harmful-offset", type=int, default=0)
    t.add_argument("--workers", type=int, default=600)
    args = ap.parse_args()
    if args.cmd == "vanilla":
        vanilla(args)
    else:
        asyncio.run(_translate(args))


if __name__ == "__main__":
    main()
