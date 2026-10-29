"""Benign audit, step 3 of 3: move audit-selected benign rows to their own category.

NOTHING IS DELETED. Selected rows move from category "benign" to
"benign_audit_rejected" and carry an `audit_reason`, so the change is inspectable and
reversible. Training / evaluation code that selects `category == "benign"` picks up the
cleaned set automatically. In the grouped splits, rejected entries move from `benign` to
a `benign_audit_rejected` list. The source dataset is never modified.

    python -m msafetemp.augmentation.audit_reclassify \\
        --in $D/wildguard/hf --out $D/wildguard/hf_audited \\
        --reclassify $D/wildguard/audit/reclassify.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

NEW_CATEGORY = "benign_audit_rejected"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", required=True)
    p.add_argument("--reclassify", required=True, help="JSON {row_id: reason} from audit_select.py")
    p.add_argument("--num_proc", type=int, default=16)
    a = p.parse_args()

    from datasets import DatasetDict, load_from_disk

    sel = json.load(open(a.reclassify))
    print(f"[reclassify] {len(sel)} rows selected", flush=True)
    if os.path.exists(a.dst):
        sys.exit(f"[reclassify] {a.dst} exists; refusing to overwrite")

    d = load_from_disk(a.src)
    print("[reclassify] loaded:", {k: len(v) for k, v in d.items()}, flush=True)

    def flat(batch):
        cats, reasons = [], []
        for rid, cat in zip(batch["id"], batch["category"]):
            r = sel.get(rid)
            cats.append(NEW_CATEGORY if r else cat)
            reasons.append(r or "")
        batch["category"] = cats
        batch["audit_reason"] = reasons
        return batch

    # Grouped splits nest the variants; the flat id is `{group_id}_benign_{variant_index}`,
    # exactly how the flat split builds it.
    def grouped(batch):
        keep_all, rej_all, nkeep, nrej = [], [], [], []
        for gid, bens in zip(batch["group_id"], batch["benign"]):
            keep, rej = [], []
            for e in bens:
                r = sel.get(f"{gid}_benign_{e['variant_index']}")
                if r:
                    rej.append(dict(e, audit_reason=r))
                else:
                    keep.append(dict(e, audit_reason=""))
            keep_all.append(keep)
            rej_all.append(rej)
            nkeep.append(len(keep))
            nrej.append(len(rej))
        batch["benign"] = keep_all
        batch[NEW_CATEGORY] = rej_all
        batch["num_benign"] = nkeep
        batch["num_" + NEW_CATEGORY] = nrej
        return batch

    out = {}
    for name, split in d.items():
        fn = grouped if name.startswith("grouped") else flat
        out[name] = split.map(fn, batched=True, batch_size=1000, num_proc=a.num_proc,
                              desc=f"reclassify {name}")
    DatasetDict(out).save_to_disk(a.dst, max_shard_size="500MB")
    print(f"[reclassify] wrote {a.dst}", flush=True)

    for name in ("train", "test"):
        c = collections.Counter(out[name]["category"])
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())), flush=True)
    moved = sum(collections.Counter(out[n]["category"])[NEW_CATEGORY] for n in ("train", "test"))
    print(f"[reclassify] moved {moved} of {len(sel)} selected rows", flush=True)
    if moved != len(sel):
        print("[reclassify] MISMATCH: some selected ids were not found in the flat splits", flush=True)


if __name__ == "__main__":
    main()
