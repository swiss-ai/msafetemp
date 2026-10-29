"""Benign audit, step 2 of 3: the final selection rule.

The rule was fixed after two independent blind annotators checked 80 flagged rows.
Per-criterion precision on that review:

    leftover_technique_annotation   85%   the attacker's own construction notes remain
    embeds_harmful_example          35%   a worked harmful specimen in the prompt text
    harmful_as_written + severity   60%   minors / consent-absent settings
    (harmful_as_written alone        9%   dropped: grades assistant policy, not the swap)
    (bypasses_consent               20%   dropped for the same reason)

A row is selected if leftover_technique_annotation or embeds_harmful_example fires, or if
harmful_as_written fires together with a severity marker found in the judge's own English
output. Overall ~60% precision at ~10% of rows, keeping 30 of the 32 confirmed-bad rows.

    python -m msafetemp.augmentation.audit_select \\
        --audit $D/wildguard/audit/audit_full.jsonl --out_dir $D/wildguard/audit

Writes reclassify.json ({row_id: reason}) and reclassify_summary.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re

from msafetemp.common.io import read_jsonl

SEV = re.compile(r"minor|child|children|teen|adolescen|underage|under.?18|pupil|school|"
                 r"non.?consen|without .{0,20}consent|coerc|forc(ed|ing)|incest|"
                 r"step.?(father|daughter)|rape|grooming|explicit sexual|sexually explicit", re.I)


def severity_marked(j):
    t = " ".join([j.get("deliverable") or "",
                  (j.get("quotes") or {}).get("harmful_as_written") or "",
                  j.get("reasoning") or ""])
    return bool(SEV.search(t))


def selected(r):
    j = r.get("judge") or {}
    if j.get("leftover_technique_annotation") or j.get("embeds_harmful_example"):
        return True
    return bool(j.get("harmful_as_written") and severity_marked(j))


def reason(r):
    j = r["judge"]
    if j.get("leftover_technique_annotation"):
        return "leftover_technique_annotation"
    if j.get("embeds_harmful_example"):
        return "embeds_harmful_example"
    return "harmful_as_written_severe"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit", required=True, help="audit_benign.py output JSONL")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    rows = list(read_jsonl(args.audit, skip_bad=True))
    ok = [r for r in rows if r.get("judge")]
    sel = {r["id"]: reason(r) for r in ok if selected(r)}
    by_reason = collections.Counter(sel.values())
    by_lang = collections.Counter(r["language"] for r in ok if r["id"] in sel)
    lang_tot = collections.Counter(r["language"] for r in ok)
    summary = {
        "rows_judged": len(ok),
        "rows_unjudged": len(rows) - len(ok),
        "selected": len(sel),
        "selected_pct": round(100.0 * len(sel) / max(len(ok), 1), 2),
        "by_reason": dict(by_reason),
        "language_pct": {k: round(100.0 * by_lang[k] / lang_tot[k], 1)
                         for k in sorted(lang_tot) if lang_tot[k] >= 500},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "reclassify.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=0)
    with open(os.path.join(args.out_dir, "reclassify_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "language_pct"}, indent=2))
    lp = summary["language_pct"]
    if lp:
        print("language spread: min %.1f%%  median %.1f%%  max %.1f%%" % (
            min(lp.values()), sorted(lp.values())[len(lp) // 2], max(lp.values())))


if __name__ == "__main__":
    main()
