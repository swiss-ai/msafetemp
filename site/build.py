#!/usr/bin/env python
"""Build the MSafeTemp project page.

Inlines data/site-data.json.gz, src/style.css and src/app.js into src/page.html and writes
dist/index.html, a single self-contained file. Standard library only.
"""
import gzip
import json
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DIST = ROOT / "dist"


def main():
    data = json.loads(gzip.decompress((ROOT / "data" / "site-data.json.gz").read_bytes()).decode("utf-8"))
    js_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    js_data = js_data.replace("</", "<\\/").replace("<!--", "<\\!--")
    page = (SRC / "page.html").read_text(encoding="utf-8")
    css = (SRC / "style.css").read_text(encoding="utf-8")
    js = (SRC / "app.js").read_text(encoding="utf-8")
    # Chinese and Korean appear only in a few slot-vocabulary values: load just those glyphs.
    cjk = sorted({ch for b in data.get("behaviors", []) for sl in b["vocab"]["slots"] for e in sl.get("examples", [])
                  if re.match(r"chinese|korean", e.get("lang") or "") for ch in (e.get("value") or "") if ord(ch) > 0x2E80})
    link = ""
    if cjk:
        link = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC&family=Noto+Sans+KR&text='
                + quote("".join(cjk)) + '&display=swap">')
    page = page.replace("<!--__CJK_FONTS__-->", link)
    page = page.replace("/*__CSS__*/", css).replace("/*__JS__*/", js).replace("__SITE_DATA__", js_data)
    assert "__SITE_DATA__" not in page and "/*__JS__*/" not in page

    # head = everything before the first <div class="page">, body = the rest
    cut = page.index('<div class="page">')
    doc = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
           + page[:cut] + "</head>\n<body>\n" + page[cut:] + "\n</body>\n</html>\n")
    DIST.mkdir(exist_ok=True)
    out = DIST / "index.html"
    out.write_text(doc, encoding="utf-8")
    print(f"{out}  {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
