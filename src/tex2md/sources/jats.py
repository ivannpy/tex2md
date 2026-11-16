"""Publisher / PubMed Central JATS XML -> Markdown body, converted with pandoc."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

from tex2md.config import FILTER, UNWRAP_FILTER
from tex2md.util.io import http_get

MD_FORMAT = ("markdown-raw_attribute-header_attributes-fenced_divs-bracketed_spans-native_divs"
             "-native_spans-link_attributes-simple_tables-multiline_tables-grid_tables-smart-citations")


def jats_body(src):
    raw = http_get(src).decode("utf-8") if re.match(r"https?://", src) else open(src, encoding="utf-8").read()
    m = re.search(r"<article\b.*</article>", raw, re.S)
    if not m:
        raise SystemExit("no <article> element in " + src)
    xml = m.group(0)

    def tex(mm):
        inner = mm.group(2)
        b = re.search(r"\\begin\{document\}(.*)\\end\{document\}", inner, re.S)
        inner = (b.group(1) if b else inner).strip()
        inner = re.sub(r"^\$\$?|\$\$?$", "", inner).strip()
        return mm.group(1) + inner + mm.group(3)
    # PMC wraps every formula in a full LaTeX document
    xml = re.sub(r"(<tex-math[^>]*>)(.*?)(</tex-math>)", tex, xml, flags=re.S)
    xml = re.sub(r"<!\[CDATA\[|\]\]>", "", xml)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "article.xml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(xml)
        p = subprocess.run(["pandoc", path, "-f", "jats", "-t", MD_FORMAT, "--citeproc", "--wrap=none",
                            "--lua-filter", FILTER, "--lua-filter", UNWRAP_FILTER],
                           capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise SystemExit("pandoc failed: " + p.stderr[-500:])
    body = p.stdout
    body = re.sub(r"\(\(([^()]*\d{4}[a-z]?(?:;[^()]*\d{4}[a-z]?)*)\)\)", r"(\1)", body)  # ((Author 2021))
    n = [0]

    def table_title(mm):
        n[0] += 1
        return f"**Tabla {n[0]}.** {mm.group(1)}"
    return re.sub(r"(?m)^###### (.+)$", table_title, body)
