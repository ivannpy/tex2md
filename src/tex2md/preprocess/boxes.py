"""tcolorbox environments: pandoc cannot expand their definitions."""
from __future__ import annotations

import re

from tex2md.util.tex import group, skip_ws


def tcolorbox_to_quote(text: str) -> str:
    """Code listings become verbatim; ordinary boxes become a quote with the title (if any) in bold."""
    for m in reversed(list(re.finditer(r"\\(newtcblisting|newtcolorbox|DeclareTColorBox|NewTColorBox|NewTCBListing)\s*\{([^}]*)\}", text))):
        i = m.end()
        specs = []
        while True:
            i = skip_ws(text, i)
            if i < len(text) and text[i] in "[{":
                g, i = group(text, i, text[i], "]" if text[i] == "[" else "}")
                specs.append(g)
            else:
                break
        name = re.escape(m.group(2))
        is_listing = "listing" in m.group(1).lower()
        text = text[:m.start()] + text[i:]
        if is_listing:
            text = re.sub(r"\\begin\{" + name + r"\}(\[[^\]]*\]|\{[^}]*\})*", r"\\begin{verbatim}", text)
            text = re.sub(r"\\end\{" + name + r"\}", r"\\end{verbatim}", text)
            continue
        nargs = int(specs[0]) if specs and specs[0].isdigit() else 0

        def box(mm, nargs=nargs):
            j, titles = mm.end(), []
            for k in range(nargs):
                j2 = skip_ws(mm.string, j)
                if j2 < len(mm.string) and mm.string[j2] == "[":
                    _, j = group(mm.string, j2, "[", "]")
                elif j2 < len(mm.string) and mm.string[j2] == "{":
                    g, j = group(mm.string, j2, "{", "}")
                    titles.append(g)
                else:
                    break
            return "\\begin{quote}" + ("\\textbf{" + titles[0] + "}\n\n" if titles and titles[0].strip() else ""), j
        out, pos = [], 0
        for mm in re.finditer(r"\\begin\{" + name + r"\}", text):
            if mm.start() < pos:
                continue
            rep, j = box(mm)
            out.append(text[pos:mm.start()] + rep)
            pos = j
        out.append(text[pos:])
        text = "".join(out)
        text = re.sub(r"\\end\{" + name + r"\}", r"\\end{quote}", text)
    return text
