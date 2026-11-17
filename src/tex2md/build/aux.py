"""Numbers of the compiled document from the .aux files (labels and floats)."""
from __future__ import annotations

import os
import re

from tex2md.util.io import read
from tex2md.util.tex import clean_num, group, skip_ws

# A caption written ``\caption{\small{...}}`` reaches the .aux with the size command expanded (\small is fragile in
# some styles): "\fontsize {9}{10}\selectfont \abovedisplayskip 6\p@ plus1.5\p@ ...{the caption}". Without this the
# text of the caption would start with digits and "plus"/"minus" and match no caption of the document.
_GLUE = r"(?:-?[0-9.]+|\\z@)(?:\\p@)?(?:\s*plus\s*-?[0-9.]+(?:\\p@)?)?(?:\s*minus\s*-?[0-9.]+(?:\\p@)?)?"
_SKIPS = (r"(?:\\(?:abovedisplayskip|abovedisplayshortskip|belowdisplayskip|belowdisplayshortskip)\s*"
          r"(?:" + _GLUE + r"|\\(?:above|below)display(?:short)?skip)?\s*)*")
_FONT_NOISE = re.compile(r"\\fontsize\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\\selectfont\s*" + _SKIPS
                         + r"|\\@setfontsize\s*\\[a-zA-Z@]+\s*\\@[a-z]+pt\s*\{[^{}]*\}\s*" + _SKIPS)


def _without_font_noise(text):
    return _FONT_NOISE.sub(" ", text)


def parse_floats(build):
    """Ordered [{kind, num, text}] for every \\caption of the compiled document, from the
    \\@writefile{lof|lot|loa} lines in the .aux files (the numbers the PDF really shows).
    kind is "figure", "table" or "algorithm"."""
    auxes = {}
    for dp, _, fns in os.walk(build):
        for f in fns:
            if f.endswith(".aux"):
                auxes[f] = read(os.path.join(dp, f))
    inputs = {m for t in auxes.values() for m in re.findall(r"\\@input\{([^}]*)\}", t)}
    roots = [f for f in auxes if f not in inputs and f not in {os.path.basename(i) for i in inputs}]
    if not roots:
        return []
    root = max(roots, key=lambda f: len(auxes[f]))
    out, seen = [], set()

    def walk(name):
        if name in seen or name not in auxes:
            return
        seen.add(name)
        t = auxes[name]
        for m in re.finditer(r"\\@input\{([^}]*)\}|\\@writefile\{lo[fta]\}\{\\contentsline \{(figure|table|algorithm)\}\{\\numberline \{([^}]*)\}(.*)", t):
            if m.group(1):
                walk(os.path.basename(m.group(1)))
            else:
                txt = re.sub(r"\\[a-zA-Z@]+\s*", " ", re.sub(r"^\\ignorespaces\s*", "", _without_font_noise(m.group(4))))
                out.append({"kind": m.group(2), "num": clean_num(m.group(3)),
                            "text": re.sub(r"[^a-z0-9]", "", txt.lower())[:40]})
    walk(root)
    return out


def parse_aux(build):
    labels, crefs = {}, {}
    for dp, _, fns in os.walk(build):
        for f in fns:
            if not f.endswith(".aux"):
                continue
            t = read(os.path.join(dp, f))
            for m in re.finditer(r"\\newlabel\{([^}]*)\}", t):
                key = m.group(1)
                i = skip_ws(t, m.end())
                if i >= len(t) or t[i] != "{":
                    continue
                body, _ = group(t, i)
                fields, k = [], 0
                while k < len(body):
                    k = skip_ws(body, k)
                    if k < len(body) and body[k] == "{":
                        g, k = group(body, k)
                        fields.append(g)
                    else:
                        k += 1
                if key.endswith("@cref"):
                    mt = re.match(r"\[([^\]]*)\]", fields[0] if fields else "")
                    if mt:
                        crefs[key[:-5]] = mt.group(1)
                elif fields and not re.search(r"\?\?|oninputline", fields[0]):
                    # "??oninputline182": label outside any counter (e.g. before \maketitle); skip it
                    labels[key] = {"num": clean_num(fields[0]),
                                   "anchor": fields[3] if len(fields) > 3 else ""}
    for key, lab in labels.items():
        typ = crefs.get(key) or lab["anchor"].split(".")[0]
        if not typ:
            pre = key.split(":")[0].lower() if ":" in key else ""
            typ = {"sec": "section", "ssec": "subsection", "app": "appendix", "fig": "figure",
                   "tab": "table", "eq": "equation", "thm": "theorem", "lem": "lemma",
                   "prop": "proposition", "def": "definition", "alg": "algorithm",
                   "cor": "corollary"}.get(pre, "")
        if typ in ("section", "subsection", "subsubsection") and re.match(r"^[A-Z]", lab["num"]):
            typ = "appendix"
        lab["type"] = typ
        del lab["anchor"]
    return {k: v for k, v in labels.items() if v["num"]}


NAMES = {"section": "Section", "subsection": "Section", "subsubsection": "Section",
         "paragraph": "Section", "appendix": "Appendix", "subappendix": "Appendix",
         "subsubappendix": "Appendix", "figure": "Figure", "subfigure": "Figure",
         "table": "Table", "subtable": "Table", "equation": "Eq.", "align": "Eq.",
         "AMS": "Eq.", "algorithm": "Algorithm", "ALC@unique": "Line", "line": "Line",
         "footnote": "Footnote", "enumi": "Item", "chapter": "Chapter"}


def ref_name(typ):
    if typ in NAMES:
        return NAMES[typ]
    return typ[:1].upper() + typ[1:] if typ else ""


def fmt_num(lab):
    n = lab["num"]
    return f"({n})" if NAMES.get(lab["type"]) == "Eq." else n
