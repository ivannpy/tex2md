"""Numbers of the compiled document from the .aux files (labels and floats)."""
from __future__ import annotations

import os
import re

from tex2md.util.io import read
from tex2md.util.tex import clean_num, group, skip_ws, strip_comments

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
    # a reference in a caption reaches the .aux as "\hyperref [x]{Figure~\ref *{x}}"; the Markdown has "Figure 6"
    nums = {m.group(1): clean_num(m.group(2)) for t in auxes.values()
            for m in re.finditer(r"\\newlabel\{([^{}]*)\}\{\{((?:[^{}]|\{[^{}]*\})*)\}", t)}

    def refs(txt):
        txt = re.sub(r"\\hyperref\s*\[[^\]{}]*\]", "", txt)
        return re.sub(r"\\(?:ref|eqref|autoref|cref|Cref)\s*\*?\s*\{([^{}]*)\}", lambda r: " " + nums.get(r.group(1), "") + " ", txt)

    def walk(name):
        if name in seen or name not in auxes:
            return
        seen.add(name)
        t = auxes[name]
        for m in re.finditer(r"\\@input\{([^}]*)\}|\\@writefile\{lo[fta]\}\{\\contentsline \{(figure|table|algorithm)\}\{\\numberline \{([^}]*)\}(.*)", t):
            if m.group(1):
                walk(os.path.basename(m.group(1)))
            else:
                cap = m.group(4).lstrip()
                if cap.startswith("{"):         # only the caption, not the page and anchor after it (short captions)
                    cap = group(cap, 0)[0]
                txt = re.sub(r"\\[a-zA-Z@]+\s*", " ", re.sub(r"^\\ignorespaces\s*", "", refs(_without_font_noise(cap))))
                out.append({"kind": m.group(2), "num": clean_num(m.group(3)),
                            "text": re.sub(r"[^a-z0-9]", "", txt.lower())[:40]})
    walk(root)
    return out


def _caption_key(tex):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\\[a-zA-Z@]+\s*", " ", tex).lower())[:40]


def use_long_captions(floats, tex):
    r"""``\caption[short]{long}`` writes the short form to the .aux, often a placeholder ("Caption") shared by several
    floats: their captions could not be told apart, and a float would take another's number. Put the long form in
    place of the short one, pairing the captions of ``tex`` with the entries of ``floats`` in document order."""
    head = re.compile(r"\\caption(?:of\s*\{[a-z]+\})?\s*\[")
    tex = strip_comments(tex)
    out = [dict(f) for f in floats]
    at = 0
    for m in head.finditer(tex):
        short_end = tex.find("]", m.end())
        if short_end < 0 or "\n\n" in tex[m.end():short_end]:
            continue
        i = skip_ws(tex, short_end + 1)
        if i >= len(tex) or tex[i] != "{":
            continue
        long, _ = group(tex, i)
        short_key, long_key = _caption_key(tex[m.end():short_end]), _caption_key(long)
        if not short_key or short_key == long_key:
            continue
        for j in range(at, len(out)):
            if out[j]["text"] == short_key:
                out[j]["text"] = long_key
                at = j + 1
                break
    return out


def parse_aux(build):
    labels, crefs = {}, {}
    for dp, _, fns in os.walk(build):
        for f in fns:
            if not f.endswith(".aux"):
                continue
            t = read(os.path.join(dp, f))
            for m in re.finditer(r"\\newlabel(?=\{)", t):
                key, i = group(t, m.end())     # (may hold braces: "alg:OeMDM{}-train", from \label{alg:\method{}-train})
                i = skip_ws(t, i)
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



_TOC = re.compile(r"\\@input\{([^}]*)\}|\\@writefile\{toc\}\{\\contentsline\s*\{(section|subsection|subsubsection)\}\s*(?=\{)")
_NUMBERLINE = re.compile(r"\\numberline\s*(?=\{)")


def parse_sections(bdir, base):
    """[{level, num, text}] of the numbered sections the compiled document wrote to its table of contents, in order:
    level 1-3, the number as printed ("2", "3.1", "A.2", "II-A") and the start of the title in the form
    ``parse_floats`` gives captions. Read from ``base.aux`` and the .aux files it ``\\@input``s only: a build folder
    may still hold the .aux of another file that was once taken for the main one."""
    out, seen = [], set()
    levels = {"section": 1, "subsection": 2, "subsubsection": 3}

    def walk(path):
        if path in seen or not os.path.isfile(path):
            return
        seen.add(path)
        t = read(path)
        for m in _TOC.finditer(t):
            if m.group(1):
                walk(os.path.join(bdir, m.group(1)))
                continue
            title, _ = group(t, m.end())
            nl = _NUMBERLINE.match(title.lstrip())
            if not nl:
                continue        # \addcontentsline of an unnumbered heading
            title = title.lstrip()
            num, k = group(title, nl.end())
            num = clean_num(num)
            if not num:
                continue
            txt = re.sub(r"\\[a-zA-Z@]+\s*", " ", _without_font_noise(title[k:]))
            out.append({"level": levels[m.group(2)], "num": num, "text": re.sub(r"[^a-z0-9]", "", txt.lower())[:40]})
    walk(os.path.join(bdir, base + ".aux"))
    return out
