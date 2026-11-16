"""tabular environments: column specs and cell clean-up."""
from __future__ import annotations

import re

from tex2md.preprocess.algorithms import _brace_args
from tex2md.util.tex import balanced_end

def escape_citation_ampersands(text):
    """Author-year labels such as "(Wang & Cho, 2019)" inside a tabular add a phantom column."""
    return re.sub(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}",
                  lambda m: re.sub(r"\(([^()&\\]*?)\s&\s([^()&\\]*?)\)", r"(\1 \\& \2)", m.group(0)),
                  text, flags=re.S)


def simple_colspec(spec):
    """Reduce a tabular column spec to plain l/c/r letters: drops @{..} !{..} >{..} <{..} and |,
    maps p/m/b{..} to l, expands *{n}{..}, and treats any other letter (C, Y, X, ...) as c."""
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c in "@!><" and i + 1 < len(spec) and spec[i + 1] == "{":
            e = balanced_end(spec, i + 1)
            i = e if e > 0 else len(spec)
        elif c == "*" and spec[i + 1:i + 2] == "{":
            e = balanced_end(spec, i + 1)
            e2 = balanced_end(spec, e) if 0 < e < len(spec) and spec[e] == "{" else -1
            if e < 0 or e2 < 0:
                break
            try:
                out.append(simple_colspec(spec[e + 1:e2 - 1]) * int(spec[i + 2:e - 1]))
            except ValueError:
                pass
            i = e2
        elif c in "pmb" and spec[i + 1:i + 2] == "{":
            e = balanced_end(spec, i + 1)
            out.append("l")
            i = e if e > 0 else len(spec)
        elif c in "lr":
            out.append(c)
            i += 1
        elif c == "L":
            out.append("l")
            i += 1
        elif c == "R":
            out.append("r")
            i += 1
        elif c.isalpha():
            out.append("c")
            i += 1
        else:
            i += 1
    return "".join(out) or "c"


def normalize_tabulars(text):
    """Make every tabular readable by pandoc: plain column specs, and one-cell tabulars used to
    stack lines inside a cell (\\begin{tabular}{@{}c@{}}a\\\\b\\end{tabular}) or \\makecell become plain text."""
    head = re.compile(r"\\begin\{(tabular\*?)\}(?:\{[^{}]*\})?(?:\[[a-z]\])?(?=\{)")
    pos = 0
    while True:
        m = head.search(text, pos)
        if not m:
            break
        e = balanced_end(text, m.end())
        if e < 0:
            pos = m.end()
            continue
        spec = text[m.end() + 1:e - 1]
        if re.fullmatch(r"@\{\}[lcr]@\{\}", spec):
            end = text.find("\\end{" + m.group(1) + "}", e)
            if end > 0 and "\\begin{tabular" not in text[e:end]:
                body = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", text[e:end])
                text = text[:m.start()] + body + text[end + len("\\end{" + m.group(1) + "}"):]
                pos = m.start()
                continue
        new = simple_colspec(spec)
        text = text[:m.end()] + "{" + new + "}" + text[e:]
        pos = m.end()
    mc = re.compile(r"\\makecell\*?(?:\[[^\]]*\])?(?=\{)")
    while True:
        m = mc.search(text)
        if not m:
            return text
        e = balanced_end(text, m.end())
        if e < 0:
            return text
        body = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", text[m.end() + 1:e - 1])
        text = text[:m.start()] + "{" + body + "}" + text[e:]


def simplify_multicolumn_specs(text):
    """pandoc drops the whole table when a multicolumn spec has @{..}, !{..} or >{..}: keep only c/l/r."""
    out, pos = [], 0
    for m in re.finditer(r"\\multicolumn(?=\s*\{)", text):
        if m.start() < pos:
            continue
        a1, e1 = _brace_args(text, m.end(), 1)
        if a1 is None:
            continue
        i2 = e1
        while i2 < len(text) and text[i2] in " \t\n":
            i2 += 1
        if i2 >= len(text) or text[i2] != "{":
            continue
        e2 = balanced_end(text, i2)
        if e2 < 0:
            continue
        spec = simple_colspec(text[i2 + 1:e2 - 1])[:1] or "c"
        out.append(text[pos:i2] + "{" + spec + "}")
        pos = e2
    out.append(text[pos:])
    return "".join(out)
