"""Locating the main .tex of a source tree."""
from __future__ import annotations

import os
import re

from tex2md.util.io import read
from tex2md.util.tex import strip_comments


def find_main(src):
    cands = []
    for dp, _, fns in os.walk(src):
        for f in fns:
            if f.endswith(".tex"):
                p = os.path.join(dp, f)
                t = strip_comments(read(p))
                if re.search(r"\\documentclass", t) and "\\begin{document}" in t:
                    cands.append(p)
    if not cands:
        return None
    pref = ("main.tex", "ms.tex", "paper.tex", "arxiv.tex")
    cands.sort(key=lambda p: (os.path.basename(p).lower() not in pref, p.count(os.sep), -os.path.getsize(p)))
    return cands[0]
