"""Locating the main .tex of a source tree."""
from __future__ import annotations

import json
import os
import re

from tex2md.util.io import read
from tex2md.util.tex import strip_comments

_INCLUDES = re.compile(r"\\(?:input|include|subfile|subfileinclude)(?![a-zA-Z@])\s*\{([^{}]+)\}")
_PREFERRED = ("main.tex", "ms.tex", "paper.tex", "arxiv.tex")


def declared_toplevel(src) -> str | None:
    """The main file the authors declared to arXiv: ``00README.json`` (``"usage": "toplevel"``) or the older
    ``00README.XXX`` (a line ``name.tex toplevelfile``). It is arXiv's own source of truth: with several files that
    have a ``\\documentclass`` (a rebuttal, a draft of notes) the heuristic below can pick the wrong one."""
    path = os.path.join(src, "00README.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        for entry in data.get("sources", []) if isinstance(data, dict) else []:
            name = entry.get("filename", "") if isinstance(entry, dict) else ""
            if entry.get("usage") == "toplevel" and name.endswith(".tex") and os.path.isfile(os.path.join(src, name)):
                return os.path.join(src, name)
    path = os.path.join(src, "00README.XXX")
    if os.path.isfile(path):
        for line in read(path).splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "toplevelfile" and os.path.isfile(os.path.join(src, parts[0])):
                return os.path.join(src, parts[0])
    return None


def _reachable(path, src, seen=None) -> set[str]:
    """.tex files of the tree that ``path`` pulls in through ``\\input``, ``\\include`` and ``\\subfile``."""
    seen = set() if seen is None else seen
    folder = os.path.dirname(path)
    for m in _INCLUDES.finditer(strip_comments(read(path))):
        name = m.group(1).strip()
        for base in (folder, src):
            for candidate in (name, name + ".tex"):
                full = os.path.normpath(os.path.join(base, candidate))
                if os.path.isfile(full) and full not in seen:
                    seen.add(full)
                    if full.endswith(".tex"):
                        _reachable(full, src, seen)
                    break
            else:
                continue
            break
    return seen


def main_candidates(src) -> list[str]:
    """Every .tex of the tree that is a complete document (``\\documentclass`` and ``\\begin{document}``), except
    the parts of one: a ``subfiles`` section or a ``standalone`` figure."""
    cands = []
    for dp, _, fns in os.walk(src):
        for f in fns:
            if f.endswith(".tex"):
                p = os.path.join(dp, f)
                t = strip_comments(read(p))
                if re.search(r"\\documentclass", t) and "\\begin{document}" in t \
                        and not re.search(r"\\documentclass\s*(?:\[[^\]]*\])?\s*\{(?:subfiles|standalone)\}", t):
                    cands.append(p)
    return sorted(cands)


def find_main(src, cands=None):
    declared = declared_toplevel(src)
    if declared:
        return declared
    cands = main_candidates(src) if cands is None else cands
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    def key(p):
        # a .bbl of the same name was compiled from this file by the authors; then the file that pulls in more of
        # the tree. The name "main.tex" only breaks ties: it was a draft of notes beside the real paper.
        has_bbl = os.path.isfile(os.path.splitext(p)[0] + ".bbl")
        return (not has_bbl, -len(_reachable(p, src)), os.path.basename(p).lower() not in _PREFERRED,
                p.count(os.sep), -os.path.getsize(p))
    return sorted(cands, key=key)[0]
