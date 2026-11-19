r"""Braces that do not pair up in the body of the document.

LaTeX compiles ``text {The instruction tuned models ...`` (a stray ``{`` the authors never noticed: the group simply
lasts until the end of the document) but pandoc stops at ``\end{document}`` with "unexpected \end", and no
environment contains the error, so there is nothing to drop and retry: the paper is lost. The repair closes such a
group where its paragraph ends and removes a ``}`` that closes nothing, and says so. In the preamble a ``{`` still
open at ``\begin{document}`` (``\usepackage{listings}{``) is removed: pandoc read the whole paper as part of it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from tex2md.util.tex import protected_spans, span_at


@dataclass
class BraceRepair:
    kind: str            # "closed" (a "}" was added), "removed" (a "}" that closed nothing was dropped) or
                         # "dropped-open" (a "{" of the preamble that is never closed was dropped)
    line: int            # 1-based line of the offending brace in the text given to pandoc
    context: str


def repair_braces(text: str) -> tuple[str, list[BraceRepair]]:
    """Return (text with balanced braces in its body, what was changed). In the preamble only a "{" that is still
    open at ``\\begin{document}`` is dropped."""
    start = text.find("\\begin{document}")
    if start < 0:
        return text, []
    verbatim = protected_spans(text, math=False)
    preamble_open = [p for p in _open_braces(text, 0, start, verbatim) if not _DEFINITION.search(_line_of(text, p))]
    if preamble_open:
        dropped = [BraceRepair("dropped-open", text.count("\n", 0, p) + 1, _context(text, p)) for p in preamble_open]
        for p in reversed(preamble_open):
            text = text[:p] + text[p + 1:]
        text, more = repair_braces(text)
        return text, sorted(dropped + more, key=lambda r: r.line)
    stack: list[int] = []
    stray: list[int] = []
    i, n = start, len(text)
    while i < n:
        span = span_at(verbatim, i)
        if span:
            i = span[1]
            continue
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            stack.append(i)
        elif c == "}":
            if stack:
                stack.pop()
            else:
                stray.append(i)
        i += 1
    if not stack and not stray:
        return text, []

    repairs: list[BraceRepair] = []
    inserts: list[tuple[int, str]] = []        # (position, text); applied from the end
    for pos in stray:
        repairs.append(BraceRepair("removed", text.count("\n", 0, pos) + 1, _context(text, pos)))
        inserts.append((pos, ""))
    end_document = text.find("\\end{document}", start)
    for pos in stack:
        stop = text.find("\n\n", pos)
        if stop < 0 or (0 <= end_document < stop):
            stop = len(text[:end_document if end_document >= 0 else n].rstrip())      # right after the last word
        repairs.append(BraceRepair("closed", text.count("\n", 0, pos) + 1, _context(text, pos)))
        inserts.append((stop, "}"))
    out = list(text)
    for pos, add in sorted(inserts, key=lambda p: -p[0]):
        if add:
            out.insert(pos, add)
        else:
            del out[pos]
    return "".join(out), sorted(repairs, key=lambda r: r.line)


_DEFINITION = re.compile(r"\\(?:(?:re)?newcommand|providecommand|DeclareRobustCommand|(?:re)?newenvironment"
                         r"|[gex]?def|NewDocumentCommand|newtheorem)(?![a-zA-Z@])")


def _line_of(text: str, pos: int) -> str:
    return text[text.rfind("\n", 0, pos) + 1:pos]


def _open_braces(text: str, i: int, end: int, verbatim) -> list[int]:
    """Positions of the "{" between ``i`` and ``end`` that are not closed before ``end``."""
    stack: list[int] = []
    while i < end:
        span = span_at(verbatim, i)
        if span:
            i = span[1]
            continue
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            stack.append(i)
        elif c == "}" and stack:
            stack.pop()
        i += 1
    return stack


def _context(text: str, pos: int) -> str:
    return " ".join(text[max(0, pos - 30):pos + 50].split())
