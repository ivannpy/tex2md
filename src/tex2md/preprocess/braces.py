r"""Braces that do not pair up in the body of the document.

LaTeX compiles ``text {The instruction tuned models ...`` (a stray ``{`` the authors never noticed: the group simply
lasts until the end of the document) but pandoc stops at ``\end{document}`` with "unexpected \end", and no
environment contains the error, so there is nothing to drop and retry: the paper is lost. The repair closes such a
group where its paragraph ends and removes a ``}`` that closes nothing, and says so.
"""
from __future__ import annotations

from dataclasses import dataclass

from tex2md.util.tex import protected_spans, span_at


@dataclass
class BraceRepair:
    kind: str            # "closed" (a "}" was added) or "removed" (a "}" that closed nothing was dropped)
    line: int            # 1-based line of the offending brace in the text given to pandoc
    context: str


def repair_braces(text: str) -> tuple[str, list[BraceRepair]]:
    """Return (text with balanced braces in its body, what was changed). The preamble is not touched."""
    start = text.find("\\begin{document}")
    if start < 0:
        return text, []
    verbatim = protected_spans(text, math=False)
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


def _context(text: str, pos: int) -> str:
    return " ".join(text[max(0, pos - 30):pos + 50].split())
