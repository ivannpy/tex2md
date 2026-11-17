"""Number every numbered equation: give the ones without ``\\label`` a label of our own *before* compiling.

LaTeX writes the number of each labelled equation to the .aux, nothing else. Unlabelled equations are
numbered too, so their numbers would be lost. ``label_numbered_equations`` adds ``\\label{tex2md-eq-N}``
to every numbered equation (and every numbered row of an ``align``); the .aux then carries all the
numbers and ``filters/lib/equations.lua`` turns the labels into ``\\tag{number}``.
"""
from __future__ import annotations

import re

from tex2md.util.tex import protected_spans, span_at

SINGLE = ("equation", "multline")                       # one number for the whole environment
MULTI = ("align", "gather", "flalign", "alignat", "eqnarray")   # one number per row
NUMBERED = "|".join(SINGLE + MULTI)

_BEGIN = re.compile(r"\\begin\{(" + NUMBERED + r")\}(\{\d+\})?")
_ROW_END = re.compile(r"\\\\\*?(?:\[[^\]]*\])?")
_BLOCKER = re.compile(r"\\(?:label|tag|notag|nonumber)\b")


def _split_rows(body: str) -> list[tuple[int, int]]:
    """Where the rows of an alignment environment end: (start, end) of each ``\\\\`` outside nested groups
    (with its optional ``*`` / ``[..]``), then the end of the body."""
    cuts: list[tuple[int, int]] = []
    depth, i = 0, 0
    while i < len(body):
        c = body[i]
        if c == "\\" and body[i + 1:i + 2] == "\\" and depth == 0:
            m = _ROW_END.match(body, i)
            end = m.end() if m else i + 2
            cuts.append((i, end))
            i = end
            continue
        if c == "\\":
            m = re.compile(r"\\(begin|end)\{[^}]*\}").match(body, i)
            if m:
                depth += 1 if m.group(1) == "begin" else -1
                i = m.end()
                continue
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    cuts.append((len(body), len(body)))
    return cuts


def _blank_comments(text: str) -> str:
    """The text with every TeX comment overwritten by blanks (same length): what TeX itself reads."""
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m.group(0)), text)


def label_numbered_equations(text: str, prefix: str = "tex2md-eq", start: int = 1) -> tuple[str, int]:
    """Return (text with labels, next free counter). Comments are not read: a commented-out ``\\begin{align}``
    is no environment, and a ``\\\\`` or a ``\\label`` inside a comment is neither a row end nor a label."""
    seen = _blank_comments(text)      # same offsets as ``text``
    verbatim = protected_spans(text, math=False)
    out, pos, n = [], 0, start
    while True:
        m = _BEGIN.search(seen, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out), n
        env = m.group(1)
        end = re.compile(r"\\end\{" + env + r"\}").search(seen, m.end())
        if not end:
            out.append(text[pos:])
            return "".join(out), n
        if span_at(verbatim, m.start()):      # LaTeX shown as an example in a listing
            out.append(text[pos:end.start()])
            pos = end.start()
            continue
        body, visible = text[m.end():end.start()], seen[m.end():end.start()]
        if env in SINGLE:
            if not _BLOCKER.search(visible):
                body += f"\\label{{{prefix}-{n}}}"
                n += 1
        elif re.search(r"\\(?:label|tag)\b", visible) and re.search(r"\\(?:nonumber|notag)\b", visible):
            pass   # amsmath moves the author's label around here ("Multiple \label's"): do not add ours
        else:
            parts, row_start = [], 0
            for cut, after in _split_rows(visible):
                content = body[row_start:cut]
                # (a row never ends inside a comment: the cut was found in the text without comments)
                if visible[row_start:cut].strip() and not _BLOCKER.search(visible[row_start:cut]):
                    content += f"\\label{{{prefix}-{n}}}"
                    n += 1
                parts.append(content + body[cut:after])
                row_start = after
            body = "".join(parts)
        out.append(text[pos:m.end()] + body)
        pos = end.start()
