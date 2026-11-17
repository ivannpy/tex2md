"""Number every numbered equation: give the ones without ``\\label`` a label of our own *before* compiling.

LaTeX writes the number of each labelled equation to the .aux, nothing else. Unlabelled equations are
numbered too, so their numbers would be lost. ``label_numbered_equations`` adds ``\\label{tex2md-eq-N}``
to every numbered equation (and every numbered row of an ``align``); the .aux then carries all the
numbers and ``filters/lib/equations.lua`` turns the labels into ``\\tag{number}``.
"""
from __future__ import annotations

import re

SINGLE = ("equation", "multline")                       # one number for the whole environment
MULTI = ("align", "gather", "flalign", "alignat", "eqnarray")   # one number per row
NUMBERED = "|".join(SINGLE + MULTI)

_BEGIN = re.compile(r"\\begin\{(" + NUMBERED + r")\}(\{\d+\})?")
_ROW_END = re.compile(r"\\\\\*?(?:\[[^\]]*\])?")
_BLOCKER = re.compile(r"\\(?:label|tag|notag|nonumber)\b")


def _split_rows(body: str) -> list[tuple[str, str]]:
    """Rows of an alignment environment: (content, separator), cutting at ``\\\\`` outside nested groups."""
    rows, depth, start, i = [], 0, 0, 0
    while i < len(body):
        c = body[i]
        if c == "\\" and body[i + 1:i + 2] == "\\" and depth == 0:
            m = _ROW_END.match(body, i)
            end = m.end() if m else i + 2
            rows.append((body[start:i], body[i:end]))
            i = start = end
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
    rows.append((body[start:], ""))
    return rows


def label_numbered_equations(text: str, prefix: str = "tex2md-eq", start: int = 1) -> tuple[str, int]:
    """Return (text with labels, next free counter)."""
    out, pos, n = [], 0, start
    while True:
        m = _BEGIN.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out), n
        env = m.group(1)
        end = re.compile(r"\\end\{" + env + r"\}").search(text, m.end())
        if not end:
            out.append(text[pos:])
            return "".join(out), n
        body = text[m.end():end.start()]
        if env in SINGLE:
            if not _BLOCKER.search(body):
                body += f"\\label{{{prefix}-{n}}}"
                n += 1
        elif re.search(r"\\(?:label|tag)\b", body) and re.search(r"\\(?:nonumber|notag)\b", body):
            pass   # amsmath moves the author's label around here ("Multiple \label's"): do not add ours
        else:
            parts = []
            for content, sep in _split_rows(body):
                if content.strip() and not _BLOCKER.search(content):
                    content += f"\\label{{{prefix}-{n}}}"
                    n += 1
                parts.append(content + sep)
            body = "".join(parts)
        out.append(text[pos:m.end()] + body)
        pos = end.start()
