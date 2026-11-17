"""Keep converting when pandoc cannot parse one construct.

pandoc stops at the first thing it cannot read and says where (``Error at "x.tex" (line L, column C)``).
Instead of giving up on the whole paper, ``convert_with_recovery`` finds the innermost environment that contains
that position, replaces it with a visible note, records it as a ``dropped-environment`` diagnostic and tries again
(a few times). The result is a paper with a clearly marked hole instead of no paper.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from tex2md.diagnostics import Diagnostic

MAX_RETRIES = 5
_ERROR_AT = re.compile(r'Error at "[^"]*" \(line (\d+), column (\d+)\)')
_ENV = re.compile(r"\\(begin|end)\{([^{}]+)\}")
NEVER_DROP = {"document"}


@dataclass
class Dropped:
    name: str
    first_line: int
    last_line: int
    reason: str
    excerpt: str


def error_location(err: str) -> tuple[int, int] | None:
    m = _ERROR_AT.search(err or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _offset(tex: str, line: int, col: int) -> int:
    lines = tex.split("\n")
    line = min(max(line, 1), len(lines))
    return sum(len(x) + 1 for x in lines[:line - 1]) + max(col - 1, 0)


def innermost_environment(tex: str, line: int, col: int) -> tuple[int, int, str] | None:
    """(start, end, name) of the smallest closed ``\\begin{x}...\\end{x}`` containing the position."""
    pos = _offset(tex, line, col)
    stack: list[re.Match] = []
    best: tuple[int, int, str] | None = None
    for m in _ENV.finditer(tex):
        if m.group(1) == "begin":
            stack.append(m)
            continue
        for i in range(len(stack) - 1, -1, -1):
            if stack[i].group(2) == m.group(2):
                start, end, name = stack[i].start(), m.end(), m.group(2)
                del stack[i:]
                if name not in NEVER_DROP and start <= pos <= end and (best is None or end - start < best[1] - best[0]):
                    best = (start, end, name)
                break
    return best


def drop_environment(tex: str, span: tuple[int, int, str], reason: str) -> tuple[str, Dropped]:
    start, end, name = span
    first = tex.count("\n", 0, start) + 1
    last = tex.count("\n", 0, end) + 1
    note = f"\n\n\\textit{{[tex2md: se omitió el entorno «{name}» (líneas {first}–{last} del LaTeX): pandoc no pudo leerlo]}}\n\n"
    excerpt = re.sub(r"\s+", " ", tex[start:end])[:100]
    return tex[:start] + note + tex[end:], Dropped(name, first, last, reason, excerpt)


def convert_with_recovery(
    tex: str,
    run_once: Callable[[str], tuple[int, str, str]],
    max_retries: int = MAX_RETRIES,
) -> tuple[str | None, str, list[Dropped], str]:
    """Run ``run_once(tex) -> (returncode, stdout, stderr)``, dropping the offending environment on failure.

    Returns ``(body or None, stderr of the last run, dropped environments, final error message)``."""
    dropped: list[Dropped] = []
    for attempt in range(max_retries + 1):
        rc, body, err = run_once(tex)
        if rc == 0 and body.strip():
            return body, err, dropped, ""
        loc = error_location(err)
        span = innermost_environment(tex, *loc) if loc else None
        if span is None or attempt == max_retries:
            break
        reason = " ".join(err.strip().split("\n")[:2])[:160]
        tex, info = drop_environment(tex, span, reason)
        dropped.append(info)
    return None, err, dropped, "pandoc failed: " + (err or "").strip()[-500:]


def diagnostics(dropped: list[Dropped]) -> list[Diagnostic]:
    if not dropped:
        return []
    names = [d.name for d in dropped]
    examples = [f"{d.name} lines {d.first_line}-{d.last_line}: {d.excerpt}" for d in dropped[:3]]
    return [Diagnostic("dropped-environment", "error",
                       "environments pandoc could not parse were replaced by a note: their content is NOT in the "
                       "Markdown (" + ", ".join(sorted(set(names))) + ")", len(dropped), examples)]
