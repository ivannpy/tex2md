"""Keep converting when pandoc cannot parse one construct.

pandoc stops at the first thing it cannot read and says where (``Error at "x.tex" (line L, column C)``) or which
macro sends it round in circles (``Loop encountered in expanding macro X``). Instead of giving up on the whole paper,
``convert_with_recovery`` finds what contains that position and removes it, records it as a diagnostic and tries again
(a few times). Two kinds of construct can go:

* an **environment** (the innermost ``\\begin{x}...\\end{x}`` around the error): replaced by a visible note, because its
  content is missing from the result (``dropped-environment``);
* a **definition** (``\\newcolumntype``, ``\\newcommand``, ``\\def``...): removed without a note, because it holds no text:
  the macro it defines just stays undefined (``dropped-definition``).

The result is a paper with a clearly marked hole instead of no paper.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from tex2md.diagnostics import Diagnostic
from tex2md.util.tex import balanced_end, bracket_end

MAX_RETRIES = 5
_ERROR_AT = re.compile(r'Error at "[^"]*" \(line (\d+), column (\d+)\)')
_LOOP = re.compile(r"Loop encountered in expanding macro (\S+)")
_ENV = re.compile(r"\\(begin|end)\{([^{}]+)\}")
NEVER_DROP = {"document"}

_DEFINER = re.compile(
    r"\\(?:(?:re)?newcommand|providecommand|DeclareRobustCommand|newcolumntype|DeclareMathOperator|(?:re)?newenvironment"
    r"|(?:New|Renew|Declare|Provide)DocumentCommand|newtheorem|DeclarePairedDelimiter(?:X|XPP)?|def|gdef|edef|xdef|let)"
    r"(?![a-zA-Z@])\*?")
_CSNAME = re.compile(r"\\[a-zA-Z@]+|\\[^a-zA-Z@]")


@dataclass
class Dropped:
    name: str
    first_line: int
    last_line: int
    reason: str
    excerpt: str
    kind: str = "environment"        # or "definition"


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


def definition_spans(tex: str) -> list[tuple[int, int, str]]:
    """(start, end, name) of every definition statement: ``\\newcommand{\\x}[2]{..}``, ``\\def\\x#1{..}``,
    ``\\newcolumntype{x}[1]{..}``, ``\\let\\a\\b``..."""
    spans = []
    for m in _DEFINER.finditer(tex):
        i = _skip_blank(tex, m.end())
        name = ""
        group_name = re.compile(r"\{\s*(\\?[^{}\s]+)\s*\}").match(tex, i)
        cs = _CSNAME.match(tex, i)
        if group_name:                       # {\name} (or {c}, a column type)
            name = group_name.group(1).lstrip("\\")
            i = group_name.end()
        elif cs:                             # \name
            name = cs.group(0).lstrip("\\")
            i = cs.end()
        if m.group(0).startswith(("\\def", "\\gdef", "\\edef", "\\xdef")):
            brace = tex.find("{", i)           # the parameter text ("#1#2") comes before the body
            end = balanced_end(tex, brace) if brace >= 0 and "\n\n" not in tex[i:brace] else -1
        elif m.group(0).startswith("\\let"):
            second = _CSNAME.match(tex, _skip_blank(tex, i + (1 if tex[i:i + 1] == "=" else 0)))
            end = second.end() if second else i
        else:
            end = i
            for _ in range(5):                 # [n] [default] {body} ...
                j = _skip_blank(tex, end)
                if tex[j:j + 1] == "[":
                    k = bracket_end(tex, j)
                elif tex[j:j + 1] == "{":
                    k = balanced_end(tex, j)
                else:
                    break
                if k < 0:
                    break
                end = k
        if end > i:
            spans.append((m.start(), end, name))
    return spans


def preamble_statement(tex: str, line: int, col: int) -> tuple[int, int, str] | None:
    r"""The line of the preamble that holds the error (extended until its braces close), when there is nothing better:
    ``\global\setlength{\fboxsep}{0pt}`` and other settings pandoc cannot read. The preamble has no text to lose."""
    pos = _offset(tex, line, col)
    document = tex.find("\\begin{document}")
    if document < 0 or pos >= document:
        return None

    def depth(chunk: str) -> int:
        return sum(1 if c == "{" else -1 if c == "}" else 0 for c in re.sub(r"\\.", "", chunk))
    start = tex.rfind("\n", 0, pos) + 1
    while start > 0 and not tex.startswith("\\", start):         # the line that opens the statement
        start = tex.rfind("\n", 0, start - 1) + 1
    end = tex.find("\n", start)
    end = len(tex) if end < 0 else end
    while depth(tex[start:end]) > 0 and end < document:      # a statement that goes on over several lines
        nxt = tex.find("\n", end + 1)
        if nxt < 0:
            break
        end = nxt
    return start, end, re.sub(r"\s+", " ", tex[start:end]).strip()[:40]


def _skip_blank(tex: str, i: int) -> int:
    while i < len(tex) and tex[i] in " \t\n":
        i += 1
    return i


def enclosing_definition(tex: str, line: int, col: int) -> tuple[int, int, str] | None:
    """The smallest definition statement containing the position."""
    pos = _offset(tex, line, col)
    inside = [s for s in definition_spans(tex) if s[0] <= pos <= s[1]]
    return min(inside, key=lambda s: s[1] - s[0]) if inside else None


def _lines(tex: str, start: int, end: int) -> tuple[int, int]:
    return tex.count("\n", 0, start) + 1, tex.count("\n", 0, end) + 1


def drop_environment(tex: str, span: tuple[int, int, str], reason: str) -> tuple[str, Dropped]:
    start, end, name = span
    first, last = _lines(tex, start, end)
    note = f"\n\n\\textit{{[tex2md: se omitió el entorno «{name}» (líneas {first}–{last} del LaTeX): pandoc no pudo leerlo]}}\n\n"
    excerpt = re.sub(r"\s+", " ", tex[start:end])[:100]
    return tex[:start] + note + tex[end:], Dropped(name, first, last, reason, excerpt)


def drop_definitions(tex: str, spans: list[tuple[int, int, str]], reason: str) -> tuple[str, list[Dropped]]:
    """Remove the given definition statements (no note: they hold no text)."""
    dropped = []
    for start, end, name in sorted(spans, key=lambda s: -s[0]):
        first, last = _lines(tex, start, end)
        dropped.append(Dropped("\\" + name if name else "definition", first, last, reason,
                               re.sub(r"\s+", " ", tex[start:end])[:100], "definition"))
        tex = tex[:start] + tex[end:]
    return tex, dropped[::-1]


def convert_with_recovery(
    tex: str,
    run_once: Callable[[str], tuple[int, str, str]],
    max_retries: int = MAX_RETRIES,
) -> tuple[str | None, str, list[Dropped], str]:
    """Run ``run_once(tex) -> (returncode, stdout, stderr)``, dropping the offending construct on failure.

    Returns ``(body or None, stderr of the last run, what was dropped, final error message)``."""
    dropped: list[Dropped] = []
    for attempt in range(max_retries + 1):
        rc, body, err = run_once(tex)
        if rc == 0 and body.strip():
            return body, err, dropped, ""
        if rc == 0:
            # pandoc ran fine and wrote nothing: there is no text in this document (a PDF wrapper, an empty source)
            return None, err, dropped, "pandoc produced an empty document: the LaTeX source holds no text"
        if attempt == max_retries:
            break
        reason = " ".join(err.strip().split("\n")[:2])[:160]
        loop = _LOOP.search(err)
        if loop:
            # a macro that expands into itself: its definitions have to go, wherever it is used
            spans = [s for s in definition_spans(tex) if s[2] == loop.group(1).lstrip("\\")]
            if spans:
                tex, info = drop_definitions(tex, spans, reason)
                dropped.extend(info)
                continue
            break
        loc = error_location(err)
        if loc is None:
            break
        env = innermost_environment(tex, *loc)
        definition = enclosing_definition(tex, *loc) or preamble_statement(tex, *loc)
        if definition and (env is None or definition[1] - definition[0] < env[1] - env[0]):
            tex, info = drop_definitions(tex, [definition], reason)
            dropped.extend(info)
        elif env:
            tex, one = drop_environment(tex, env, reason)
            dropped.append(one)
        else:
            break
    return None, err, dropped, "pandoc failed: " + (err or "").strip()[-500:]


def diagnostics(dropped: list[Dropped]) -> list[Diagnostic]:
    out = []
    envs = [d for d in dropped if d.kind == "environment"]
    defs = [d for d in dropped if d.kind == "definition"]
    if envs:
        names = [d.name for d in envs]
        examples = [f"{d.name} lines {d.first_line}-{d.last_line}: {d.excerpt}" for d in envs[:3]]
        out.append(Diagnostic("dropped-environment", "error",
                              "environments pandoc could not parse were replaced by a note: their content is NOT in the "
                              "Markdown (" + ", ".join(sorted(set(names))) + ")", len(envs), examples))
    if defs:
        examples = [f"{d.name} lines {d.first_line}-{d.last_line}: {d.excerpt}" for d in defs[:3]]
        out.append(Diagnostic("dropped-definition", "warning",
                              "definitions pandoc could not parse (or that sent it into a loop) were removed: the "
                              "macros they define stay undefined (" + ", ".join(sorted({d.name for d in defs})) + ")",
                              len(defs), examples))
    return out
