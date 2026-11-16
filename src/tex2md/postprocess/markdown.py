"""Markdown clean-up after pandoc (headings, math)."""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end

MATH_MACRO_RENAMES = (
    # LaTeX/package macros that KaTeX and MathJax lack -> the closest standard ones
    (r"\\Tilde", "\\tilde"), (r"\\Bar(?![a-zA-Z])", "\\bar"), (r"\\widebar", "\\overline"),
    (r"\\mathbold", "\\mathbf"), (r"\\mathpzc", "\\mathcal"), (r"\\mathbbm", "\\mathbb"),
    (r"\\mathds", "\\mathbb"), (r"\\protect", ""), (r"\\phantomsection", ""),
    (r"\\operatornamewithlimits", "\\operatorname"), (r"\\textup", "\\text"),
    (r"\\qedhere", ""), (r"\\numberthis", ""), (r"\\hdots", "\\ldots"), (r"\\dotsc", "\\ldots"),
    (r"\\(?:begin|end)\{minipage\}(?:\{[^{}]*\})?", ""), (r"\\begin\{cases\*\}", "\\begin{cases}"),
    (r"\\end\{cases\*\}", "\\end{cases}"),
)


def cap_heading_levels(body):
    """Markdown stops at ######; deeper pandoc headings (\\subparagraph below \\paragraph) become bold lines."""
    return re.sub(r"(?m)^#{7,}[ \t]+(.+?)[ \t]*#*$", lambda m: "**" + m.group(1) + "**\n", body)


def normalize_math_macros(body):
    # "\\mathbf\\mathbf x" (a font macro applied twice) is an error in KaTeX
    body = re.sub(r"\\(mathbf|boldsymbol|mathcal|mathrm|mathbb|mathit|mathsf|mathtt|mathscr|mathfrak)\s*(?:\\\1(?![a-zA-Z]))+",
                  lambda m: "\\" + m.group(1), body)
    for pat, rep in MATH_MACRO_RENAMES:
        pat = pat if pat.endswith(")") else pat + r"(?![a-zA-Z])"
        body = re.sub(pat, lambda m, rep=rep: rep, body)
    # \ensuremath{X} only matters in text mode; in a math span it is just X
    out, pos = [], 0
    for m in re.finditer(r"\\ensuremath(?=\{)", body):
        if m.start() < pos:
            continue
        e = balanced_end(body, m.end())
        if e > 0:
            out.append(body[pos:m.start()] + body[m.end() + 1:e - 1])
            pos = e
    out.append(body[pos:])
    return "".join(out)


def fix_text_math(body):
    """Math typed inside \\text{...} ($..$ nested in a math span) breaks Markdown math:
    \\text{if $j$ is in $B$} -> \\text{if }j\\text{ is in }B."""
    out, pos = [], 0
    for m in re.finditer(r"\\(?:text|mbox|textnormal|textrm|hbox)(?=\{)", body):
        if m.start() < pos:
            continue
        e = balanced_end(body, m.end())
        if e < 0:
            continue
        inner = body[m.end() + 1:e - 1]
        parts = re.split(r"(?<!\\)\$", inner)
        if len(parts) < 3 or len(parts) % 2 == 0:
            continue
        pieces = []
        for k, part in enumerate(parts):
            if k % 2:
                pieces.append(part.strip())
            elif part.strip():
                pieces.append("\\text{" + part + "}")
        out.append(body[pos:m.start()] + (" ".join(pieces) if pieces else ""))
        pos = e
    out.append(body[pos:])
    return "".join(out)


def isolate_display_math(body):
    """Put every $$...$$ block on its own paragraph (blank lines before and after)."""
    pat = re.compile(r"\$\$(.+?)\$\$[ \t]*", re.S)
    def repl(m):
        line_start = body.rfind("\n", 0, m.start()) + 1
        if re.match(r"\s*(?:[-*+>]|\d+[.)])\s", body[line_start:m.start()]):
            return m.group(0)  # inside a list item or quote: blank lines would break it
        return "\n\n$$" + m.group(1) + "$$\n\n"
    if body.count("$$") % 2:
        return body
    return pat.sub(repl, body)
