"""tabular environments: column specs and cell clean-up."""
from __future__ import annotations

import contextlib
import re

from tex2md.preprocess.algorithms import _brace_args
from tex2md.util.tex import balanced_end, bracket_end, group, unwrap

_CITATION_AMPERSAND = re.compile(r"\(([^()&\\]*?)\s&\s((?:[^()&\\]|\([^()&\\]*\))*?)\)")


def _escape_citation_ampersands(cells):
    return _CITATION_AMPERSAND.sub(r"(\1 \\& \2)", cells)


def escape_citation_ampersands(text):
    """Author-year labels such as "(Wang & Cho, 2019)" inside a tabular add a phantom column."""
    return re.sub(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", lambda m: _escape_citation_ampersands(m.group(0)),
                  text, flags=re.S)


def _skip_column_args(spec, i):
    """Index after the ``[..]`` / ``{..}`` arguments of a column type (``S[table-format=2.1]``, ``w{c}{2cm}``)."""
    while i < len(spec) and spec[i] in "[{":
        if spec[i] == "[":
            end = spec.find("]", i)
            i = end + 1 if end >= 0 else len(spec)
        else:
            end = balanced_end(spec, i)
            i = end if end > 0 else len(spec)
    return i


def simple_colspec(spec):
    """Reduce a tabular column spec to plain l/c/r letters: drops @{..} !{..} >{..} <{..} and |,
    maps p/m/b{..} to l, expands *{n}{..}, and treats any other letter (C, Y, X, S, ...) as c,
    together with the arguments it takes (``S[table-format=2.1]``, ``w{c}{2cm}``, ``X[2,l]``)."""
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
            with contextlib.suppress(ValueError):
                out.append(simple_colspec(spec[e + 1:e2 - 1]) * int(spec[i + 2:e - 1]))
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
            i = _skip_column_args(spec, i + 1)
        elif c == "R":
            out.append("r")
            i = _skip_column_args(spec, i + 1)
        elif c.isalpha():
            out.append("c")
            i = i + 1 if c == "c" else _skip_column_args(spec, i + 1)
        else:
            i += 1
    return "".join(out) or "c"


_MULTICOLUMN_N = re.compile(r"\\multicolumn\s*\{\s*(\d+)\s*\}")


def _row_width(body):
    """Cells of the usual row of a tabular body (``\\multicolumn{n}`` counts n): the most frequent count among the
    rows, the larger on a tie. LaTeX only warns when the rows have more cells than the column spec; pandoc then
    drops the table, caption and all. (Not the widest row: rows written by a macro end inside it, out of sight, and
    run together into one long "row".)"""
    # (the "&" of an aligned or array inside a cell are not cells of the table)
    body = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)", "", body, flags=re.S)
    body = re.sub(r"\\begin\{([a-zA-Z]+\*?)\}.*?\\end\{\1\}", "", body, flags=re.S)
    rows, cells, depth, i = [], 1, 0, 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            if depth == 0 and (body.startswith("\\\\", i) or body.startswith("\\tabularnewline", i)):
                rows.append(cells)
                cells = 1
            else:
                mc = _MULTICOLUMN_N.match(body, i)
                if mc and depth == 0:
                    cells += int(mc.group(1)) - 1
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "&" and depth == 0:
            cells += 1
        i += 1
    if not rows:
        return cells
    return max(rows, key=lambda n: (rows.count(n), n))


def normalize_tabulars(text):
    """Make every tabular readable by pandoc: plain column specs, and one-cell tabulars used to
    stack lines inside a cell (\\begin{tabular}{@{}c@{}}a\\\\b\\end{tabular}) or \\makecell become plain text."""
    # tabular* takes a width before the column spec; a plain tabular does not (its next group is the first cell)
    head = re.compile(r"\\begin\{(tabular\*\}\s*\{[^{}]*|tabular)\}\s*(?:\[[a-z]\])?\s*(?=\{)")
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
            close = "\\end{tabular*}" if m.group(1).startswith("tabular*") else "\\end{tabular}"
            end = text.find(close, e)
            if end > 0 and "\\begin{tabular" not in text[e:end]:
                body = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", text[e:end])
                text = text[:m.start()] + body + text[end + len(close):]
                pos = m.start()
                continue
        new = simple_colspec(spec)
        close = text.find("\\end{tabular", e)
        if close > 0 and "\\begin{tabular" not in text[e:close]:
            new += "c" * max(0, _row_width(_escape_citation_ampersands(text[e:close])) - len(new))
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


def strip_layout_wrappers(text):
    """center/size environments and \\fontsize carry no content (only adjustbox takes an argument)."""
    # only adjustbox takes an argument: "\begin{center}{\fontsize..." must keep its brace
    text = re.sub(r"\\begin\{adjustbox\}(\{[^}]*\})?|\\begin\{(center|small|footnotesize|scriptsize|tiny)\}", "", text)
    text = re.sub(r"\\fontsize\s*\{[^}]*\}\s*\{[^}]*\}(\s*\\selectfont)?", "", text)
    text = re.sub(r"\\end\{(adjustbox|center|small|footnotesize|scriptsize|tiny)\}", "", text)
    return text


def tabu_to_tabular(text):
    r"""The ``tabu`` package: ``\begin{tabu} to \linewidth {spec}`` (or ``spread``), ``longtabu``,
    ``\tabucline{-}`` and ``\rowfont{...}``. Pandoc does not know it and prints the whole table as raw LaTeX."""
    text = re.sub(r"\\begin\{(?:long)?tabu\}\s*(?:(?:to|spread)\s*[^{]*?)?\s*(?=\{)", r"\\begin{tabular}", text)
    text = re.sub(r"\\end\{(?:long)?tabu\}", r"\\end{tabular}", text)
    text = re.sub(r"\\tabucline\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", r"\\hline", text)
    return re.sub(r"\\rowfont\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", text)


def plain_tabular_envs(text):
    """tabularx/tabulary/tabular* take a width before the column spec; NiceTabular does not."""
    text = re.sub(r"\\begin\{(tabularx|tabulary|tabular\*)\}\s*\{(?:[^{}]|\{[^{}]*\})*\}", r"\\begin{tabular}", text)
    text = re.sub(r"\\begin\{NiceTabular\*?\}", r"\\begin{tabular}", text)
    text = re.sub(r"\\end\{(NiceTabular\*?|tabularx|tabulary|tabular\*)\}", r"\\end{tabular}", text)
    return text


def drop_empty_math_cells(text):
    """An empty "$$" cell opens a display equation in pandoc and swallows the whole tabular."""
    text = re.sub(r"(&[ \t]*)\$\$[ \t]*(?=&|\\\\)", r"\1", text)
    text = re.sub(r"\\CodeBefore\b.*?\\Body\b", "", text, flags=re.S)
    return text


def centered_column_types(text):
    """siunitx S columns, X columns and user column types (\\newcolumntype) -> c."""
    def colspec(m):
        spec = re.sub(r"S(\[[^\]]*\])?", "c", m.group(1))
        out, i = [], 0
        while i < len(spec):  # user-defined column types (\newcolumntype) -> c
            ch = spec[i]
            if ch == "{":
                g, i = group(spec, i)
                out.append("{" + g + "}")
                continue
            out.append(ch if (ch in "lcrpmb|@!<>*" or not ch.isalpha()) else "c")
            i += 1
        spec = "".join(out)
        return "\\begin{tabular}{" + spec + "}"
    return re.sub(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", colspec, text)


def unwrap_cell_macros(text):
    """\\resizebox, \\makecell, \\multirow, ...: keep their content, drop the layout."""
    def oneline(body):
        return re.sub(r"\\\\(\[[^\]]*\])?", " ", body)
    for cmd, n in (("resizebox", 2), ("scalebox", 1), ("adjustbox", 1), ("makebox", 0), ("num", 0)):
        text = unwrap(text, cmd, n)
    for cmd in ("makecell", "shortstack", "thead"):
        text = unwrap(text, cmd, 0, oneline)
    # \pbox[t][5ex]{\linewidth}{a\\b}: a cell of stacked lines (pbox package), often inside a cell macro
    text = re.sub(r"\\pbox(?![a-zA-Z@])\s*(?:\[[^\]]*\]\s*)*", r"\\pbox", text)
    text = unwrap(text, "pbox", 1, oneline)
    # \multirow{2}*{Channels}: the width written as a bare "*" (LaTeX reads one token), not as {*}
    text = re.sub(r"(\\multirow\s*(?:\[[^\]]*\]\s*)?\{[^{}]*\})\s*\*", r"\1{*}", text)
    text = unwrap(text, "multirow", 2, oneline)
    return text


def cell_line_breaks(text):
    r"""Inside a tabular, a ``\\`` within a brace group (``\multicolumn{6}{l}{{note 1 \\ note 2}}``) and a blank
    line: LaTeX prints a line break inside the cell, but pandoc cannot read the table and prints it raw. The
    ``\\`` becomes a space (not in math: ``\substack{a\\b}``) and the blank line goes. ``\vspace`` goes too:
    pandoc drops the cell it starts."""
    out, pos = [], 0
    head = re.compile(r"\\begin\{tabular\}")
    while True:
        m = head.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        end = text.find("\\end{tabular}", m.end())
        if end < 0 or "\\begin{tabular}" in text[m.end():end]:
            out.append(text[pos:m.end()])
            pos = m.end()
            continue
        body = re.sub(r"\n[ \t]*(?:%[^\n]*)?\n(?:[ \t]*\n)*", "\n", text[m.end():end])
        # pandoc drops a cell that starts with \vspace (and what follows it in that cell)
        body = re.sub(r"\\vspace\*?\s*\{[^{}]*\}", "", body)
        cells, depth, math, i = [], 0, False, 0
        while i < len(body):
            c = body[i]
            if c == "\\":
                if body.startswith("\\\\", i) and depth > 0 and not math:
                    cells.append(" ")
                else:
                    cells.append(body[i:i + 2])
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == "$":
                math = not math
            cells.append(c)
            i += 1
        out.append(text[pos:m.end()] + "".join(cells) + "\\end{tabular}")
        pos = end + len("\\end{tabular}")


def simplify_longtable_specs(text):
    r"""``\begin{longtable}[]{@{} >{\raggedright\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{.1}} ...}``
    (what pandoc itself writes) -> plain l/c/r columns, which pandoc reads back as a table.

    Also drops the page-break machinery: the header repeated for the following pages
    (``\endfirsthead ... \endhead``) and ``\endfoot``/``\endlastfoot``, which swallow the digits that follow."""
    text = re.sub(r"\\endfirsthead\b.*?\\endhead\b", "", text, flags=re.S)
    text = re.sub(r"\\end(?:last)?foot\b|\\endhead\b|\\endfirsthead\b", "", text)
    head = re.compile(r"\\begin\{longtable\*?\}(?:\[[^\]]*\])?(?=\s*\{)")
    pos = 0
    while True:
        m = head.search(text, pos)
        if not m:
            return text
        start = m.end() + len(text[m.end():]) - len(text[m.end():].lstrip())
        e = balanced_end(text, start)
        if e < 0:
            pos = m.end()
            continue
        new = "\\begin{longtable}{" + simple_colspec(text[start + 1:e - 1]) + "}"
        text = text[:m.start()] + new + text[e:]
        pos = m.start() + len(new)


def cr_to_row_end(text):
    r"""``\cr`` ends a row in plain TeX tables (and in some authors' tabulars): LaTeX takes it, pandoc does not and the
    whole table turns into a paragraph of cells. Inside a tabular it becomes ``\\``."""
    def one(m):
        return re.sub(r"\\cr(?![a-zA-Z@])", lambda _m: "\\\\", m.group(0))
    return re.sub(r"\\begin\{(tabular\*?|longtable\*?|tabularx|tabulary|array|NiceTabular\*?)\}.*?\\end\{\1\}", one, text, flags=re.S)


_ARGS_GLUED = {"multicolumn": 3, "multirow": 3}


def tighten_span_arguments(text):
    r"""``\multicolumn{1}{l}`` + a line break + ``{Text}``: LaTeX reads the three arguments across the break, pandoc stops
    at it and turns the whole table into a paragraph (caption and numbers lost). The blanks between the arguments of
    ``\multicolumn`` and ``\multirow`` are taken out (an optional ``[..]`` of ``\multirow`` is kept)."""
    pattern = re.compile(r"\\(multicolumn|multirow)(?![a-zA-Z@])\*?")
    out, pos = [], 0
    for m in pattern.finditer(text):
        if m.start() < pos:
            continue
        i, parts, groups, ok = m.end(), [], 0, True
        while groups < _ARGS_GLUED[m.group(1)]:
            j = i
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if text[j:j + 1] == "[" and m.group(1) == "multirow" and not parts:       # \multirow[vpos]{n}{width}{text}
                k = bracket_end(text, j)
                if k < 0:
                    ok = False
                    break
                parts.append(text[j:k])
                i = k
                continue
            e = balanced_end(text, j) if text[j:j + 1] == "{" else -1
            if e < 0:
                if groups == _ARGS_GLUED[m.group(1)] - 1 and text[j:j + 1] in ("&", "}", "\\"):
                    parts.append("{}")           # \multirow{2}{*} with no text: the cell ends where the text should be
                    groups += 1
                    i = j
                    break
                ok = False
                break
            parts.append(text[j:e])
            groups += 1
            i = e
        if not ok:
            continue
        parts[-1] = tighten_span_arguments(parts[-1])       # (a \multirow inside the text of a \multicolumn)
        out.append(text[pos:m.end()] + "".join(parts))
        pos = i
    out.append(text[pos:])
    return "".join(out)
