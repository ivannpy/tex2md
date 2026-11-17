"""Markdown clean-up after pandoc (headings, math)."""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end, bracket_end

MATH_MACRO_RENAMES = (
    # LaTeX/package macros that KaTeX and MathJax lack -> the closest standard ones
    (r"\\Tilde", "\\tilde"), (r"\\Bar(?![a-zA-Z])", "\\bar"), (r"\\widebar", "\\overline"),
    (r"\\mathbold", "\\mathbf"), (r"\\mathpzc", "\\mathcal"), (r"\\mathbbm", "\\mathbb"),
    (r"\\mathds", "\\mathbb"), (r"\\protect", ""), (r"\\phantomsection", ""),
    (r"\\operatornamewithlimits", "\\operatorname"), (r"\\textup", "\\text"),
    (r"\\qedhere", ""), (r"\\numberthis", ""), (r"\\hdots", "\\ldots"), (r"\\dotsc", "\\ldots"),
    (r"\\(?:begin|end)\{minipage\}(?:\{[^{}]*\})?", ""), (r"\\begin\{cases\*\}", "\\begin{cases}"),
    (r"\\end\{cases\*\}", "\\end{cases}"),
    (r"\\mathparagraph", "\\P"), (r"\\mathsection", "\\S"), (r"\\sl", ""),
)


def cap_heading_levels(body):
    """Markdown stops at ######; deeper pandoc headings (\\subparagraph below \\paragraph) become bold lines."""
    return re.sub(r"(?m)^#{7,}[ \t]+(.+?)[ \t]*#*$", lambda m: "**" + m.group(1) + "**\n", body)


def normalize_math_macros(body):
    # "\\mathbf\\mathbf x" (a font macro applied twice) is an error in KaTeX
    body = re.sub(r"\\(mathbf|boldsymbol|mathcal|mathrm|mathbb|mathit|mathsf|mathtt|mathscr|mathfrak)\s*(?:\\\1(?![a-zA-Z]))+",
                  lambda m: "\\" + m.group(1), body)
    for pat, rep in MATH_MACRO_RENAMES:
        # a control word must end at a non-letter; a pattern that ends in "}" or ")" is already complete
        pat = pat + r"(?![a-zA-Z])" if pat[-1].isalpha() else pat
        body = re.sub(pat, lambda m, rep=rep: rep, body)
    # a picture (or a box that only holds one) inside a formula: the figures are not extracted
    body = re.sub(r"\\vcenter\s*\{\s*\\hbox\s*\{\s*(\\includegraphics[^{}]*(?:\{[^{}]*\})?)\s*\}\s*\}", "", body)
    body = re.sub(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})?", "", body)
    body = re.sub(r"\$\{\s*\}\$", "", body)           # the formula that was only that picture
    # commands KaTeX lacks, with a close equivalent that keeps the text
    body = re.sub(r"\\textsc(?=\s*\{)", r"\\text", body)
    body = _unwrap_commands(body, ("ensuremath", "scalebox1", "resizebox2"))
    return _nicefrac(_unwrap_makebox(body))


def _unwrap_makebox(body):
    r"""``\makebox[\widthof{512}][c]{512}`` (a width-aligned cell inside a formula) -> ``512``."""
    out, pos = [], 0
    for m in re.finditer(r"\\makebox(?![a-zA-Z@])\*?", body):
        if m.start() < pos:
            continue
        i = m.end()
        while body[i:i + 1] == "[":
            i = bracket_end(body, i)
            if i < 0:
                break
        e = balanced_end(body, i) if i > 0 and body[i:i + 1] == "{" else -1
        if e > 0:
            out.append(body[pos:m.start()] + body[i + 1:e - 1])
            pos = e
    out.append(body[pos:])
    return "".join(out)


def _unwrap_commands(body, names):
    r"""``\ensuremath{X}`` -> X, ``\scalebox{.8}{X}`` -> X (a trailing 2 in the name: that many groups come first)."""
    for spec in names:
        name, skip = (spec[:-1], int(spec[-1])) if spec[-1].isdigit() else (spec, 0)
        out, pos = [], 0
        for m in re.finditer(r"\\" + name + r"(?![a-zA-Z@])\*?", body):
            if m.start() < pos:
                continue
            i = m.end()
            ok = True
            for _ in range(skip):
                while i < len(body) and body[i] in " \t\n":
                    i += 1
                e = balanced_end(body, i) if body[i:i + 1] == "{" else -1
                if e < 0:
                    ok = False
                    break
                i = e
            while i < len(body) and body[i] in " \t\n":
                i += 1
            e = balanced_end(body, i) if ok and body[i:i + 1] == "{" else -1
            if e > 0:
                out.append(body[pos:m.start()] + body[i + 1:e - 1])
                pos = e
        out.append(body[pos:])
        body = "".join(out)
    return body


def _nicefrac(body):
    r"""``\nicefrac{a}{b}`` (KaTeX has no such command) -> ``{a}/{b}``."""
    out, pos = [], 0
    for m in re.finditer(r"\\(?:nicefrac|sfrac)(?![a-zA-Z@])\s*(?=\{)", body):
        if m.start() < pos:
            continue
        e1 = balanced_end(body, m.end())
        e2 = balanced_end(body, e1) if e1 > 0 and body[e1:e1 + 1] == "{" else -1
        if e2 > 0:
            out.append(body[pos:m.start()] + body[m.end():e1] + "/" + body[e1:e2])
            pos = e2
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


_LIST_ITEM = re.compile(r"( *)(?:[-*+]|\(?(?:\d+|[a-zA-Z]|[ivxlcdm]+|[IVXLCDM]+|#)[.)]) +(?=\S)")
_HTML_BLOCK = re.compile(r"</?(?:table|thead|tbody|tfoot|tr|td|th|colgroup|col|p|pre|code|ul|ol|li|div)\b")
# pandoc fences code with backticks only; a line of "~~~~" is a row of TeX spaces inside an equation
_FENCE_OPEN = re.compile(r"[ ]{0,3}(`{3,})(?:[^`]*|\s*\{.*\})$")
_CODE_MARK = "\x00code%d\x00"


def protect_code(body):
    """Take the code out of the way of the math clean-ups, which read every "$" as math: fenced and indented
    code blocks and `inline code` become placeholders. Returns (body, stash) for ``restore_code``."""
    stash: list[str] = []

    def keep(text):
        stash.append(text)
        return _CODE_MARK % (len(stash) - 1)

    lines = body.split("\n")
    out, i, content = [], 0, 0     # content: column where the text of the current list item starts (0: no list)
    while i < len(lines):
        line = lines[i]
        fence = _FENCE_OPEN.match(line)
        if fence:
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(fence.group(1)):
                j += 1
            out.append(keep("\n".join(lines[i:j + 1])))
            i = j + 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        # what is indented under a list item is its text; code there is indented four more
        if line.strip() and indent >= content + 4 and (i == 0 or not lines[i - 1].strip()):
            j = i
            while j + 1 < len(lines) and (not lines[j + 1].strip() or len(lines[j + 1]) - len(lines[j + 1].lstrip(" ")) >= content + 4):
                j += 1
            while not lines[j].strip():      # the blank lines after the block are not part of it
                j -= 1
            out.append(keep("\n".join(lines[i:j + 1])))
            i = j + 1
            continue
        item = _LIST_ITEM.match(line)
        if item and len(item.group(1)) < content + 4:
            content = item.end()
        elif line.strip() and indent == 0:
            content = 0
        out.append(line)
        i += 1
    body = "\n".join(out)
    return re.sub(r"(?<![`\\])(`+)(?!`)[^\n]+?(?<!`)\1(?!`)", lambda m: keep(m.group(0)), body), stash


def restore_code(body, stash):
    return re.sub(r"\x00code(\d+)\x00", lambda m: stash[int(m.group(1))], body)


def isolate_display_math(body):
    """Put every $$...$$ block on its own paragraph (blank lines before and after). A block inside a list
    item, a quote, a table row or an HTML table stays where it is: blank lines would break those."""
    out, pos = [], 0
    for start, end, display in math_spans(body):
        if not display or start < pos:
            continue
        line_start = body.rfind("\n", 0, start) + 1
        prefix = body[line_start:start]
        if prefix[:1] in (" ", "\t", "|", ">") or _LIST_ITEM.match(prefix) or _HTML_BLOCK.match(prefix):
            continue
        tail = re.compile(r"[ \t]*").match(body, end).end()
        out.append(body[pos:start] + "\n\n" + body[start:end] + "\n\n")
        pos = tail
    out.append(body[pos:])
    return "".join(out)


_MARK = "\u2063"


def restore_math_text(body):
    r"""Undo ``text-in-macro-bodies``: it writes ``\textrm{<U+2063>...}`` so a macro survives in text mode.
    In running text pandoc has already turned that into plain words, so every ``\textrm{<U+2063>`` left is in
    an equation: the author's ``\text{...}`` goes back. The invisible mark is removed everywhere."""
    return body.replace("\\textrm{" + _MARK, "\\text{").replace(_MARK, "")


_MATHML = re.compile(r"<math\b[^>]*>.*?<annotation encoding=\"application/x-tex\">(.*?)</annotation>.*?</math>", re.S)


def mathml_to_tex(body):
    """Tables pandoc can only write as HTML carry formulas as MathML; put back the TeX they came from."""
    import html
    return _MATHML.sub(lambda m: "$" + html.unescape(m.group(1)).strip() + "$", body)


_IFTHENELSE = re.compile(r"\\ifthenelse\s*(?=\{)")


def resolve_ifthenelse(body):
    r"""``\ifthenelse{\equal{X}{}}{A}{B}`` left in an equation by a macro with an optional part
    (``\newcommand{\E}[2]{\ifthenelse{\equal{#1}{}}{\mathbb{E}[#2]}{\mathbb{E}_{#1}[#2]}}``): pandoc has
    already put the arguments in, so the test can be decided here. No math engine knows ``\ifthenelse``."""
    pos = 0
    while True:
        m = _IFTHENELSE.search(body, pos)
        if not m:
            return body
        ends, i = [], m.end()
        for _ in range(3):
            while i < len(body) and body[i] in " \t\n":
                i += 1
            e = balanced_end(body, i) if i < len(body) and body[i] == "{" else -1
            if e < 0:
                break
            ends.append((i, e))
            i = e
        pos = m.end()
        if len(ends) < 3:
            continue
        test = body[ends[0][0] + 1:ends[0][1] - 1].strip()
        value = None
        if test.startswith("\\equal"):
            a = test.find("{")
            ae = balanced_end(test, a) if a >= 0 else -1
            b = ae if ae > 0 and test[ae:ae + 1] == "{" else test.find("{", ae) if ae > 0 else -1
            be = balanced_end(test, b) if b >= 0 else -1
            if be > 0 and not test[be:].strip():
                value = test[a + 1:ae - 1].strip() == test[b + 1:be - 1].strip()
        elif test.startswith("\\isempty"):
            a = test.find("{")
            ae = balanced_end(test, a) if a >= 0 else -1
            if ae > 0 and not test[ae:].strip():
                value = not test[a + 1:ae - 1].strip()
        if value is None:
            continue
        start, end = ends[1] if value else ends[2]
        body = body[:m.start()] + body[start + 1:end - 1].strip() + body[ends[2][1]:]
        pos = m.start()


_COUNTER_HACKS = re.compile(r"\\(?:addtocounter|setcounter|stepcounter)\s*\{[^{}]*\}(?:\s*\{[^{}]*\})?|\\tag\{\\theequation[^{}]*\}")


def drop_counter_hacks(body):
    r"""``\addtocounter{equation}{1}\tag{\theequation}``: LaTeX counter juggling that no Markdown math engine
    knows; the equation already carries its real ``\tag{N}``."""
    return _COUNTER_HACKS.sub("", body)


_COMMENT = re.compile(r"(?<!\\)%[^\n]*")


def math_spans(body):
    """Left-to-right scan for math: ``$$ ... $$`` and ``$ ... $`` (an inline span may cross a line but not a
    blank line; ``\\$`` is a dollar sign). Yields (start, end, is_display)."""
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == "\\":
            i += 2
            continue
        if c != "$":
            i += 1
            continue
        display = body.startswith("$$", i)
        closer = "$$" if display else "$"
        j, start = i + len(closer), i
        while j < n:
            if body[j] == "\\":
                j += 2
                continue
            if body.startswith(closer, j):
                break
            if not display and body.startswith("\n\n", j):
                j = n
                break
            j += 1
        if j >= n:
            i += len(closer)
            continue
        yield start, j + len(closer), display
        i = j + len(closer)


def tidy_math(body):
    r"""Clean-ups inside Markdown math: TeX comments (``% ...`` to the end of the line, usually a bare ``%``
    that glues lines together in LaTeX) are dropped; an inline formula written over several lines becomes one
    line (a line break inside it splits a table row, and a continuation that starts with "-" or "+" reads as a
    list); two inline formulas written back to back (``$1$$\uparrow$``, which reads as the start of a display
    block) are merged into one; and inside a table row "|" becomes ``\vert`` (it would end the cell)."""
    spans = list(math_spans(body))
    out, pos, prev = [], 0, None
    for start, end, display in spans:
        text = body[start:end]
        if "%" in text:
            # only between the delimiters: "...}%$" (a macro that ends in a comment, then the closing dollar)
            # must keep its "$"
            n = 2 if display else 1
            text = text[:n] + _COMMENT.sub("", text[n:-n]) + text[-n:]
        if not display:
            if "\n" in text:
                text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
            if "|" in text and body.startswith("|", body.rfind("\n", 0, start) + 1):
                text = re.sub(r"\\\|", r"\\Vert ", text)
                text = re.sub(r"\|", r"\\vert ", text)
        if prev and not display and prev[1] == start and not prev[2]:
            # glue to the previous inline span: drop the closing and the opening dollar
            out[-1] = out[-1][:-1] + " " + text[1:]
        else:
            out.append(body[pos:start] + text)
        pos, prev = end, (start, end, display)
    out.append(body[pos:])
    return "".join(out)
