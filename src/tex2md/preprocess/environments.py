"""Environments pandoc does not know and would print as text, and glossary commands."""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end, group, skip_ws

# drawing code: pandoc drops tikzpicture itself but prints these as running text
_PICTURES = "forest|pgfpicture|circuitikz|pspicture|dependency"


def drop_picture_code(text: str) -> str:
    r"""``forest`` trees and other drawing environments, and inline ``\tikz ...;`` pictures: their source is
    code, not text (pandoc drops ``\tikz`` itself and prints the coordinates that follow)."""
    text = re.sub(r"\\begin\{(" + _PICTURES + r")\*?\}.*?\\end\{\1\*?\}", "", text, flags=re.S)
    text = re.sub(r"\\tikz(?![a-zA-Z@])\s*(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}", "", text)
    return re.sub(r"\\tikz(?![a-zA-Z@])\s*(?:\[[^\]]*\])?(?:[^;{}\n]|\{[^{}]*\})*;", "", text)


def unwrap_multicols(text: str) -> str:
    r"""``\begin{multicols}{2}``: pandoc keeps the content but prints the "2" in front of it."""
    text = re.sub(r"\\begin\{multicols\*?\}\s*\{[^{}]*\}(?:\s*\[[^\]]*\])?", "", text)
    return re.sub(r"\\end\{multicols\*?\}", "", text)


def restatable_to_theorem(text: str) -> str:
    r"""thm-restate: ``\begin{restatable}[title]{theorem}{name}`` -> ``\begin{theorem}[title]``. pandoc would
    print "theoremname" as the first words of the statement."""
    tok = re.compile(r"\\(begin|end)\{restatable\*?\}")
    out: list[str] = []
    stack: list[str] = []
    pos = 0
    for m in tok.finditer(text):
        if m.start() < pos:
            continue
        if m.group(1) == "end":
            if stack:
                out.append(text[pos:m.start()] + "\\end{" + stack.pop() + "}")
                pos = m.end()
            continue
        i, opt = skip_ws(text, m.end()), ""
        if i < len(text) and text[i] == "[":
            inner, j = group(text, i, "[", "]")
            opt, i = "[" + inner + "]", skip_ws(text, j)
        e1 = balanced_end(text, i) if i < len(text) and text[i] == "{" else -1
        e2 = balanced_end(text, e1) if e1 > 0 and text[e1:e1 + 1] == "{" else -1
        if e2 < 0:
            continue
        env = text[i + 1:e1 - 1].strip()
        stack.append(env)
        out.append(text[pos:m.start()] + "\\begin{" + env + "}" + opt)
        pos = e2
    out.append(text[pos:])
    return "".join(out)


def tblr_to_tabular(text: str) -> str:
    r"""tabularray: ``\begin{tblr}{colspec={lcc}, hlines}`` -> ``\begin{tabular}{lcc}`` (pandoc does not know
    ``tblr`` and prints the whole table as one paragraph)."""
    head = re.compile(r"\\begin\{(tblr|longtblr|talltblr)\}\s*(?:\[[^\]]*\])?\s*(?=\{)")
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
        inner = re.search(r"colspec\s*=\s*(?=\{)", spec)
        if inner:
            ie = balanced_end(spec, inner.end())
            spec = spec[inner.end() + 1:ie - 1] if ie > 0 else "c"
        elif "=" in spec:
            spec = "c"      # only keys, no column list: let pandoc size the table from the rows
        new = "\\begin{tabular}{" + spec + "}"
        text = text[:m.start()] + new + text[e:]
        pos = m.start() + len(new)
    return re.sub(r"\\end\{(?:tblr|longtblr|talltblr)\}", r"\\end{tabular}", text)


_ACRONYM = re.compile(r"\\(?:newacronym|newabbreviation)\s*(?:\[[^\]]*\])?\s*(?=\{)")
_ENTRY = re.compile(r"\\newglossaryentry\s*(?=\{)")
_GLS = re.compile(r"\\(gls|Gls|GLS|glspl|Glspl|GLSpl|acrshort|Acrshort|acrlong|Acrlong|acrfull|Acrfull|acs|acl|acf"
                  r"|glsentryshort|glsentrylong|glsentrytext|glsentryname|glstext|glsresetall)(?![a-zA-Z@])\*?"
                  r"\s*(?:\[[^\]]*\])?\s*(?:\{([^{}]*)\})?")


def _three_groups(text, i):
    out = []
    for _ in range(3):
        i = skip_ws(text, i)
        e = balanced_end(text, i) if i < len(text) and text[i] == "{" else -1
        if e < 0:
            return None
        out.append(text[i + 1:e - 1])
        i = e
    return out


def expand_glossary(text: str) -> str:
    r"""glossaries: ``\gls{llm}`` -> "large language model (LLM)" the first time and "LLM" afterwards, from the
    ``\newacronym{llm}{LLM}{large language model}`` of the document (pandoc prints the label, "llm")."""
    entries = {}      # label -> (short, long or None)
    for m in _ACRONYM.finditer(text):
        g = _three_groups(text, m.end())
        if g:
            entries[g[0].strip()] = (g[1], g[2])
    for m in _ENTRY.finditer(text):
        e1 = balanced_end(text, m.end())
        e2 = balanced_end(text, skip_ws(text, e1)) if e1 > 0 and text[skip_ws(text, e1):skip_ws(text, e1) + 1] == "{" else -1
        if e2 < 0:
            continue
        body = text[skip_ws(text, e1) + 1:e2 - 1]
        name = re.search(r"(?:^|,)\s*(?:name|text)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|[^,{}]*)", body)
        if name:
            entries[text[m.end() + 1:e1 - 1].strip()] = (name.group(1).strip().strip("{}"), None)
    if not entries:
        return text
    body_at = max(text.find("\\begin{document}"), 0)
    used: set[str] = set()

    def capital(s):
        return s[:1].upper() + s[1:]

    def use(m):
        cmd, label = m.group(1), (m.group(2) or "").strip()
        if cmd == "glsresetall":
            used.clear()
            return ""
        if m.start() < body_at or label not in entries:
            return m.group(0)
        short, long = entries[label]
        plural = "s" if cmd.lower().endswith("pl") else ""
        kind = cmd.lower()
        if kind in ("acrshort", "acs", "glsentryshort") or long is None:
            out = short + plural
        elif kind in ("acrlong", "acl", "glsentrylong"):
            out = long + plural
        elif kind in ("acrfull", "acf") or label not in used:
            out = f"{long}{plural} ({short}{plural})"
        else:
            out = short + plural
        if kind.startswith("gls") and not kind.startswith("glsentry"):
            used.add(label)
        if cmd.isupper() or cmd == "GLSpl":
            return out.upper()
        return capital(out) if cmd[0].isupper() else out
    return _GLS.sub(use, text)


_LEFT_BRACE_TABULAR = re.compile(
    r"(\\left\s*(?:\\\{|\\lbrace|\\lgroup|\\lbrack|\\\[|\(|\.))\s*\\begin\{tabular\}\s*(?=\{)")


def tabular_in_math(text: str) -> str:
    r"""``$x = \left\{ \begin{tabular}{ll} $1$ & $q \le p$ \\ ... \end{tabular}\right.$``: a tabular used as a
    piecewise definition inside a formula. Its cells have their own ``$``, which pandoc takes for the end of the
    formula and reads the rest of the paper with math and text swapped. The tabular becomes an ``array`` (whose
    cells are already math) with a plain column spec, and the ``$`` of its cells go."""
    from tex2md.preprocess.tables import simple_colspec
    pos = 0
    while True:
        m = _LEFT_BRACE_TABULAR.search(text, pos)
        if not m:
            return text
        spec_end = balanced_end(text, m.end())
        end = text.find("\\end{tabular}", spec_end) if spec_end > 0 else -1
        if end < 0:
            return text
        body = re.sub(r"(?<!\\)\$", "", text[spec_end:end])
        new = m.group(1) + "\\begin{array}{" + simple_colspec(text[m.end() + 1:spec_end - 1]) + "}" + body + "\\end{array}"
        text = text[:m.start()] + new + text[end + len("\\end{tabular}"):]
        pos = m.start() + len(new)


_MATH_BOX = re.compile(r"\\(?:resizebox\*?\s*\{[^{}]*\}\s*\{[^{}]*\}|scalebox\s*\{[^{}]*\}(?:\s*\[[^\]]*\])?)\s*(?=\{\s*\$)")
_DISPLAY_OPEN = re.compile(r"\\\[|\\begin\{(?:equation|align|gather|multline|flalign|eqnarray)\*?\}")
_DISPLAY_CLOSE = re.compile(r"\\\]|\\end\{(?:equation|align|gather|multline|flalign|eqnarray)\*?\}")


def _in_display_math(text: str, pos: int) -> bool:
    """True when ``pos`` lies after a display-math opening that has not been closed (looks back one paragraph)."""
    start = text.rfind("\n\n", 0, pos)
    chunk = text[start + 1 if start >= 0 else 0:pos]
    opens = [m.end() for m in _DISPLAY_OPEN.finditer(chunk)]
    if not opens:
        return False
    return not _DISPLAY_CLOSE.search(chunk, opens[-1])


def boxes_around_math(text: str) -> str:
    r"""``\[\resizebox{\linewidth}{!}{$\displaystyle ...$}\]``: a formula scaled to the line width. The box is
    layout, and inside display math its ``$`` read as the end of the equation (the rest of the paper swaps math
    and text). The formula stays: bare inside display math, ``$...$`` in running text.

    ``\fitbox{W}{...}`` (a class file's \resizebox that only shrinks: pandoc never sees its definition and drops
    the table or the formula in it) is read as ``\resizebox{W}{!}{...}`` first."""
    text = re.sub(r"\\fitbox(?![a-zA-Z@])\s*(\{[^{}]*\})", r"\\resizebox\1{!}", text)
    pos = 0
    while True:
        m = _MATH_BOX.search(text, pos)
        if not m:
            return text
        e = balanced_end(text, m.end())
        inner = text[m.end() + 1:e - 1].strip() if e > 0 else ""
        if not (inner.startswith("$") and inner.endswith("$") and not inner.startswith("$$")
                and len(re.findall(r"(?<!\\)\$", inner)) == 2):
            pos = m.end()
            continue
        new = inner[1:-1].strip() if _in_display_math(text, m.start()) else inner
        text = text[:m.start()] + new + text[e:]
        pos = m.start() + len(new)


def acks_to_section(text: str) -> str:
    r"""JMLR's ``\acks{...}`` and NeurIPS' ``\begin{ack}...\end{ack}`` print an Acknowledgments heading and the text; the
    macro or environment lives in the style file, so pandoc drops the first and the heading of the second."""
    body_at = text.find("\\begin{document}")
    pos = 0
    pattern = re.compile(r"\\acks?(?![a-zA-Z@])\s*(?=\{)")
    while True:
        m = pattern.search(text, pos)
        if not m:
            break
        e = balanced_end(text, m.end())
        if m.start() < body_at or e < 0:
            pos = m.end()
            continue
        new = "\n\n\\section*{Acknowledgments}\n\n" + text[m.end() + 1:e - 1].strip() + "\n\n"
        text = text[:m.start()] + new + text[e:]
        pos = m.start() + len(new)
    return re.sub(r"\\begin\{ack\}(.*?)\\end\{ack\}", lambda m: "\n\n\\section*{Acknowledgments}\n\n" + m.group(1).strip() + "\n\n",
                  text, flags=re.S)


def unwrap_picture_puts(text: str) -> str:
    r"""``\begin{picture}(w,h) \put(x,y){...} \end{picture}`` laying out an image and a table: their content, without
    the coordinates (pandoc prints them, and the float around it is no longer read as a table)."""
    def inner(m):
        body = re.sub(r"\\(?:put|makebox)\s*\([^()]*\)(?:\[[^\]]*\])?\s*(?=\{)", "", m.group(1))
        return re.sub(r"\\(?:multiput|qbezier|line|vector|circle\*?|oval)\s*(?:\([^()]*\)\s*)+(?:\{[^{}]*\})?", "", body)
    return re.sub(r"\\begin\{picture\}\s*(?:\([^()]*\)\s*){1,2}(.*?)\\end\{picture\}", inner, text, flags=re.S)


# ctable's row commands: first, middle and last rule, and a plain row end
_CTABLE_ROWS = [(r"\\FL(?![a-zA-Z@])(?:\[[^\]]*\])?", r"\\toprule "), (r"\\ML(?![a-zA-Z@])(?:\[[^\]]*\])?", r"\\\\ \\midrule "),
                (r"\\LL(?![a-zA-Z@])(?:\[[^\]]*\])?", r"\\\\ \\bottomrule "), (r"\\NN(?![a-zA-Z@])(?:\[[^\]]*\])?", r"\\\\ ")]


def ctable_to_table(text: str) -> str:
    r"""ctable: ``\ctable[caption={..}, label={..}]{cols}{notes}{rows}`` -> a ``table`` with that caption, label and
    tabular, and its ``\FL``/``\ML``/``\NN``/``\LL`` row commands -> booktabs rules. pandoc does not know the
    command and prints its arguments as text."""
    if not re.search(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{[^{}]*\bctable\b", text):
        return text         # (\FL, \NN... may be the author's own macros)
    out, pos = [], 0
    for m in re.finditer(r"\\ctable(?![a-zA-Z@])\s*", text):
        if m.start() < pos:
            continue
        i, opts = m.end(), ""
        if i < len(text) and text[i] == "[":
            opts, i = group(text, i, "[", "]")
        args = []
        for _ in range(3):
            i = skip_ws(text, i)
            if i >= len(text) or text[i] != "{":
                break
            arg, i = group(text, i)
            args.append(arg)
        if len(args) < 3:
            continue

        def option(name, opts=opts):
            o = re.search(r"(?:^|,)\s*" + name + r"\s*=\s*", opts)
            if not o:
                return ""
            j = skip_ws(opts, o.end())
            if j < len(opts) and opts[j] == "{":
                return group(opts, j)[0]
            return opts[j:].split(",")[0].strip()
        cap, lab = option("caption"), option("label")
        table = "\\begin{table}" + (f"\\caption{{{cap}}}" if cap else "") + (f"\\label{{{lab}}}" if lab else "")
        table += f"\\begin{{tabular}}{{{args[0]}}}{args[2]}\\end{{tabular}}" + (f"\n{args[1]}\n" if args[1].strip() else "")
        out.append(text[pos:m.start()] + table + "\\end{table}")
        pos = i
    text = "".join(out) + text[pos:]
    for pat, rep in _CTABLE_ROWS:
        text = re.sub(pat, rep, text)
    return text
