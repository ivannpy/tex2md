"""User macros, template styles and macro-definition clean-up."""
from __future__ import annotations

import os
import re

from tex2md.util.io import read
from tex2md.util.tex import balanced_end, bracket_end, protected_spans, skip_ws, span_at

SIZES = r"(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)"


def drop_macro_defs(text, pattern, replace_uses=None, cmd=r"(?:re)?newcommand"):
    """Remove \\newcommand-style definitions whose name matches `pattern` (regex over the
    macro name). With replace_uses, calls to the dropped macro become that literal."""
    head = re.compile(r"\\" + cmd + r"\*?\s*\{?\\(" + pattern + r")\}?\s*(\[(\d)\])?\s*(\[[^\]]*\])?\s*(?=\{)")
    dropped = {}
    pos = 0
    while True:
        m = head.search(text, pos)
        if not m:
            break
        end = balanced_end(text, m.end())
        if end < 0:
            pos = m.end()
            continue
        dropped[m.group(1)] = (int(m.group(3) or 0), bool(m.group(4)))
        text = text[:m.start()] + text[end:]
        pos = m.start()
    if replace_uses is not None:
        for name, (n, opt) in dropped.items():
            use = r"\\" + name + r"(?![a-zA-Z@])" + (r"(?:\[[^\]]*\])?" if opt else "") + r"(?:\s*\{[^{}]*\}){0,%d}" % (n - 1 if opt else n)  # noqa: UP031
            text = re.sub(use, replace_uses, text)
    return text


def fix_stray_macro_braces(text):
    """A zero-argument macro followed by a "{" that never closes in its paragraph (\\owt{. ...)
    is a typo for \\owt{}; TeX survives it, pandoc does not."""
    names = set(re.findall(r"\\def\\([a-zA-Z]+)\{", text)) | \
        set(re.findall(r"\\newcommand\*?\{\\([a-zA-Z]+)\}\s*\{", text))
    for name in names:
        out, pos = [], 0
        for m in re.finditer(r"\\" + name + r"(?![a-zA-Z@])\{(?!\})", text):
            if m.start() < pos:
                continue
            para_end = text.find("\n\n", m.end())
            para = text[m.end():para_end if para_end >= 0 else len(text)]
            depth = 1
            for c in re.sub(r"\\.", "", para):
                depth += (c == "{") - (c == "}")
                if depth == 0:
                    break
            if depth > 0:
                out.append(text[pos:m.end() - 1])
                pos = m.end()
        out.append(text[pos:])
        text = "".join(out)
    return text


TEMPLATE_STY = re.compile(r"(?i)^(natbib|fancyhdr|neurips|nips|iclr|icml|colm|cvpr|iccv|eccv|acl|emnlp|naacl|aaai|ijcai|tmlr|jmlr|"
                          r"ieee|llncs|spconf|styles?)|(?:^|_)(?:19|20)\d\d|conference$")


DEF_HEAD = re.compile(r"\\(?:newcommand|providecommand|DeclareMathOperator|DeclarePairedDelimiter|DeclareRobustCommand|def|gdef)\*?"
                      r"(?:\s*\{\s*\\([A-Za-z@]+)\s*\}|\s*\\([A-Za-z@]+))")


PROTECTED_MACROS = {"section", "subsection", "subsubsection", "paragraph", "subparagraph", "maketitle", "title", "author",
                    "date", "and", "thanks", "keywords", "abstract", "item", "caption", "cite", "label", "ref", "bibliography"}


def local_style_macros(text, bdir):
    """Macros defined in local .sty files loaded with \\usepackage (notation.sty, myabbrev.sty, ...).
    pandoc never opens them, so their macros stay unexpanded in the math. Returns definitions only:
    the ones the document uses, plus (transitively) the ones those definitions use."""
    cands = []   # (name, definition text, body text)
    doc_body = text[text.find("\\begin{document}"):]
    for names in re.findall(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", text):
        for name in (n.strip() for n in names.split(",")):
            path = os.path.join(bdir, name + ".sty")
            if not name or TEMPLATE_STY.search(name) or not os.path.isfile(path):
                continue
            sty = re.sub(r"(?m)(?<!\\)%.*$", "", read(path))
            sty = re.sub(r"\\makeatletter.*?\\makeatother", "", sty, flags=re.S)
            pos = 0
            while True:
                m = DEF_HEAD.search(sty, pos)
                if not m:
                    break
                i = m.end()
                # skip argument specs up to the body's opening brace
                while i < len(sty) and (sty[i] != "{" or sty[i - 1] == "\\"):
                    if sty[i] == "\n" and sty[i + 1:i + 2] == "\n":
                        break
                    i += 1
                e = balanced_end(sty, i) if i < len(sty) and sty[i] == "{" else -1
                if e > 0 and "@" not in m.group(0) and "@" not in (m.group(1) or m.group(2)) and "\\@" not in sty[m.start():e]:
                    j = e
                    if m.group(0).startswith("\\DeclareMathOperator") or "DeclarePairedDelimiter" in m.group(0):
                        k = sty.find("{", e)
                        j = balanced_end(sty, k) if k >= 0 and not sty[e:k].strip() else e
                    name = m.group(1) or m.group(2)
                    if (name not in PROTECTED_MACROS
                            and not re.search(r"##|\\(?:def|gdef|let|if[a-zA-Z]*|expandafter|csname|hss|llap|rlap|ifx)\b", sty[m.end():j])):
                        cands.append((name, sty[m.start():j], sty[m.end():j]))
                    pos = j
                else:
                    pos = m.end()
    wanted = {n for n, _, _ in cands if re.search(r"\\" + n + r"(?![A-Za-z@])", doc_body)}
    grew = True
    while grew:   # a used macro may be built from other macros of the same .sty
        grew = False
        for _, _, body_w in [c for c in cands if c[0] in wanted]:
            for n, _, _ in cands:
                if n not in wanted and re.search(r"\\" + n + r"(?![A-Za-z@])", body_w):
                    wanted.add(n)
                    grew = True
    return "\n".join(d for n, d, _ in cands if n in wanted)


def drop_fragile_macro_defs(text):
    """Macros that typeset with TeX primitives pandoc cannot expand (\\texttt via \\ttfamily,
    algorithm commands, colour boxes)."""
    if re.search(r"\\renewcommand\s*\{\\texttt\}[^{]*\{[^}]*\\ttfamily", text):
        text = drop_macro_defs(text, "texttt")
    text = drop_macro_defs(text, "[a-zA-Z]+", cmd=r"alg(?:re)?newcommand")
    boxes = set()
    for m in re.finditer(r"\\(?:newcommand|DeclareRobustCommand)\*?\s*\{?\\([a-zA-Z]+)\}?\s*(?:\[\d\])?\s*(?:\[[^\]]*\])?\s*(?=\{)", text):
        end = balanced_end(text, m.end())
        if end > 0 and "\\colorbox" in text[m.end():end]:
            boxes.add(m.group(1))
    for name in boxes:
        text = drop_macro_defs(text, name, "\u25a0", cmd=r"(?:(?:re)?newcommand|DeclareRobustCommand)")
    return text


def drop_primitive_spacing(text):
    """Primitive spacing with length registers (\\vskip\\bigskipamount) confuses pandoc, and so does \\strut (an
    invisible rule): in ``\\textbf{\\strut z}`` pandoc stops with an error, inside a caption it drops the table."""
    verbatim = protected_spans(text, math=False)          # (a listing keeps its code as written)
    text = re.sub(r"\\strut(?![a-zA-Z@])\s*", lambda m: m.group(0) if span_at(verbatim, m.start()) else "", text)
    return re.sub(r"\\[vh]skip\s*-?\s*(?:\\[a-zA-Z@]+|[0-9.]+\s*[a-z]{2})(?:\s*(?:plus|minus)\s*-?[0-9.]*\s*(?:\\[a-zA-Z@]+|[a-z]{2}))*", "", text)


def drop_digit_operators(text):
    """Operator names TeX tolerates but pandoc rejects (\\expm1)."""
    return re.sub(r"\\DeclareMathOperator\*?\s*\{\\[a-zA-Z]+[0-9][^}]*\}\s*\{[^}]*\}", "", text)


def drop_size_redefs(text):
    """Redefined font sizes send pandoc's macro expansion into a loop."""
    text = re.sub(r"\\(?:re)?newcommand\*?\s*\{?\\" + SIZES + r"\}?\s*(\[[^\]]*\])?\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)
    text = re.sub(r"\\def\s*\\" + SIZES + r"\b\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)
    return text


# \textrm{ + an invisible separator: postprocess/markdown.restore_math_text recognises it in equations
TEXT_MARK = "\\\\textrm{\u2063"

_MACRO_DEF = re.compile(
    r"(?P<head>\\(?:(?:re)?newcommand\*?|providecommand\*?|DeclareRobustCommand\*?)\s*\{?\\(?P<n1>[a-zA-Z@]+)\}?\s*(?:\[\d\])?\s*(?:\[[^\]]*\])?\s*(?=\{))"
    r"|(?P<head2>\\(?:e|g|x)?def\s*\\(?P<n2>[a-zA-Z@]+)\s*(?:#\d)*\s*(?=\{))")

_MATH = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]|\\\(.*?\\\)"
    r"|\\begin\{((?:equation|align|alignat|flalign|gather|multline|eqnarray|displaymath|math)\*?)\}.*?\\end\{\1\}", re.S)


def text_in_macro_bodies(text):
    r"""``\def\method{{\text{UDLM}}}``: outside math pandoc drops ``\text{...}`` (and the macro with it).
    ``\textrm`` is understood in text mode by pandoc (and by KaTeX/MathJax in math), so macros whose body uses
    ``\text`` AND that appear outside math get ``\textrm`` instead. ``restore_math_text`` undoes it in equations."""
    found = []
    for m in _MACRO_DEF.finditer(text):
        end = balanced_end(text, m.end())
        if end < 0 or "\\text" not in text[m.end():end]:
            continue
        found.append((m.start(), m.end(), end, m.group("n1") or m.group("n2")))
    if not found:
        return text
    # text with the definitions and the math regions blanked out: what is left is running text
    prose = text
    for start, _, end, _n in sorted(found, reverse=True):
        prose = prose[:start] + " " * (end - start) + prose[end:]
    prose = _MATH.sub(lambda mm: " " * len(mm.group(0)), prose)
    used = {n for _, _, _, n in found if re.search(r"\\" + re.escape(n) + r"(?![a-zA-Z@])", prose)}
    out, pos = [], 0
    for start, mend, end, n in found:
        if n not in used or start < pos:
            continue
        out.append(text[pos:mend] + re.sub(r"\\text(?![a-zA-Z])\s*\{", TEXT_MARK, text[mend:end]))
        pos = end
    out.append(text[pos:])
    return "".join(out)


_PAIRED = re.compile(r"\\DeclarePairedDelimiter(?P<x>X(?:PP)?)?\*?\s*(?:\{\s*\\(?P<n1>[a-zA-Z]+)\s*\}|\\(?P<n2>[a-zA-Z]+))"
                     r"\s*(?:\[(?P<nargs>\d)\])?\s*(?=\{)")


def _one_line(body):
    """A macro body without the "%" that end its lines in the source."""
    return re.sub(r"%[^\n]*\n\s*", "", body + "\n").strip()


def _glue(left, right):
    """left + right, with a blank when that would otherwise join a control word and a letter (\\lVert + x)."""
    return left + (" " if re.search(r"\\[a-zA-Z]+$", left) and re.match(r"[a-zA-Z]", right) else "") + right


def expand_paired_delimiters(text):
    r"""mathtools ``\DeclarePairedDelimiter\parentheses{(}{)}``: pandoc ignores it and leaves ``\parentheses*{x}``
    in the equations. Drop the declaration and expand the uses: ``\name{x}`` -> ``( x )``, the starred form
    ``\name*{x}`` -> ``\left( x \right)`` and ``\name[\big]{x}`` -> ``\bigl( x \bigr)``.
    ``\DeclarePairedDelimiterX\name[2]{(}{)}{#1\,\delimsize\|\,#2}`` is expanded the same way, with its body
    between the delimiters and ``\delimsize`` set to the size of the form used."""
    defs = {}
    while True:
        m = _PAIRED.search(text)
        if not m:
            break
        ends, i = [], m.end()
        # X: {left}{right}{body}; XPP: {pre}{left}{right}{post}{body}
        for _ in range({None: 2, "X": 3, "XPP": 5}[m.group("x")]):
            e = balanced_end(text, i) if text[i:i + 1] == "{" else -1
            if e < 0:
                break
            ends.append((i, e))
            i = skip_ws(text, e)
        if len(ends) < {None: 2, "X": 3, "XPP": 5}[m.group("x")]:
            text = text[:m.start()] + text[m.end():]
            continue
        parts = [text[a + 1:b - 1] for a, b in ends]
        if m.group("x") == "XPP":
            pre, left, right, post, body = parts
        else:
            pre, post = "", ""
            left, right, body = (parts + [None])[:3]
        defs[m.group("n1") or m.group("n2")] = (left, right, body, int(m.group("nargs") or 1), pre, post)
        text = text[:m.start()] + text[ends[-1][1]:]
    for name, (left, right, body, nargs, pre, post) in defs.items():
        use = re.compile(r"\\" + name + r"(?![a-zA-Z])(\*)?\s*(?:\[\s*(\\[a-zA-Z]+)\s*\])?\s*(?=\{)")
        pos = 0
        while True:
            m = use.search(text, pos)
            if not m:
                break
            args, e = [], m.end()
            for _ in range(nargs):
                j = skip_ws(text, e)
                e2 = balanced_end(text, j) if text[j:j + 1] == "{" else -1
                if e2 < 0:
                    break
                args.append(text[j + 1:e2 - 1])
                e = e2
            if len(args) < nargs:
                pos = m.end()
                continue
            size = m.group(2) or ""
            inner = args[0]
            if body is not None:
                inner = _one_line(body)
                for k, arg in enumerate(args, 1):
                    inner = inner.replace(f"#{k}", arg)
                middle = "\\middle" if m.group(1) else size
                inner = re.sub(r"\\delimsize(?![a-zA-Z])\s*", lambda _m, mid=middle: mid, inner)
            if m.group(1):
                rep = "\\left" + left + " " + inner + " \\right" + right
            elif size:
                rep = size + "l" + left + " " + inner + " " + size + "r" + right
            else:
                rep = _glue(_glue(left, inner), right)
            rep = _glue(_glue(pre, rep), post)
            # "\\norm{x}y": the closing \\rVert must not run into the letter that follows the use
            text = text[:m.start()] + _glue(rep, text[e:e + 1])[:-1 if e < len(text) else None] + text[e:]
            pos = m.start()
        if re.search(r"\\" + name + r"(?![a-zA-Z])", text):
            # still named inside another macro (\newcommand{\kl}{\operatorname{KL}\klxy}): a plain definition
            # pandoc can expand when it expands that macro
            inner = "#1" if body is None else re.sub(r"\\delimsize(?![a-zA-Z])\s*", "", _one_line(body))
            plain = _glue(_glue(pre, _glue(_glue(left, inner), right)), post)
            at = max(text.find("\\begin{document}"), 0)
            text = text[:at] + "\\newcommand{\\" + name + "}[" + str(nargs) + "]{" + plain + "}\n" + text[at:]
    return text


_DELIMITED = re.compile(r"\\(?:e|g|x)?def\s*\\(?P<name>[a-zA-Z]+)\s*#1\s*\\(?P<end>[a-zA-Z]+)\s*(?=\{)")


def expand_delimited_macros(text):
    r"""``\def\bb#1\ee{\begin{align*}#1\end{align*}}`` then ``\bb ... \ee``: pandoc cannot read macros whose
    argument is delimited by a control sequence, so the uses are expanded here and the definition dropped."""
    while True:
        m = _DELIMITED.search(text)
        if not m:
            return text
        e = balanced_end(text, m.end())
        if e < 0:
            return text[:m.start()] + text[m.end():]
        body = text[m.end() + 1:e - 1]
        name, end = m.group("name"), m.group("end")
        text = text[:m.start()] + text[e:]
        use = re.compile(r"\\" + name + r"(?![a-zA-Z])\s*(.*?)\\" + end + r"(?![a-zA-Z])", re.S)
        text = use.sub(lambda mm, body=body: body.replace("#1", mm.group(1)), text)


_XSTRING = re.compile(r"\\(IfBeginWith|IfEndWith|IfSubStr|IfEq|IfStrEq|IfInteger|IfBlankTF)(?![a-zA-Z@])\*?")
_NEWCOMMAND = re.compile(r"\\(?:re)?newcommand\*?\s*\{?\\(?P<name>[a-zA-Z]+)\}?\s*(?:\[(?P<n>\d)\])?\s*(?=\{)")


def _xstring_value(cmd, args):
    a, b = args[0].strip(), args[1].strip()
    if cmd == "IfBeginWith":
        return a.startswith(b)
    if cmd == "IfEndWith":
        return a.endswith(b)
    if cmd == "IfSubStr":
        return b in a
    if cmd in ("IfEq", "IfStrEq"):
        return a == b
    if cmd == "IfInteger":
        return bool(re.fullmatch(r"[+-]?\d+", a))
    return not a      # IfBlankTF


def expand_conditional_macros(text):
    r"""A macro whose body decides with an xstring test (``\newcommand{\valdelta}[2]{#1\IfBeginWith{#2}{-}{red}{green}}``)
    loses its arguments in pandoc, which does not know ``\IfBeginWith``: the delta printed next to every value of a
    results table disappears. Those macros (and the ones that call them) are expanded here at each use, the tests
    are decided on their now literal arguments, and the definitions are dropped."""
    defs = {}
    for m in _NEWCOMMAND.finditer(text):
        end = balanced_end(text, m.end())
        if end > 0:
            defs[m.group("name")] = (int(m.group("n") or 0), text[m.end() + 1:end - 1], m.start(), end)
    target = {n for n, d in defs.items() if _XSTRING.search(d[1])}
    if not target:
        return text
    grew = True
    while grew:       # a macro that calls one of them needs the same treatment
        grew = False
        for n, d in defs.items():
            if n not in target and any(re.search(r"\\" + t + r"(?![a-zA-Z@])", d[1]) for t in target):
                target.add(n)
                grew = True
    for _, _, start, end in sorted((defs[n] for n in target), key=lambda d: d[2], reverse=True):
        text = text[:start] + text[end:]
    use = re.compile(r"\\(" + "|".join(sorted(map(re.escape, target), key=len, reverse=True)) + r")(?![a-zA-Z@])")
    for _ in range(20000):                 # a recursive macro would never end: give up on it
        m = use.search(text)
        if not m:
            break
        nargs, body, _, _ = defs[m.group(1)]
        args, i = [], m.end()
        for _k in range(nargs):
            i = skip_ws(text, i)
            e = balanced_end(text, i) if text[i:i + 1] == "{" else -1
            if e < 0:
                break
            args.append(text[i + 1:e - 1])
            i = e
        if len(args) < nargs:
            text = text[:m.start()] + text[m.end():]
            continue
        expanded = re.sub(r"#(\d)", lambda k, args=args: args[int(k.group(1)) - 1] if int(k.group(1)) <= len(args) else k.group(0), body)
        text = text[:m.start()] + "{" + expanded + "}" + text[i:]
    else:
        return text
    pos = 0
    while True:                             # decide the tests whose arguments are literal
        m = _XSTRING.search(text, pos)
        if not m:
            return text
        groups, i = [], m.end()
        for _k in range(4 if m.group(1) != "IfInteger" and m.group(1) != "IfBlankTF" else 3):
            i = skip_ws(text, i)
            e = balanced_end(text, i) if text[i:i + 1] == "{" else -1
            if e < 0:
                break
            groups.append(text[i + 1:e - 1])
            i = e
        need = 3 if m.group(1) in ("IfInteger", "IfBlankTF") else 4
        if len(groups) < need or any("#" in g for g in groups[:need - 2]) or any("\\" in g for g in groups[:need - 2]):
            pos = m.end()
            continue
        chosen = groups[need - 2] if _xstring_value(m.group(1), groups) else groups[need - 1]
        text = text[:m.start()] + chosen + text[i:]
        pos = m.start()


def drop_newcolumntype(text):
    r"""``\newcolumntype{x}[1]{>{\centering}p{#1pt}}``: only LaTeX's own tabular reads it. The column letters it
    defines are already turned into ``c`` (``centered_column_types``), and pandoc stops at its ``#1``."""
    pos = 0
    head = re.compile(r"\\newcolumntype\*?\s*(?=\{)")
    while True:
        m = head.search(text, pos)
        if not m:
            return text
        end = balanced_end(text, m.end())          # {x}
        i = skip_ws(text, end) if end > 0 else -1
        if i >= 0 and text[i:i + 1] == "[":
            i = skip_ws(text, bracket_end(text, i))
        end = balanced_end(text, i) if i >= 0 and text[i:i + 1] == "{" else -1
        if end < 0:
            pos = m.end()
            continue
        text = text[:m.start()] + text[end:]
        pos = m.start()
