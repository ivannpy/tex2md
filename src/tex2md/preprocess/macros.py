"""User macros, template styles and macro-definition clean-up."""
from __future__ import annotations

import os
import re

from tex2md.util.io import read
from tex2md.util.tex import balanced_end

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
                while i < len(sty) and sty[i] != "{" or (i > 0 and sty[i - 1] == "\\"):
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
    """Primitive spacing with length registers (\\vskip\\bigskipamount) confuses pandoc."""
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


_PAIRED = re.compile(r"\\DeclarePairedDelimiter\*?\s*(?:\{\s*\\(?P<n1>[a-zA-Z]+)\s*\}|\\(?P<n2>[a-zA-Z]+))\s*(?=\{)")


def expand_paired_delimiters(text):
    r"""mathtools ``\DeclarePairedDelimiter\parentheses{(}{)}``: pandoc ignores it and leaves ``\parentheses*{x}``
    in the equations. Drop the declaration and expand the uses: ``\name{x}`` -> ``( x )``, the starred form
    ``\name*{x}`` -> ``\left( x \right)`` and ``\name[\big]{x}`` -> ``\bigl( x \bigr)``."""
    defs = {}
    while True:
        m = _PAIRED.search(text)
        if not m:
            break
        e1 = balanced_end(text, m.end())
        e2 = balanced_end(text, e1) if e1 > 0 and text[e1:e1 + 1] == "{" else -1
        if e2 < 0:
            text = text[:m.start()] + text[m.end():]
            continue
        defs[m.group("n1") or m.group("n2")] = (text[m.end() + 1:e1 - 1], text[e1 + 1:e2 - 1])
        text = text[:m.start()] + text[e2:]
    for name, (left, right) in defs.items():
        use = re.compile(r"\\" + name + r"(?![a-zA-Z])(\*)?\s*(?:\[\s*(\\[a-zA-Z]+)\s*\])?\s*(?=\{)")
        pos = 0
        while True:
            m = use.search(text, pos)
            if not m:
                break
            e = balanced_end(text, m.end())
            if e < 0:
                pos = m.end()
                continue
            arg = text[m.end() + 1:e - 1]
            if m.group(1):
                rep = "\\left" + left + " " + arg + " \\right" + right
            elif m.group(2):
                rep = m.group(2) + "l" + left + " " + arg + " " + m.group(2) + "r" + right
            else:
                rep = left + arg + right
            text = text[:m.start()] + rep + text[e:]
            pos = m.start()
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
