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
            use = r"\\" + name + r"(?![a-zA-Z@])" + (r"(?:\[[^\]]*\])?" if opt else "") + r"(?:\s*\{[^{}]*\}){0,%d}" % (n - 1 if opt else n)
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
    pandoc never opens them, so their macros stay unexpanded in the math. Returns definitions only."""
    out = []
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
                    used = re.search(r"\\" + name + r"(?![A-Za-z@])", doc_body)
                    if (used and name not in PROTECTED_MACROS
                            and not re.search(r"##|\\(?:def|gdef|let|if[a-zA-Z]*|expandafter|csname|hss|llap|rlap|ifx)\b", sty[m.end():j])):
                        out.append(sty[m.start():j])
                    pos = j
                else:
                    pos = m.end()
    return "\n".join(out)
