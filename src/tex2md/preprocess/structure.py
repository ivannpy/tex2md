"""Document-level rewrites: appendix marker, author abbreviations, comments, title hacks."""
from __future__ import annotations

import re


def mark_appendix(text: str) -> str:
    """\\appendix becomes a marker paragraph; the pandoc filter then numbers sections A, B, ..."""
    body_at = text.find("\\begin{document}")
    m = re.compile(r"\\appendix(?![a-zA-Z@])").search(text, max(body_at, 0))
    if m:
        text = text[:m.start()] + "\n\nTEX2MDAPPENDIXMARKER\n\n" + text[m.end():]
    return text


def expand_abbreviations(text: str) -> str:
    """\\onedot and the cvpr.sty-style abbreviations (\\eg, \\ie, ...)."""
    text = re.sub(r"\\(?:DeclareRobustCommand|def|(?:re)?newcommand)\*?\s*\{?\\onedot\}?", r"\\def\\tex2mdunused", text)
    text = re.sub(r"\\onedot(?![a-zA-Z])\s*\.?", ".", text)
    # cvpr.sty-style abbreviations (\eg, \ie, ...) live in the style file, which pandoc
    # never sees, so it drops them; expand the ones the document does not define itself
    for name, lit in (("eg", "e.g."), ("Eg", "E.g."), ("ie", "i.e."), ("Ie", "I.e."), ("cf", "cf."),
                      ("Cf", "Cf."), ("etc", "etc."), ("vs", "vs."), ("wrt", "w.r.t."),
                      ("etal", "et al.")):
        if not re.search(r"\\(?:def|(?:re)?newcommand|DeclareRobustCommand)\*?\s*\{?\\" + name + r"(?![a-zA-Z@])", text):
            text = re.sub(r"\\" + name + r"(?![a-zA-Z@])(?:\s*\{\})?[ \t]*", lit + " ", text)
    return text


def strip_comment_lines(text: str) -> str:
    """Comment-only lines can sit inside optional arguments (\\includegraphics[..., %\\n ...]) and trip pandoc."""
    return re.sub(r"(?m)^[ \t]*%.*\n", "", text)


def drop_title_hacks(text: str) -> str:
    """Preamble \\makeatletter blocks that redefine \\maketitle (TeX internals like \\@topnum) break pandoc."""
    pre_end = text.find("\\begin{document}")
    if pre_end > 0:
        pre = re.sub(r"\\makeatletter(?:(?!\\makeatother).)*?\\@maketitle(?:(?!\\makeatother).)*?\\makeatother",
                     "", text[:pre_end], flags=re.S)
        text = pre + text[pre_end:]
    return text


def label_options(text: str) -> str:
    """cleveref \\label[type]{...} -> \\label{...}."""
    return re.sub(r"\\label\s*\[[^\]]*\]\s*\{", r"\\label{", text)
