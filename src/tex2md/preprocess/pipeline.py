"""The ordered LaTeX preprocessing pipeline that feeds pandoc."""
from __future__ import annotations

import re

from tex2md.preprocess.algorithms import rewrite_algorithm_captions, rewrite_algorithmic
from tex2md.preprocess.citations import PREAMBLE, bibliography_tex, resolve_cites
from tex2md.preprocess.floats import flatten_subfloats, retype_floats, split_captioned_blocks, tables_without_tabular, unnest_tables, wrap_loose_captions
from tex2md.preprocess.macros import SIZES, drop_macro_defs, fix_stray_macro_braces
from tex2md.preprocess.references import expand_wrappers, resolve_refs
from tex2md.preprocess.tables import normalize_tabulars, simplify_multicolumn_specs
from tex2md.util.tex import balanced_end, group, skip_ws, unwrap

def preprocess(text, labels, bib, stats):
    body_at = text.find("\\begin{document}")
    m = re.compile(r"\\appendix(?![a-zA-Z@])").search(text, max(body_at, 0))
    if m:
        text = text[:m.start()] + "\n\nTEX2MDAPPENDIXMARKER\n\n" + text[m.end():]
    text = re.sub(r"\\(?:DeclareRobustCommand|def|(?:re)?newcommand)\*?\s*\{?\\onedot\}?", r"\\def\\tex2mdunused", text)
    text = re.sub(r"\\onedot(?![a-zA-Z])\s*\.?", ".", text)
    # cvpr.sty-style abbreviations (\eg, \ie, ...) live in the style file, which pandoc
    # never sees, so it drops them; expand the ones the document does not define itself
    for name, lit in (("eg", "e.g."), ("Eg", "E.g."), ("ie", "i.e."), ("Ie", "I.e."), ("cf", "cf."),
                      ("Cf", "Cf."), ("etc", "etc."), ("vs", "vs."), ("wrt", "w.r.t."),
                      ("etal", "et al.")):
        if not re.search(r"\\(?:def|(?:re)?newcommand|DeclareRobustCommand)\*?\s*\{?\\" + name + r"(?![a-zA-Z@])", text):
            text = re.sub(r"\\" + name + r"(?![a-zA-Z@])(?:\s*\{\})?[ \t]*", lit + " ", text)
    text = fix_stray_macro_braces(text)
    text = re.sub(r"\\begin\{(wraptable|wrapfigure)\}(\{[^}]*\}|\[[^\]]*\])*", lambda m: "\\begin{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    text = re.sub(r"\\end\{(wraptable|wrapfigure)\}", lambda m: "\\end{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    text = tables_without_tabular(text)
    text = retype_floats(text)
    text = wrap_loose_captions(text)
    text = split_captioned_blocks(text)
    text = re.sub(r"\\(?:begin|end)\{minipage\}(?:\[[^\]]*\])?(?:\s*\{[^{}]*\})?", "", text)
    text = re.sub(r"\\parbox\s*(?:\[[^\]]*\])*\s*\{[^{}]*\}\s*(?=\{)", "", text)  # \\parbox{w}{...} -> {...}
    text = re.sub(r"\\(begin|end)\{SC(figure|table)\}(?:\s*\[[^\]]*\])*", r"\\\1{\2}", text)
    text = flatten_subfloats(text)
    text = unnest_tables(text)
    # comment-only lines can sit inside optional arguments (\includegraphics[..., %\n ...]) and trip pandoc
    text = re.sub(r"(?m)^[ \t]*%.*\n", "", text)
    # preamble \makeatletter blocks that redefine \maketitle (TeX internals like \@topnum) break pandoc
    pre_end = text.find("\\begin{document}")
    if pre_end > 0:
        pre = re.sub(r"\\makeatletter(?:(?!\\makeatother).)*?\\@maketitle(?:(?!\\makeatother).)*?\\makeatother",
                     "", text[:pre_end], flags=re.S)
        text = pre + text[pre_end:]
    # macros that typeset with TeX primitives pandoc cannot expand
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
    # primitive spacing with length registers (\vskip\bigskipamount) confuses pandoc
    text = re.sub(r"\\[vh]skip\s*-?\s*(?:\\[a-zA-Z@]+|[0-9.]+\s*[a-z]{2})(?:\s*(?:plus|minus)\s*-?[0-9.]*\s*(?:\\[a-zA-Z@]+|[a-z]{2}))*", "", text)
    # operator names TeX tolerates but pandoc rejects (\expm1)
    text = re.sub(r"\\DeclareMathOperator\*?\s*\{\\[a-zA-Z]+[0-9][^}]*\}\s*\{[^}]*\}", "", text)
    # redefined font sizes send pandoc's macro expansion into a loop
    text = re.sub(r"\\(?:re)?newcommand\*?\s*\{?\\" + SIZES + r"\}?\s*(\[[^\]]*\])?\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)
    text = re.sub(r"\\def\s*\\" + SIZES + r"\b\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)
    text = re.sub(r"\\label\s*\[[^\]]*\]\s*\{", r"\\label{", text)  # cleveref \label[type]{...}
    text = simplify_multicolumn_specs(text)
    # tcolorbox definitions (\newtcolorbox{name}[n][default]{...#1...}) break pandoc. Code listings
    # become verbatim; ordinary boxes become a quote with the title (if any) in bold.
    for m in reversed(list(re.finditer(r"\\(newtcblisting|newtcolorbox|DeclareTColorBox|NewTColorBox|NewTCBListing)\s*\{([^}]*)\}", text))):
        i = m.end()
        specs = []
        while True:
            i = skip_ws(text, i)
            if i < len(text) and text[i] in "[{":
                g, i = group(text, i, text[i], "]" if text[i] == "[" else "}")
                specs.append(g)
            else:
                break
        name = re.escape(m.group(2))
        is_listing = "listing" in m.group(1).lower()
        text = text[:m.start()] + text[i:]
        if is_listing:
            text = re.sub(r"\\begin\{" + name + r"\}(\[[^\]]*\]|\{[^}]*\})*", r"\\begin{verbatim}", text)
            text = re.sub(r"\\end\{" + name + r"\}", r"\\end{verbatim}", text)
            continue
        nargs = int(specs[0]) if specs and specs[0].isdigit() else 0
        def box(mm, nargs=nargs):
            j, titles = mm.end(), []
            for k in range(nargs):
                j2 = skip_ws(mm.string, j)
                if j2 < len(mm.string) and mm.string[j2] == "[":
                    _, j = group(mm.string, j2, "[", "]")
                elif j2 < len(mm.string) and mm.string[j2] == "{":
                    g, j = group(mm.string, j2, "{", "}")
                    titles.append(g)
                else:
                    break
            return "\\begin{quote}" + ("\\textbf{" + titles[0] + "}\n\n" if titles and titles[0].strip() else ""), j
        out, pos = [], 0
        for mm in re.finditer(r"\\begin\{" + name + r"\}", text):
            if mm.start() < pos:
                continue
            rep, j = box(mm)
            out.append(text[pos:mm.start()] + rep)
            pos = j
        out.append(text[pos:])
        text = "".join(out)
        text = re.sub(r"\\end\{" + name + r"\}", r"\\end{quote}", text)
    text = expand_wrappers(text)
    text = resolve_refs(text, labels, stats)
    if bib:
        text = resolve_cites(text, bib, stats)
        text = re.sub(r"\\bibliography\{[^}]*\}|\\printbibliography(\[[^\]]*\])?", lambda _: "", text)
        text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", text, flags=re.S)
        text = text.replace("\\end{document}", bibliography_tex(bib) + "\\end{document}")
    # environments pandoc does not know: treat as their plain counterparts
    text = re.sub(r"\\begin\{(wraptable|wrapfigure)\}(\{[^}]*\}|\[[^\]]*\])*", lambda m: "\\begin{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    text = re.sub(r"\\(begin|end)\{(table|figure)\*\}", r"\\\1{\2}", text)
    text = re.sub(r"\\end\{(wraptable|wrapfigure)\}", lambda m: "\\end{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    # only adjustbox takes an argument: "\begin{center}{\fontsize..." must keep its brace
    text = re.sub(r"\\begin\{adjustbox\}(\{[^}]*\})?|\\begin\{(center|small|footnotesize|scriptsize|tiny)\}", "", text)
    text = re.sub(r"\\fontsize\s*\{[^}]*\}\s*\{[^}]*\}(\s*\\selectfont)?", "", text)
    text = re.sub(r"\\end\{(adjustbox|center|small|footnotesize|scriptsize|tiny)\}", "", text)
    # tabularx/tabulary/tabular* take a width before the column spec; NiceTabular does not
    text = re.sub(r"\\begin\{(tabularx|tabulary|tabular\*)\}\s*\{(?:[^{}]|\{[^{}]*\})*\}", r"\\begin{tabular}", text)
    text = re.sub(r"\\begin\{NiceTabular\*?\}", r"\\begin{tabular}", text)
    text = re.sub(r"\\end\{(NiceTabular\*?|tabularx|tabulary|tabular\*)\}", r"\\end{tabular}", text)
    text = normalize_tabulars(text)
    text = rewrite_algorithmic(text)
    text = rewrite_algorithm_captions(text, labels)
    # an empty "$$" cell opens a display equation in pandoc and swallows the whole tabular
    text = re.sub(r"(&[ \t]*)\$\$[ \t]*(?=&|\\\\)", r"\1", text)
    text = re.sub(r"\\CodeBefore\b.*?\\Body\b", "", text, flags=re.S)
    # siunitx S columns and X columns -> centered columns
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
    text = re.sub(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", colspec, text)
    def oneline(body):
        return re.sub(r"\\\\(\[[^\]]*\])?", " ", body)
    for cmd, n in (("resizebox", 2), ("scalebox", 1), ("adjustbox", 1), ("makebox", 0), ("num", 0)):
        text = unwrap(text, cmd, n)
    for cmd in ("makecell", "shortstack", "thead"):
        text = unwrap(text, cmd, 0, oneline)
    text = unwrap(text, "multirow", 2, oneline)
    text = re.sub(r"\\captionof\{(table|figure)\}", r"\\caption", text)
    text = re.sub(r"\\bm\b", r"\\boldsymbol", text)
    return PREAMBLE + text
