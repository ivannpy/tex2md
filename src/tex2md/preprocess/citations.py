"""Bibliography (.bbl) parsing and citation resolution."""
from __future__ import annotations

import re
import unicodedata

from tex2md.preprocess.symbols import TEXT_SYMBOLS
from tex2md.util.tex import balanced_end, group, skip_ws, unwrap


def parse_bbl(t):
    """natbib/plain .bbl text -> ordered list of entries. None for biblatex."""
    if "\\entry{" in t and "\\bibitem" not in t:
        return None
    items = []
    for m in re.finditer(r"\\bibitem\s*", t):
        i = m.end()
        opt = None
        if i < len(t) and t[i] == "[":
            opt, i = group(t, i, "[", "]")
        i = skip_ws(t, i)
        if i >= len(t) or t[i] != "{":
            continue
        key, i = group(t, i)
        items.append([key.strip(), opt, i])
    out = []
    for n, (key, opt, start) in enumerate(items):
        end = items[n + 1][2] if n + 1 < len(items) else len(t)
        body = t[start:end]
        if n + 1 < len(items):
            body = body[:body.rfind("\\bibitem")]
        body = re.sub(r"\\end\{thebibliography\}.*", "", body, flags=re.S)
        short = year = None
        if opt:
            o = opt.strip()
            if o.startswith("{") and o.endswith("}"):
                o = o[1:-1]
            o = re.sub(r"\\natexlab\{([^}]*)\}", r"\1", o)
            named = _named_label(o)
            mm = re.match(r"(.*?)\(([^()]*)\)", o, re.S)
            if named:
                short, year = named
            elif mm:
                short, year = mm.group(1).strip(), mm.group(2).strip()
            else:
                short = o
        out.append({"key": key, "short": short, "year": year, "num": n + 1, "body": body.strip()})
    return out


def _named_label(label):
    r"""(authors, year) of the labels written by the AAAI/``named`` and old ACL styles:
    ``\protect\citeauthoryear{Ba and Caruana}{2014}`` (or ``{full list}{short}{year}``) and
    ``\protect\citename{Bahdanau \bgroup et al.\egroup }2015``. None for any other label."""
    m = re.match(r"(?:\\protect\s*)?\\(citeauthoryear|citename)\s*(?=\{)", label)
    if not m:
        return None
    groups, i = [], m.end()
    while i < len(label) and label[i] == "{":
        g, i = group(label, i)
        groups.append(g)
    if m.group(1) == "citename":
        authors, year = groups[0], label[i:]
    elif len(groups) >= 2:
        authors, year = groups[-2], groups[-1]
    else:
        return None
    authors = re.sub(r"\\[be]group(?![a-zA-Z])\s*", "", authors).replace("\\mbox.", ".")
    return " ".join(authors.split()), year.strip()


def first_authors(body):
    """"Ilya Sutskever, Oriol Vinyals, and Quoc Le. 2014. Title" -> "Sutskever et al."; None if unparseable."""
    m = re.match(r"(.+?)\.\s+(?:19|20)\d\d[a-z]?\.", plain_bib(body), re.S)
    if not m:
        return None
    people = [x.strip() for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(1)) if x.strip()]
    if not people or len(people) > 40:
        return None
    last = [p.split()[-1] for p in people]
    if len(last) == 1:
        return last[0]
    if len(last) == 2:
        return f"{last[0]} and {last[1]}"
    return f"{last[0]} et al."


def resolve_cites(text, bib, stats):
    by_key = {b["key"]: b for b in bib}
    author_year = any(b["year"] for b in bib)

    def cite(cmd, pre, post, keys):
        entries = []
        for k in [k.strip() for k in keys.split(",") if k.strip()]:
            b = by_key.get(k)
            if not b:
                stats["unresolved_cites"] += 1
                entries.append((k, None))
            else:
                entries.append((k, b))
        if cmd == "newcite":  # ACL/EMNLP style: "Author (year)", i.e. citet
            if author_year:
                cmd = "citet"
            else:
                names = [first_authors(b["body"]) if b else None for _, b in entries]
                if all(names):
                    return "; ".join(f"{n} [{b['num']}]" for n, (_, b) in zip(names, entries))
        if not author_year:
            nums = ", ".join(str(b["num"]) if b else "?" for _, b in entries)
            body = nums + (", " + post if post else "")
            return (pre + " " if pre else "") + "[" + body + "]"
        def a(b, k):
            return plain_bib(b["short"]) if b and b["short"] else k
        def y(b):
            return b["year"] if b and b["year"] else "?"
        if cmd in ("citeauthor", "Citeauthor"):
            return "; ".join(a(b, k) for k, b in entries)
        if cmd == "citeyear":
            return ", ".join(y(b) for _, b in entries)
        if cmd == "citeyearpar":
            return "(" + ", ".join(y(b) for _, b in entries) + (", " + post if post else "") + ")"
        if cmd in ("citet", "Citet", "textcite", "Textcite", "citealt"):
            s = "; ".join(f"{a(b, k)} ({y(b)})" if cmd != "citealt" else f"{a(b, k)} {y(b)}" for k, b in entries)
            return s + (f" ({post})" if post else "")
        inner = "; ".join(f"{a(b, k)}, {y(b)}" for k, b in entries)
        inner = (pre + " " if pre else "") + inner + (", " + post if post else "")
        return inner if cmd in ("citealp", "Citealp", "citenum") else f"({inner})"

    pat = re.compile(r"\\(cite|citep|citet|Citet|Citep|citealp|Citealp|citealt|Citealt|citeauthor|Citeauthor|citefullauthor"
                     r"|citeyear|citeyearpar|citeyearNP|citenum|parencite|textcite|Textcite|autocite|footcite|newcite"
                     r"|shortcite|citeA|citeN|citeNP|citeasnoun)\*?\s*(\[[^\]]*\])?\s*(\[[^\]]*\])?\s*\{([^}]*)\}")
    # other packages' names for the natbib commands (apacite, chicago, harvard, aaai)
    same_as = {"cite": "citep", "parencite": "citep", "autocite": "citep", "Citep": "citep", "citeA": "citet",
               "citeN": "citet", "citeasnoun": "citet", "citeNP": "citealp", "Citealt": "citealt",
               "shortcite": "citeyearpar", "citeyearNP": "citeyear", "citefullauthor": "citeauthor"}

    def sub(m):
        cmd, o1, o2, keys = m.groups()
        o1 = o1[1:-1] if o1 else ""
        o2 = o2[1:-1] if o2 else ""
        pre, post = (o1, o2) if m.group(3) else ("", o1)
        if cmd == "footcite":
            return "\\footnote{" + cite("citealp", pre, post, keys) + ".}"
        if cmd == "citenum" and not author_year:      # the bare number, without brackets
            return cite("citep", pre, post, keys).strip("[]")
        return cite(same_as.get(cmd, cmd), pre, post, keys)
    return pat.sub(sub, text)


def plain_bib(s):
    """Reduce a .bbl entry to plain text; pandoc chokes on many bibliography macros."""
    s = re.sub(r"(?<!\\)%.*", "", s)
    # keep inline math intact ($\Psi$ would otherwise become $$, which opens display math)
    maths = []

    def keep(m):
        maths.append(m.group(0))
        return f"TEXTWOMDMATH{len(maths) - 1}X"
    s = re.sub(r"(?<!\\)\$[^$]+(?<!\\)\$", keep, s)
    s = _plain_bib_text(s)
    return re.sub(r"TEXTWOMDMATH(\d+)X", lambda m: maths[int(m.group(1))], s)


_COMBINING = {"`": "\u0300", "'": "\u0301", "^": "\u0302", "~": "\u0303", '"': "\u0308", "=": "\u0304",
              ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
              "r": "\u030a", "d": "\u0323", "b": "\u0331"}
_LETTERS = {"o": "\u00f8", "O": "\u00d8", "l": "\u0142", "L": "\u0141", "ss": "\u00df", "ae": "\u00e6",
            "AE": "\u00c6", "oe": "\u0153", "OE": "\u0152", "aa": "\u00e5", "AA": "\u00c5", "i": "\u0131",
            "j": "\u0237", "dh": "\u00f0", "DH": "\u00d0", "th": "\u00fe", "TH": "\u00de", "dj": "\u0111",
            "DJ": "\u0110", "ng": "\u014b", "NG": "\u014a"}
_LETTER = re.compile(r"\\(" + "|".join(sorted(_LETTERS, key=len, reverse=True)) + r")(?![a-zA-Z@])(?:\{\}|\s*)")
_BASE = "[a-zA-Z\u0131\u0237]"
# \'e  \'{e}  \v{c}  \v c  (an accent named by a letter needs braces or a blank before its argument)
_ACCENT = re.compile(r"\\(?:([`'^~\"=.])\s*(?:\{\s*(" + _BASE + r")\s*\}|(" + _BASE + r"))"
                     r"|([uvHckrdb])(?:\s*\{\s*(" + _BASE + r")\s*\}|\s+(" + _BASE + r")))")


def tex_accents(s):
    r"""LaTeX accents and special letters as Unicode: ``{\L}ukasz`` -> Łukasz, ``Kydl{\'\i}{\v{c}}ek`` -> Kydlíček.
    Left as commands, the clean-up below strips their braces and what is left is no longer LaTeX (``\vcek``)."""
    s = _LETTER.sub(lambda m: _LETTERS[m.group(1)], s)

    def accent(m):
        mark = _COMBINING[m.group(1) or m.group(4)]
        base = m.group(2) or m.group(3) or m.group(5) or m.group(6)
        base = {"\u0131": "i", "\u0237": "j"}.get(base, base)     # an accent replaces the dot of i and j
        return unicodedata.normalize("NFC", base + mark)
    return _ACCENT.sub(accent, s)


def _plain_bib_text(s):
    s = re.sub(r"\\newblock\b|\\urlprefix\b|\\BibitemOpen\b|\\bibAnnoteFile\{[^}]*\}", " ", s)
    s = re.sub(r"\\BibitemShut\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\penalty\s*-?\d+ ?", "", s)    # "34:\penalty0 17981": the digit is the argument, not text
    for cmd in ("href", "bibinfo", "bibfield", "Eprint"):
        s = unwrap(s, cmd, 1)
    s = re.sub(r"\\(url|doi|path)\s*\{([^}]*)\}", r"\2", s)
    for cmd in ("emph", "textit", "textbf", "textsc", "textrm", "textsf", "texttt", "mbox", "natexlab", "em", "it", "bf", "showarticletitle", "showeprint"):
        s = unwrap(s, cmd, 0)
    s = re.sub(r"\\(em|it|bf|sc|rm|sf|tt)\b\s*", "", s)
    s = tex_accents(s)
    s = re.sub(r"\\(" + "|".join(TEXT_SYMBOLS) + r")(?![a-zA-Z@])(?:\{\}|[ \t]*)", lambda m: TEXT_SYMBOLS[m.group(1)], s)
    # the result goes back into LaTeX: \% and \# stay escaped (a bare % would comment out the rest)
    s = re.sub(r"(?<!\\)~", " ", s).replace("\\&", "&").replace("\\_", "_")
    s = re.sub(r"\\(?![\"'`^~=.ucvHkrbdt]\W?)[a-zA-Z@]+\*?", "", s)
    s = re.sub(r"(?<!\\)[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def bibliography_tex(bib):
    rows = ["\\section*{References}"]
    for b in bib:
        lab = f"{b['short']} ({b['year']})" if b["year"] else f"[{b['num']}]"
        rows.append(f"{plain_bib(lab)}. {plain_bib(b['body'])}")
    return "\n\n".join(rows) + "\n"


PREAMBLE = r"""
\newcommand{\newblock}{}
\newcommand{\natexlab}[1]{#1}
\newcommand{\urlprefix}{}
\newcommand{\bibinfo}[2]{#2}
\newcommand{\bibfield}[2]{#2}
\newcommand{\BibitemOpen}{}
\newcommand{\BibitemShut}[1]{}
\newcommand{\Eprint}[2]{#2}
\newcommand{\doi}[1]{doi:#1}
\newcommand{\eprint}[1]{#1}
"""


def inject_bibliography(text, bib, stats):
    """Replace citations by their labels and append the bibliography as a plain list."""
    text = resolve_cites(text, bib, stats)
    text = re.sub(r"\\bibliography\{[^}]*\}|\\printbibliography(\[[^\]]*\])?", lambda _: "", text)
    text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", text, flags=re.S)
    return text.replace("\\end{document}", bibliography_tex(bib) + "\\end{document}")


def simplify_citeproc(text):
    r"""``\citeproc{key}{{[}12{]}}`` (pandoc's own LaTeX output) -> ``[12]``: the second argument already is the
    citation text of the original document, so use it instead of pandoc's ``[@key]`` placeholder. (The macro's
    definition, ``\begingroup\def\citeproctext...``, also breaks pandoc's table parser inside cells.)"""
    out, pos = [], 0
    for m in re.finditer(r"\\citeproc\s*\{[^{}]*\}\s*(?=\{)", text):
        if m.start() < pos or re.search(r"(?:NewDocumentCommand|newcommand|def)\s*\{?$", text[max(0, m.start() - 30):m.start()]):
            continue   # the definition of the macro itself
        e = balanced_end(text, m.end())
        if e < 0:
            continue
        label = text[m.end() + 1:e - 1]   # keeps the {[}1{]} brackets: a bare [1] is eaten by pandoc here
        out.append(text[pos:m.start()] + label)
        pos = e
    out.append(text[pos:])
    return "".join(out)
