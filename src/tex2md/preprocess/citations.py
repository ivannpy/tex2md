"""Bibliography (.bbl) parsing and citation resolution."""
from __future__ import annotations

import re

from tex2md.util.tex import group, skip_ws, unwrap

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
            mm = re.match(r"(.*?)\(([^()]*)\)", o, re.S)
            if mm:
                short, year = mm.group(1).strip(), mm.group(2).strip()
            else:
                short = o
        out.append({"key": key, "short": short, "year": year, "num": n + 1, "body": body.strip()})
    return out


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
        if cmd in ("citet", "Citet", "textcite", "Textcite", "citealt"):
            s = "; ".join(f"{a(b, k)} ({y(b)})" if cmd != "citealt" else f"{a(b, k)} {y(b)}" for k, b in entries)
            return s + (f" ({post})" if post else "")
        inner = "; ".join(f"{a(b, k)}, {y(b)}" for k, b in entries)
        inner = (pre + " " if pre else "") + inner + (", " + post if post else "")
        return inner if cmd in ("citealp", "Citealp") else f"({inner})"

    pat = re.compile(r"\\(cite|citep|citet|Citet|Citep|citealp|Citealp|citealt|citeauthor|Citeauthor|citeyear|parencite|textcite|Textcite|autocite|newcite)\*?\s*(\[[^\]]*\])?\s*(\[[^\]]*\])?\s*\{([^}]*)\}")

    def sub(m):
        cmd, o1, o2, keys = m.groups()
        o1 = o1[1:-1] if o1 else ""
        o2 = o2[1:-1] if o2 else ""
        pre, post = (o1, o2) if m.group(3) else ("", o1)
        cmd = {"cite": "citep", "parencite": "citep", "autocite": "citep", "Citep": "citep"}.get(cmd, cmd)
        return cite(cmd, pre, post, keys)
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


def _plain_bib_text(s):
    s = re.sub(r"\\newblock\b|\\urlprefix\b|\\BibitemOpen\b|\\bibAnnoteFile\{[^}]*\}", " ", s)
    s = re.sub(r"\\BibitemShut\s*\{[^}]*\}", "", s)
    for cmd in ("href", "bibinfo", "bibfield", "Eprint"):
        s = unwrap(s, cmd, 1)
    s = re.sub(r"\\(url|doi|path)\s*\{([^}]*)\}", r"\2", s)
    for cmd in ("emph", "textit", "textbf", "textsc", "textrm", "textsf", "texttt", "mbox", "natexlab", "em", "it", "bf", "showarticletitle", "showeprint"):
        s = unwrap(s, cmd, 0)
    s = re.sub(r"\\(em|it|bf|sc|rm|sf|tt)\b\s*", "", s)
    # the result goes back into LaTeX: \% and \# stay escaped (a bare % would comment out the rest)
    s = s.replace("~", " ").replace("\\&", "&").replace("\\_", "_")
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
