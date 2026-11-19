"""Typed diagnostics: what a conversion could not do well, in a form a human or a script can act on.

Each ``Diagnostic`` has a stable ``code``, a severity (``error`` = content is wrong or missing, ``warning`` =
probably degraded, ``info`` = worth knowing), a count and a few examples. They come from three places:

* ``scan_markdown``: what is left in the finished Markdown (unknown math commands, ``??``, HTML...);
* ``from_pandoc_stderr``: the warnings pandoc prints, which used to be discarded;
* the conversion itself (``dropped-environment`` when pandoc had to be retried, unresolved ``\\ref``/``\\cite``,
  ``lost-text`` when words of the LaTeX given to pandoc are nowhere in the Markdown).

The conversion report carries them under ``"diagnostics"``; ``audit`` prints them for a whole library.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from tex2md.postprocess.markdown import math_spans, protect_code

SEVERITIES = ("error", "warning", "info")

# Commands KaTeX / MathJax understand (anything else inside $...$ will not render in a Markdown viewer).
KNOWN_MATH = frozenset(["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota", "kappa", "varkappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega", "varGamma", "varDelta", "varTheta", "varLambda", "varXi", "varPi", "varSigma", "varUpsilon", "varPhi", "varPsi", "varOmega", "digamma", "frac", "dfrac", "tfrac", "cfrac", "sqrt", "sum", "prod", "coprod", "int", "iint", "iiint", "oint", "lim", "limsup", "liminf", "sup", "inf", "max", "min", "arg", "log", "ln", "lg", "exp", "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth", "det", "dim", "ker", "hom", "deg", "gcd", "Pr", "mod", "bmod", "pmod", "pod", "mathbb", "mathcal", "mathbf", "mathrm", "mathit", "mathsf", "mathtt", "mathfrak", "mathscr", "mathbbm", "mathds", "boldsymbol", "bm", "bf", "rm", "it", "sf", "tt", "cal", "Bbb", "frak", "text", "textbf", "textit", "textrm", "textsf", "texttt", "textnormal", "textup", "textmd", "operatorname", "mbox", "hbox", "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr", "middle", "cdot", "cdots", "ldots", "dots", "vdots", "ddots", "dotsb", "dotsc", "dotsi", "dotsm", "times", "div", "ast", "star", "circ", "bullet", "pm", "mp", "oplus", "ominus", "otimes", "oslash", "odot", "bigoplus", "bigotimes", "bigodot", "bigcup", "bigcap", "bigsqcup", "bigvee", "bigwedge", "biguplus", "leq", "geq", "le", "ge", "leqslant", "geqslant", "neq", "ne", "ll", "gg", "lll", "ggg", "approx", "sim", "simeq", "cong", "equiv", "propto", "asymp", "doteq", "prec", "succ", "preceq", "succeq", "nprec", "nsucc", "lesssim", "gtrsim", "lessim", "ngeq", "nleq", "in", "notin", "ni", "subset", "supset", "subseteq", "supseteq", "subsetneq", "supsetneq", "nsubseteq", "cup", "cap", "setminus", "smallsetminus", "emptyset", "varnothing", "sqcup", "sqcap", "sqsubseteq", "sqsupseteq", "uplus", "forall", "exists", "nexists", "nabla", "partial", "infty", "aleph", "hbar", "ell", "imath", "jmath", "wp", "Re", "Im", "top", "bot", "perp", "parallel", "nparallel", "angle", "measuredangle", "triangle", "triangleq", "triangleleft", "triangleright", "vartriangle", "backslash", "prime", "surd", "to", "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "leftrightarrow", "Leftrightarrow", "mapsto", "longmapsto", "longrightarrow", "longleftarrow", "Longrightarrow", "Longleftarrow", "Longleftrightarrow", "uparrow", "downarrow", "Uparrow", "Downarrow", "updownarrow", "nearrow", "searrow", "nwarrow", "swarrow", "hookrightarrow", "hookleftarrow", "rightharpoonup", "leftharpoonup", "rightleftharpoons", "leadsto", "gets", "implies", "iff", "impliedby", "xrightarrow", "xleftarrow", "xRightarrow", "xLeftarrow", "dashrightarrow", "hat", "widehat", "tilde", "widetilde", "bar", "overline", "underline", "vec", "dot", "ddot", "dddot", "acute", "grave", "breve", "check", "mathring", "overrightarrow", "overleftarrow", "overleftrightarrow", "underbrace", "overbrace", "overset", "underset", "stackrel", "overbracket", "underbracket", "begin", "end", "array", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "cases", "aligned", "align", "alignedat", "gathered", "split", "quad", "qquad", "hspace", "hskip", "kern", "mkern", "mskip", "space", "thinspace", "medspace", "thickspace", "negthinspace", "negmedspace", "negthickspace", "enspace", "nobreakspace", "phantom", "hphantom", "vphantom", "smash", "strut", "mathstrut", "mid", "vert", "Vert", "lvert", "rvert", "lVert", "rVert", "langle", "rangle", "lfloor", "rfloor", "lceil", "rceil", "lbrace", "rbrace", "lbrack", "rbrack", "lgroup", "rgroup", "llbracket", "rrbracket", "mathopen", "mathclose", "mathpunct", "mathinner", "mathrel", "mathbin", "mathop", "mathord", "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits", "nolimits", "nonumber", "notag", "tag", "label", "tilde", "not", "neg", "lnot", "land", "lor", "wedge", "vee", "oplus", "top", "colon", "vcentcolon", "coloneqq", "eqqcolon", "dagger", "ddagger", "dag", "ddag", "color", "textcolor", "colorbox", "fcolorbox", "boxed", "fbox", "cancel", "bcancel", "xcancel", "sout", "overline", "binom", "dbinom", "tbinom", "choose", "atop", "brace", "brack", "over", "sideset", "substack", "Box", "square", "blacksquare", "diamond", "Diamond", "lozenge", "blacklozenge", "bigstar", "clubsuit", "diamondsuit", "heartsuit", "spadesuit", "flat", "natural", "sharp", "checkmark", "maltese", "because", "therefore", "bigtriangleup", "bigtriangledown", "tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge", "intercal", "P", "S", "copyright", "pounds", "textdollar", "textless", "textgreater", "textbackslash", "textunderscore", "Alpha", "Beta", "Epsilon", "Zeta", "Eta", "Iota", "Kappa", "Mu", "Nu", "Omicron", "Rho", "Tau", "Chi", "hline", "hdashline", "cr", "newline", "coloneq", "Coloneq", "Coloneqq", "eqcolon", "Eqcolon", "Eqqcolon", "colonapprox", "Colonapprox", "colonsim", "Colonsim", "lessapprox", "gtrapprox", "preccurlyeq", "succcurlyeq", "curlyeqprec", "curlyeqsucc", "rightsquigarrow", "leftrightsquigarrow", "vcenter", "dotplus", "ltimes", "rtimes", "boxplus", "boxminus", "boxtimes", "boxdot"])

_CMD = re.compile(r"\\([A-Za-z]+)")
_FENCE = re.compile(r"```.*?```", re.S)


@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    count: int = 1
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _examples(counter: Counter, n: int = 5) -> list[str]:
    return [f"{k} x{v}" for k, v in counter.most_common(n)]


def _unbalanced_math(body: str) -> list[str]:
    """Start of each paragraph whose unescaped ``$`` do not pair up (code and HTML tables are not looked at)."""
    code_free, _ = protect_code(body)
    bad = []
    for para in re.split(r"\n\s*\n", code_free):
        if "<td" in para or "<table" in para:
            continue
        if len(re.findall(r"(?<!\\)\$", para)) % 2:
            bad.append(" ".join(para.split())[:100])
    return bad


def scan_markdown(body: str) -> list[Diagnostic]:
    """Problems visible in a finished Markdown body."""
    body = _FENCE.sub(" ", body)
    out: list[Diagnostic] = []

    unknown: Counter = Counter()
    untagged = 0
    for start, end, display in math_spans(body):
        span = body[start:end]
        for m in _CMD.finditer(span):
            if m.group(1) not in KNOWN_MATH:
                unknown["\\" + m.group(1)] += 1
        if display and re.search(r"\\begin\{(?:equation|align|alignat|flalign|gather|multline|eqnarray)\}", span) \
                and "\\tag{" not in span:
            untagged += 1
    if unknown:
        out.append(Diagnostic("unknown-math-macro", "warning",
                              "commands in equations that KaTeX/MathJax do not know (a macro of the author's that "
                              "was not expanded, or unsupported LaTeX)",
                              sum(unknown.values()), _examples(unknown)))
    if untagged:
        out.append(Diagnostic("equation-without-tag", "info",
                              "equation environments with no \\tag{N}: the compiled document gave them no number (autonum, "
                              "\\nonumber) or the number could not be recovered",
                              untagged))

    unbalanced = _unbalanced_math(body)
    if unbalanced:
        out.append(Diagnostic("unbalanced-math", "error",
                              "paragraphs with an odd number of $: a formula lost or gained a delimiter, so the "
                              "text and the math around it are swapped", len(unbalanced), unbalanced[:3]))

    prose = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", body, flags=re.S)
    refs = re.findall(r"[^\n]{0,30}(?<![\w?.!])\?\?(?![\w?!])[^\n]{0,20}", prose)
    if refs:
        out.append(Diagnostic("unresolved-reference", "error",
                              "a \\ref/\\cref that could not be resolved prints as ??", len(refs),
                              [r.strip() for r in refs[:3]]))
    cites = re.findall(r"\[@[^\]\n]+\]", prose)
    if cites:
        out.append(Diagnostic("unresolved-citation", "warning",
                              "citations left as pandoc keys ([@key]): the bibliography could not be matched",
                              len(cites), sorted(set(cites))[:3]))
    # (not text of the paper: an escaped "\<table\>" or a tag inside code)
    code_free = re.sub(r"(?ms)^```.*?^```|^(?: {4}|\t).*?$|`[^`\n]*`", " ", body)
    html = Counter(re.findall(r"(?<!\\)<(table|div|span|math|embed|figure)\b", code_free))
    if html:
        out.append(Diagnostic("html-residue", "warning",
                              "raw HTML left in the Markdown (usually a table whose cells hold code blocks)",
                              sum(html.values()), _examples(html)))
    floats = re.findall(r"^[ \t>]*\*\*(Figura|Tabla)\.\*\*", body, re.M)
    if floats:
        out.append(Diagnostic("unnumbered-float", "info",
                              "figures/tables with no number: panels, or floats without their own \\caption",
                              len(floats), _examples(Counter(floats))))
    return out


# --- text of the LaTeX that never reached the Markdown ------------------------------------------------------

_LOST_RUN = 12      # words in a row, none of whose 4-grams is in the Markdown, before it counts as lost text
_NOT_TEXT = re.compile(
    r"\\begin\{(tikzpicture|axis|pgfpicture|forest|comment|filecontents\*?)\}.*?\\end\{\1\}"
    r"|\\(?:begin|end)\s*\{[^{}]*\}(?:\s*\[[^\]]*\])?"                      # environment name and its options
    r"|\\(?:label|ref|eqref|cref|Cref|autoref|pageref|includegraphics|adjincludegraphics|input|include|nocite"
    r"|bibliographystyle|bibliography|usepackage|vspace|hspace|color|pagestyle|thispagestyle|url|nolinkurl|todo"
    r"|captionsetup|lstset|lstdefinestyle|lstdefinelanguage|tikzset|hypersetup|pgfplotsset|tcbset|newtcolorbox"
    r"|newcommand|renewcommand|providecommand|def|setlength|addtolength|setcounter|addtocounter|definecolor"
    r"|colorlet|fontsize|multicolumn|multirow|textcolor|href|icml[a-zA-Z]*|author|affil|institute|email|address"
    r"|date|title|thanks|addcontentsline|addtocontents)(?![a-zA-Z@])\*?"
    r"(?:\s*\[[^\]]*\])*(?:\s*\{(?:[^{}]|\{[^{}]*\})*\}){0,3}", re.S)


_IF_TOKEN = re.compile(r"\\(if(?!f\b|thenelse|toggle|bool)[a-zA-Z@]*|else|fi)(?![a-zA-Z@])")


def _drop_untaken_branches(tex: str) -> str:
    r"""Remove the branch of every ``\ifNAME ... \else ... \fi`` that the document does not take: ``\iffalse``,
    ``\iftrue`` and the switches it declares with ``\newif`` (pandoc follows them too, so that text is not
    lost: it is hidden). Nested conditionals are matched by depth."""
    names = set(re.findall(r"\\newif\s*\\if([a-zA-Z@]+)", tex)) | {"false", "true"}
    opener = re.compile(r"(?<!\\newif)(?<!\\newif )\\if(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True))
                        + r")(?![a-zA-Z@])")
    pos = 0
    while True:
        m = opener.search(tex, pos)
        if not m:
            return tex
        name = m.group(1)
        if name in ("false", "true"):
            taken = name == "true"
        else:
            sets = re.findall(r"\\" + re.escape(name) + r"(true|false)(?![a-zA-Z@])", tex[:m.start()])
            taken = bool(sets) and sets[-1] == "true"
        depth, middle, close = 1, None, None
        for t in _IF_TOKEN.finditer(tex, m.end()):
            word = t.group(1)
            if word == "fi":
                depth -= 1
                if depth == 0:
                    close = t
                    break
            elif word == "else":
                if depth == 1 and middle is None:
                    middle = t
            else:
                depth += 1
        if close is None:
            pos = m.end()
            continue
        if taken:
            kept = tex[m.end():middle.start() if middle else close.start()]
        else:
            kept = tex[middle.end():close.start()] if middle else ""
        tex = tex[:m.start()] + kept + tex[close.end():]
        pos = m.start()


_EMPTY_MACRO = re.compile(r"\\(?:re)?newcommand\*?\s*\{?\\([a-zA-Z]+)\}?\s*\[(\d)\]\s*\{\s*\}")


def _drop_hidden_arguments(tex: str, text: str) -> str:
    r"""``\newcommand{\eat}[1]{}``: what the document hides with a macro that swallows its arguments is not text of it."""
    from tex2md.util.tex import balanced_end
    for m in _EMPTY_MACRO.finditer(tex):
        name, nargs = m.group(1), int(m.group(2))
        out, pos = [], 0
        for use in re.finditer(r"\\" + name + r"(?![a-zA-Z@])", text):
            if use.start() < pos:
                continue
            end = use.end()
            for _ in range(nargs):
                j = end
                while j < len(text) and text[j] in " \t\n":
                    j += 1
                e = balanced_end(text, j) if text[j:j + 1] == "{" else -1
                if e < 0:
                    break
                end = e
            out.append(text[pos:use.start()])
            pos = end
        out.append(text[pos:])
        text = "".join(out)
    return text


_CONFIGURATION = re.compile(
    r"\\(?:lstset|lstdefinestyle|lstdefinelanguage|tikzset|pgfplotsset|tcbset|hypersetup|newtcolorbox|renewtcolorbox"
    r"|DeclareTCBox|DeclareTCBListing|newenvironment|renewenvironment|NewDocumentCommand|DeclareDocumentCommand"
    r"|NewDocumentEnvironment|newtheorem|DeclareMathOperator|usetikzlibrary|(?:re)?newcommand|providecommand)(?![a-zA-Z@])\*?")


def _drop_configuration(text: str) -> str:
    r"""The arguments of the commands that only configure (``\lstdefinelanguage{L}{ ... {{\color{c}x}}{1} ...}``): they
    nest deeper than the pattern of ``_NOT_TEXT`` follows, so they are skipped by balanced braces."""
    from tex2md.util.tex import balanced_end, bracket_end
    out, pos = [], 0
    for m in _CONFIGURATION.finditer(text):
        if m.start() < pos:
            continue
        end, groups = m.end(), 0
        while groups < 4:
            j = end
            while j < len(text) and text[j] in " \t\n":
                j += 1
            if text[j:j + 1] == "[":
                e = bracket_end(text, j)
            elif text[j:j + 1] == "{":
                e = balanced_end(text, j)
                groups += 1
            else:
                break
            if e < 0:
                break
            end = e
        out.append(text[pos:m.start()])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _words(text: str) -> list[str]:
    import unicodedata
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.findall(r"[a-z]{2,}", text.lower())


def scan_lost_text(tex: str, body: str) -> list[Diagnostic]:
    """Runs of words of the LaTeX given to pandoc that are nowhere in the Markdown: text that a command or an
    environment pandoc does not know took with it. A safety net, so that such a loss is seen, not a proof:
    drawing code, author blocks and text the document itself hides are left out."""
    from tex2md.util.tex import protected_spans
    tex = _drop_untaken_branches(tex)      # (on the whole text: the switches are declared in the preamble)
    # from the abstract on: the title block (authors, affiliations) is not part of the body
    source = tex[max(tex.find("\\begin{abstract}"), tex.find("\\maketitle"), tex.find("\\begin{document}"), 0):]
    plain, pos = [], 0
    for a, b in protected_spans(source):
        plain.append(source[pos:a])
        # verbatim stays (it is text of the paper); math goes (the Markdown writes it differently)
        plain.append(source[a:b] if source.startswith(("\\begin{verbatim", "\\begin{lstlisting", "\\verb"), a) else " ")
        pos = b
    plain.append(source[pos:])
    text = _drop_hidden_arguments(tex, "".join(plain))
    text = _NOT_TEXT.sub(" ", _drop_configuration(text))
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    src = _words(text)
    written, pos = [], 0
    for a, b, _display in math_spans(body):      # (escape-aware: "\\$13,000 ... \\$400,000" is not a formula)
        written.append(body[pos:a])
        pos = b
    out = _words(" ".join(written) + " " + body[pos:])
    grams = {tuple(out[i:i + 4]) for i in range(len(out) - 3)}
    runs, i = [], 0
    while i < len(src) - 3:
        if tuple(src[i:i + 4]) in grams:
            i += 1
            continue
        j = i
        while j < len(src) - 3 and tuple(src[j:j + 4]) not in grams:
            j += 1
        if j - i >= _LOST_RUN:
            runs.append(src[i:j + 3])
        i = j
    if not runs:
        return []
    runs.sort(key=len, reverse=True)
    return [Diagnostic("lost-text", "warning",
                       f"text of the LaTeX source that is not in the Markdown (words in runs of {_LOST_RUN} or more): "
                       "a command or environment pandoc does not know may have taken it",
                       sum(len(r) for r in runs),
                       [f"{len(r)} words: {' '.join(r[:14])}..." for r in runs[:3]])]


_PANDOC_WARNING = re.compile(r"^\[WARNING\]\s*(.*)$", re.M)


def from_pandoc_stderr(err: str) -> list[Diagnostic]:
    """pandoc's own warnings (``[WARNING] Could not convert TeX math ...``), grouped by kind."""
    kinds: Counter = Counter()
    sample: dict[str, str] = {}
    for m in _PANDOC_WARNING.finditer(err or ""):
        text = m.group(1).strip()
        key = re.sub(r"[`'\"].*", "", text)[:60].strip() or text[:60]
        kinds[key] += 1
        sample.setdefault(key, text[:140])
    return [Diagnostic("pandoc-warning", "warning", "pandoc warning: " + key, n, [sample[key]])
            for key, n in kinds.most_common()]


def counts(diags: list[Diagnostic]) -> dict[str, int]:
    c = Counter(d.severity for d in diags)
    return {s: c.get(s, 0) for s in SEVERITIES}


_HEADING_NUM = re.compile(r"^#{1,6}\s+([0-9A-Z]+(?:[.-][0-9A-Z]+)*)\.?\s", re.M)
MISSING_SECTIONS_MIN = 2         # sections of the .aux with no heading before the conversion is called incomplete
MISSING_SECTIONS_SHARE = 0.2     # ... and the share of all of them that this must be


def scan_missing_sections(sections: list[dict], body: str) -> list[Diagnostic]:
    """Numbered sections of the compiled document (its .aux) with no heading of that number in the Markdown. Half a
    paper loaded with a command latexpand does not know (``\\subfile``) used to give a short Markdown with the
    report in ``ok``; a heading written without its number (or with another one) is caught too."""
    nums = [s["num"] for s in sections if s["level"] <= 2]
    if not nums:
        return []
    written = set(_HEADING_NUM.findall(body))
    missing = [n for n in dict.fromkeys(nums) if n not in written]
    if len(missing) < MISSING_SECTIONS_MIN or len(missing) < MISSING_SECTIONS_SHARE * len(set(nums)):
        return []
    return [Diagnostic("missing-sections", "error",
                       f"{len(missing)} of the {len(set(nums))} numbered sections of the compiled document have no "
                       "heading in the Markdown: part of the body was not converted, or the headings lost their numbers",
                       len(missing), missing[:8])]


def main_file_diagnostics(main: str, candidates: list[str]) -> list[Diagnostic]:
    """Several files of the e-print are complete documents and arXiv's ``00README`` does not say which one is the
    paper: the one taken was chosen by a heuristic (a .bbl of the same name, how many files it pulls in)."""
    if len(candidates) < 2:
        return []
    return [Diagnostic("ambiguous-main-file", "warning",
                       f"{len(candidates)} files have \\documentclass and \\begin{{document}} and the source declares "
                       f"no main file: {main} was taken (check that it is the paper and not a rebuttal or a draft)",
                       len(candidates), candidates[:5])]
