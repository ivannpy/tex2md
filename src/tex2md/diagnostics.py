"""Typed diagnostics: what a conversion could not do well, in a form a human or a script can act on.

Each ``Diagnostic`` has a stable ``code``, a severity (``error`` = content is wrong or missing, ``warning`` =
probably degraded, ``info`` = worth knowing), a count and a few examples. They come from three places:

* ``scan_markdown``: what is left in the finished Markdown (unknown math commands, ``??``, HTML...);
* ``from_pandoc_stderr``: the warnings pandoc prints, which used to be discarded;
* the conversion itself (``dropped-environment`` when pandoc had to be retried, unresolved ``\\ref``/``\\cite``).

The conversion report carries them under ``"diagnostics"``; ``audit`` prints them for a whole library.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from tex2md.postprocess.markdown import math_spans

SEVERITIES = ("error", "warning", "info")

# Commands KaTeX / MathJax understand (anything else inside $...$ will not render in a Markdown viewer).
KNOWN_MATH = frozenset(["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota", "kappa", "varkappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega", "varGamma", "varDelta", "varTheta", "varLambda", "varXi", "varPi", "varSigma", "varUpsilon", "varPhi", "varPsi", "varOmega", "digamma", "frac", "dfrac", "tfrac", "cfrac", "sqrt", "sum", "prod", "coprod", "int", "iint", "iiint", "oint", "lim", "limsup", "liminf", "sup", "inf", "max", "min", "arg", "log", "ln", "lg", "exp", "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth", "det", "dim", "ker", "hom", "deg", "gcd", "Pr", "mod", "bmod", "pmod", "pod", "mathbb", "mathcal", "mathbf", "mathrm", "mathit", "mathsf", "mathtt", "mathfrak", "mathscr", "mathbbm", "mathds", "boldsymbol", "bm", "bf", "rm", "it", "sf", "tt", "cal", "Bbb", "frak", "text", "textbf", "textit", "textrm", "textsf", "texttt", "textnormal", "textup", "textmd", "operatorname", "mbox", "hbox", "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr", "middle", "cdot", "cdots", "ldots", "dots", "vdots", "ddots", "dotsb", "dotsc", "dotsi", "dotsm", "times", "div", "ast", "star", "circ", "bullet", "pm", "mp", "oplus", "ominus", "otimes", "oslash", "odot", "bigoplus", "bigotimes", "bigodot", "bigcup", "bigcap", "bigsqcup", "bigvee", "bigwedge", "biguplus", "leq", "geq", "le", "ge", "leqslant", "geqslant", "neq", "ne", "ll", "gg", "lll", "ggg", "approx", "sim", "simeq", "cong", "equiv", "propto", "asymp", "doteq", "prec", "succ", "preceq", "succeq", "nprec", "nsucc", "lesssim", "gtrsim", "lessim", "ngeq", "nleq", "in", "notin", "ni", "subset", "supset", "subseteq", "supseteq", "subsetneq", "supsetneq", "nsubseteq", "cup", "cap", "setminus", "smallsetminus", "emptyset", "varnothing", "sqcup", "sqcap", "sqsubseteq", "sqsupseteq", "uplus", "forall", "exists", "nexists", "nabla", "partial", "infty", "aleph", "hbar", "ell", "imath", "jmath", "wp", "Re", "Im", "top", "bot", "perp", "parallel", "nparallel", "angle", "measuredangle", "triangle", "triangleq", "triangleleft", "triangleright", "vartriangle", "backslash", "prime", "surd", "to", "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "leftrightarrow", "Leftrightarrow", "mapsto", "longmapsto", "longrightarrow", "longleftarrow", "Longrightarrow", "Longleftarrow", "Longleftrightarrow", "uparrow", "downarrow", "Uparrow", "Downarrow", "updownarrow", "nearrow", "searrow", "nwarrow", "swarrow", "hookrightarrow", "hookleftarrow", "rightharpoonup", "leftharpoonup", "rightleftharpoons", "leadsto", "gets", "implies", "iff", "impliedby", "xrightarrow", "xleftarrow", "xRightarrow", "xLeftarrow", "dashrightarrow", "hat", "widehat", "tilde", "widetilde", "bar", "overline", "underline", "vec", "dot", "ddot", "dddot", "acute", "grave", "breve", "check", "mathring", "overrightarrow", "overleftarrow", "overleftrightarrow", "underbrace", "overbrace", "overset", "underset", "stackrel", "overbracket", "underbracket", "begin", "end", "array", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "cases", "aligned", "align", "alignedat", "gathered", "split", "quad", "qquad", "hspace", "hskip", "kern", "mkern", "mskip", "space", "thinspace", "medspace", "thickspace", "negthinspace", "negmedspace", "negthickspace", "enspace", "nobreakspace", "phantom", "hphantom", "vphantom", "smash", "strut", "mathstrut", "mid", "vert", "Vert", "lvert", "rvert", "lVert", "rVert", "langle", "rangle", "lfloor", "rfloor", "lceil", "rceil", "lbrace", "rbrace", "lbrack", "rbrack", "lgroup", "rgroup", "llbracket", "rrbracket", "mathopen", "mathclose", "mathpunct", "mathinner", "mathrel", "mathbin", "mathop", "mathord", "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits", "nolimits", "nonumber", "notag", "tag", "label", "tilde", "not", "neg", "lnot", "land", "lor", "wedge", "vee", "oplus", "top", "colon", "vcentcolon", "coloneqq", "eqqcolon", "dagger", "ddagger", "dag", "ddag", "color", "textcolor", "colorbox", "fcolorbox", "boxed", "fbox", "cancel", "bcancel", "xcancel", "sout", "overline", "binom", "dbinom", "tbinom", "choose", "atop", "brace", "brack", "over", "sideset", "substack", "Box", "square", "blacksquare", "diamond", "Diamond", "lozenge", "blacklozenge", "bigstar", "clubsuit", "diamondsuit", "heartsuit", "spadesuit", "flat", "natural", "sharp", "checkmark", "maltese", "because", "therefore", "bigtriangleup", "bigtriangledown", "tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge", "intercal", "P", "S", "copyright", "pounds", "textdollar", "textless", "textgreater", "textbackslash", "textunderscore", "Alpha", "Beta", "Epsilon", "Zeta", "Eta", "Iota", "Kappa", "Mu", "Nu", "Omicron", "Rho", "Tau", "Chi", "hline", "hdashline", "cr", "newline"])

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
                              "numbered equation environments with no \\tag{N} (the number could not be recovered)",
                              untagged))

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
    html = Counter(re.findall(r"<(table|div|span|math|embed|figure)\b", body))
    if html:
        out.append(Diagnostic("html-residue", "warning",
                              "raw HTML left in the Markdown (usually a table whose cells hold code blocks)",
                              sum(html.values()), _examples(html)))
    floats = re.findall(r"^\*\*(Figura|Tabla)\.\*\*", body, re.M)
    if floats:
        out.append(Diagnostic("unnumbered-float", "info",
                              "figures/tables with no number: panels, or floats without their own \\caption",
                              len(floats), _examples(Counter(floats))))
    return out


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
