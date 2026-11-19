r"""Constructs whose text pandoc drops without a word.

pandoc gives every command it does not know the brace groups that follow it and throws them away with it:
``\noindent{\bf Proof.}``, ``\cellcolor{gray!20}{58.6}``, ``\fbox{...}``, ``\color{red}{text}`` all lose
their text. The rules here remove the layout command, or unwrap the box, so the text reaches the reader.
Math and verbatim are left alone (there the commands are either meaningful or literal).
"""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end, bracket_end, group, rewrite_outside_math, skip_optional, skip_ws

# Declarations without arguments: a "{" right after them is ordinary text, not an argument
_NO_ARGUMENT = (
    "noindent|indent|centering|raggedleft|raggedright|smallskip|medskip|bigskip|newpage|clearpage|"
    "cleardoublepage|pagebreak|nopagebreak|linebreak|nolinebreak|allowbreak|nobreak|break|medbreak|"
    "smallbreak|bigbreak|goodbreak|sloppy|fussy|leavevmode|null|ignorespaces|unskip|relax|protect|samepage|"
    "flushbottom|raggedbottom|onecolumn|FloatBarrier|justifying|RaggedRight|RaggedLeft|Centering|"
    "singlespacing|onehalfspacing|doublespacing|frenchspacing|nonfrenchspacing|boldmath|unboldmath|sffamily|"
    "rmfamily|upshape|mdseries|normalfont|selectfont|sc|sf|sl|hss|hfil|vfil|strut|obeylines|obeyspaces|"
    "bgroup|egroup|begingroup|endgroup")
_SPACE = "quad|qquad|enspace|enskip|thinspace"
_ONE_ARGUMENT = "vspace|hspace|phantom|hphantom|vphantom|thispagestyle|pagestyle|linespread|color"
_TWO_ARGUMENTS = "setlength|addtolength|setcounter|addtocounter"

_DEFINITION_HEAD = re.compile(r"\\(?:(?:re)?newcommand|providecommand|DeclareRobustCommand|[egx]?def|let)\*?\s*\{?\s*(?:\\[a-zA-Z@]+\s*=?\s*)?$")


def _being_defined(text, start):
    r"""True when the command at ``start`` is the name in ``\renewcommand\centering{...}`` or ``\let\old\centering``."""
    return bool(_DEFINITION_HEAD.search(text, max(0, start - 60), start))


_DETACH = re.compile(
    r"\\(?:(?P<plain>" + _NO_ARGUMENT + r")|(?P<space>" + _SPACE + r")"
    r"|(?P<one>" + _ONE_ARGUMENT + r")|(?P<two>" + _TWO_ARGUMENTS + r"))(?![a-zA-Z@])\*?")


def detach_swallowed_groups(text: str) -> str:
    r"""``\noindent{\bf Proof.}``, ``\vspace{1em}{text}``, ``\color{red}{text}``: drop the command (and its own
    arguments) when a brace group follows, so pandoc does not take that group for an argument and lose it."""
    def handler(m, t):
        if _being_defined(t, m.start()):
            return None
        i = m.end()
        for _ in range(1 if m.group("one") else 2 if m.group("two") else 0):
            i = skip_optional(t, i)
            if i >= len(t) or t[i] != "{":
                return None
            i = balanced_end(t, i)
            if i < 0:
                return None
        j = skip_ws(t, i)
        if j >= len(t) or t[j] != "{" or "\n\n" in t[i:j]:
            return None
        if m.group("space") or m.group(0).startswith("\\hspace"):
            return "~", j
        return "", j
    return rewrite_outside_math(text, _DETACH, handler)


# box -> number of arguments before the content (optional arguments may sit anywhere before it)
_BOXES = {"centerline": 0, "leftline": 0, "rightline": 0,     # (pandoc stops at a \par inside them)
          "fbox": 0, "framebox": 0, "makebox": 0, "raisebox": 1, "rotatebox": 1, "smash": 0, "llap": 0,
          "rlap": 0, "clap": 0, "textls": 0, "tcbox": 0, "ovalbox": 0, "Ovalbox": 0, "shadowbox": 0,
          "doublebox": 0, "fcolorbox": 2, "textquote": 0}
# unknown to pandoc, with a close equivalent it does know
_RENAMED = {"uwave": "underline", "uuline": "underline", "dashuline": "underline", "dotuline": "underline",
            "xout": "sout", "textproc": "textsc"}
_BOX = re.compile(r"\\(" + "|".join(list(_BOXES) + list(_RENAMED)) + r")(?![a-zA-Z@])\*?")


def unwrap_text_boxes(text: str) -> str:
    r"""``\fbox{X}``, ``\framebox[w]{X}``, ``\raisebox{2pt}{X}``, ``\rotatebox{90}{X}``... -> ``{X}``.
    pandoc does not know these boxes and drops what they hold (sample texts, rotated row labels)."""
    def handler(m, t):
        name = m.group(1)
        if _being_defined(t, m.start()):
            return None
        i = m.end()
        for _ in range(_BOXES.get(name, 0)):
            i = skip_optional(t, i)
            if i >= len(t) or t[i] != "{":
                return None
            i = balanced_end(t, i)
            if i < 0:
                return None
        i = skip_optional(t, i)
        if i >= len(t) or t[i] != "{":
            return None
        e = balanced_end(t, i)
        if e < 0:
            return None
        if name in _RENAMED:
            return "\\" + _RENAMED[name] + t[i:e], e
        if name == "textquote":
            return "``" + t[i + 1:e - 1] + "''", e
        return t[i:e], e
    return rewrite_outside_math(text, _BOX, handler)


_TEXT_CMD = re.compile(r"\\text(?![a-zA-Z@])\s*(?=\{)")


def text_outside_math(text: str) -> str:
    r"""``\text{// a comment}`` in running text (it belongs in math, but authors use it in algorithms and table
    cells): pandoc drops it there with what it holds. Only the body is touched: in a macro definition ``\text``
    is usually meant for math (``text-in-macro-bodies`` deals with those)."""
    body_at = text.find("\\begin{document}")
    if body_at < 0:
        return text

    def handler(m, t):
        if m.start() < body_at or _DEFINES_ON_LINE.search(t, t.rfind("\n", 0, m.start()) + 1, m.start()):
            return None
        e = balanced_end(t, m.end())
        return (t[m.end():e], e) if e > 0 else None
    return rewrite_outside_math(text, _TEXT_CMD, handler)


_DEFINES_ON_LINE = re.compile(r"\\(?:(?:re)?newcommand|providecommand|DeclareRobustCommand|[egx]?def|DeclareMathOperator)\b")


def table_rules_and_colors(text: str) -> str:
    r"""Rules and colours of a tabular that pandoc misreads.

    * ``\cmidrule(lr){2-3}`` / ``\cline{2-3}``: pandoc takes the command for a rule and prints its range as the
      text of the next cell ("2-3 Model"); they become ``\midrule`` / ``\hline``.
    * ``\rowcolor``, ``\cellcolor``, ``\addlinespace``, ``\specialrule``...: anything in front of a
      ``\multicolumn`` makes pandoc drop that cell, and ``\cellcolor{c}{58.6}`` loses the number.
    """
    # a rule of the author's own made of primitives (\newcommand{\thickhline}{\noalign{...}\hrule height 1pt ...}):
    # pandoc prints "height 1pt" in the next cell, and the cell turns the table into HTML
    head = re.compile(r"\\(?:re)?newcommand\*?\s*\{?\\([a-zA-Z]+)\}?\s*(?=\{)")
    names, pos = set(), 0
    while True:             # (searched again after each removal: a macro may be defined twice)
        m = head.search(text, pos)
        if not m:
            break
        body, end = group(text, m.end())
        if re.search(r"\\noalign|\\hrule", body) and "#" not in body:
            names.add(m.group(1))
            text = text[:m.start()] + text[end:]
            pos = m.start()
        else:
            pos = m.end()
    for name in names:
        text = re.sub(r"\\" + name + r"(?![a-zA-Z@])", r"\\hline", text)
    text = re.sub(r"\\cmidrule\s*(?:\[[^\]]*\])?\s*(?:\([^)]*\))?\s*\{[^{}]*\}", r"\\midrule ", text)
    text = re.sub(r"\\cline\s*\{[^{}]*\}", r"\\hline ", text)
    text = re.sub(r"\\specialrule\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{[^{}]*\}", r"\\midrule ", text)
    text = re.sub(r"\\(?:Xhline\s*\{[^{}]*\}|hhline\s*\{[^{}]*\}|cdashline\s*\{[^{}]*\}(?:\[[^\]]*\])?"
                  r"|hdashline(?![a-zA-Z])(?:\[[^\]]*\])?|(?:thickhline|firsthline|lasthline)(?![a-zA-Z]))",
                  r"\\hline ", text)
    text = re.sub(r"\\(?:rowcolor|cellcolor|arrayrulecolor)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\rowcolors\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:addlinespace(?![a-zA-Z])\s*(?:\[[^\]]*\])?|bigstrut(?![a-zA-Z])(?:\[[^\]]*\])?"
                  r"|morecmidrules(?![a-zA-Z]))", "", text)
    text = re.sub(r"\\noalign\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)
    # \rule{0pt}{2.6ex}: an invisible strut
    return re.sub(r"\\rule\s*(?:\[[^\]]*\])?\s*\{\s*0(?:\.0*)?\s*(?:pt|em|ex|cm|mm|in)?\s*\}\s*\{[^{}]*\}", "", text)


_RULE = re.compile(r"\\rule(?![a-zA-Z@])\s*(\[[^\]]*\])?\s*\{([^{}]*)\}\s*\{[^{}]*\}")


def drop_drawn_rules(text: str) -> str:
    r"""``\rule[.5ex]{1em}{.2em}`` in running text (the coloured line of a legend): there is nothing to read in
    it, and pandoc stops with an error at the optional argument. A rule as wide as the text
    (``\rule{\linewidth}{1pt}``) is a separator and stays."""
    def handler(m, _t):
        if not m.group(1) and re.search(r"width|\\hsize", m.group(2)):
            return None
        return "", m.end()
    return rewrite_outside_math(text, _RULE, handler, passes=1)


_NOTE = re.compile(r"\\(?P<kind>footnotemark|footnotetext|tablefootnote)(?![a-zA-Z@])\s*(?P<opt>\[[^\]]*\])?")


def footnotetext_to_footnote(text: str) -> str:
    r"""``\footnotemark`` ... ``\footnotetext{X}`` -> ``\footnote{X}`` at the mark. pandoc pairs only the
    simplest case and drops the note otherwise (numbered marks, a ``\footnotetext`` without mark). Marks of the
    title block are not used: the authors are not part of the body, and the note would go with them."""
    title_end = max(text.find("\\maketitle"), text.find("\\begin{document}"))
    marks, edits = [], []   # unpaired marks: (start, end, opt); edits: (start, end, replacement)

    def scan(m, t):
        kind, opt = m.group("kind"), m.group("opt")
        if kind == "footnotemark":
            if m.start() > title_end:
                marks.append((m.start(), m.end() + (2 if t[m.end():m.end() + 2] == "{}" else 0), opt))
            return None
        i = skip_ws(t, m.end())
        if i >= len(t) or t[i] != "{":
            return None
        e = balanced_end(t, i)
        if e < 0:
            return None
        note = "\\footnote" + t[i:e]
        if kind == "footnotetext":
            # the mark with the same number, else the oldest one waiting a little before this text
            same = [k for k in marks if k[2] == opt] or [k for k in marks if m.start() - k[1] < 4000]
            if same:
                marks.remove(same[0])
                edits.append((same[0][0], same[0][1], note))
                note = ""
        edits.append((m.start(), e, note))
        return None

    rewrite_outside_math(text, _NOTE, scan, passes=1)
    for start, end, rep in sorted(edits, reverse=True):
        text = text[:start] + rep + text[end:]
    return text


_SUB = re.compile(r"\\(?P<cmd>subfloat|subfigure|subtable|subcaptionbox|subcaption)(?![a-zA-Z@])(?P<star>\*?)")
_FLOAT_BEGIN = re.compile(r"\\begin\{(?:table|figure)\*?\}")


def _sub_label(text, at):
    """"a", "b", ... for the next sub-caption of the float open at ``at`` (counts the ones already written)."""
    starts = [m.end() for m in _FLOAT_BEGIN.finditer(text, 0, at)]
    n = len(re.findall(r"\\par\\textit\{\([a-z0-9]+\)", text[starts[-1]:at])) if starts else 0
    return chr(ord("a") + n) if n < 26 else str(n)


def subfloat_captions(text: str) -> str:
    r"""Command-style sub-floats: ``\subfloat[cap]{body}``, ``\subfigure[cap]{body}``, ``\subcaptionbox{cap}{body}``
    and ``\subcaption{cap}`` -> the body followed by an "(a) cap" line (pandoc drops command and caption)."""
    pos = 0
    while True:
        m = _SUB.search(text, pos)
        if not m:
            return text
        cmd, i = m.group("cmd"), m.end()
        caption, numbered = None, not m.group("star")
        if cmd in ("subcaptionbox", "subcaption"):
            i = skip_ws(text, i)
            e = balanced_end(text, i) if i < len(text) and text[i] == "{" else -1
            if e < 0:
                pos = m.end()
                continue
            caption, i = text[i + 1:e - 1], e
        else:
            while True:   # \subfloat[list entry][caption]: the last optional argument is the caption
                j = skip_ws(text, i)
                if j >= len(text) or text[j] != "[":
                    break
                e = bracket_end(text, j)
                if e < 0:
                    break
                caption, i = text[j + 1:e - 1], e
        body = ""
        if cmd != "subcaption":
            i = skip_optional(text, i) if cmd == "subcaptionbox" else skip_ws(text, i)
            e = balanced_end(text, i) if i < len(text) and text[i] == "{" else -1
            if e < 0:
                pos = m.end()
                continue
            body, i = text[i + 1:e - 1], e
        line = ""
        if caption is not None:
            label = "(" + _sub_label(text, m.start()) + ") " if numbered else ""
            line = "\\par\\textit{" + (label + caption.strip()).strip() + "}\\par "
        text = text[:m.start()] + body + line + text[i:]
        pos = m.start()


_FLOAT = re.compile(r"\\begin\{(table|figure)(\*?)\}")
_CAPTION = re.compile(r"\\caption(\*?)(?![a-zA-Z@])\s*(?:\[[^\]]*\])?\s*(?=\{)")


def starred_captions_to_text(text: str) -> str:
    r"""A float with its numbered ``\caption`` plus ``\caption*{(a) ...}`` lines under each panel: pandoc keeps
    one caption per float, so the starred ones become plain lines."""
    out, pos = [], 0
    while True:
        m = _FLOAT.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        close = "\\end{" + m.group(1) + m.group(2) + "}"
        end = text.find(close, m.end())
        if end < 0:
            out.append(text[pos:])
            return "".join(out)
        body = text[m.end():end]
        captions = []
        for c in _CAPTION.finditer(body):
            e = balanced_end(body, c.end())
            if e > 0:
                captions.append((c.start(), c.end(), e, bool(c.group(1))))
        starred = [c for c in captions if c[3]]
        if len(captions) >= 2 and starred and not re.search(r"\\begin\{(?:table|figure)", body):
            keep = None if len(starred) < len(captions) else starred[-1]   # no numbered caption: the last stays
            for start, cend, e, _ in reversed([c for c in starred if c is not keep]):
                body = body[:start] + "\\par\\textit{" + body[cend + 1:e - 1].strip() + "}\\par " + body[e:]
        out.append(text[pos:m.end()] + body + close)
        pos = end + len(close)


_SIMPLE_MACRO = re.compile(r"\\(?:(?:re)?newcommand\*?\s*\{?\\([a-zA-Z]+)\}?|def\s*\\([a-zA-Z]+))\s*\{([^{}\\#]*)\}")
_IFTHEN = re.compile(r"\\(ifthenelse|ifbool|iftoggle|notbool|nottoggle)(?![a-zA-Z@])\s*(?=\{)")
_FLAG = r"\\(?:setboolean|setbool|settoggle)\s*\{%s\}\s*\{(true|false)\}|\\(?:bool|toggle)(true|false)\s*\{%s\}|\\%s(true|false)(?![a-zA-Z])"


def _flag(name, text, at):
    """Value of a boolean/toggle at ``at``: its last assignment before it (False when only declared), else None."""
    name = re.escape(name.strip())
    assignments = list(re.finditer(_FLAG % (name, name, name), text[:at]))
    if assignments:
        last = assignments[-1]
        return (last.group(1) or last.group(2) or last.group(3)) == "true"
    if re.search(r"\\(?:newboolean|newbool|newtoggle|providebool|providetoggle)\s*\{" + name + r"\}", text[:at]):
        return False
    return None


def resolve_conditionals(text: str) -> str:
    r"""``\ifthenelse{\equal{\anonymous}{0}}{A}{B}``, ``\ifthenelse{\boolean{x}}{A}{B}``, ``\iftoggle{x}{A}{B}``:
    keep the branch the document takes. pandoc knows none of them and drops both branches (acknowledgments,
    author contributions). A test that cannot be decided here is left as it is. ``\if 0`` becomes ``\iffalse``."""
    # "\if 0 ... \fi" compares "0" with the space after it: always false, a block the authors switched off (pandoc
    # keeps it, with its "??")
    text = re.sub(r"\\if\s*[0-9](?=[ \t\n%])", r"\\iffalse", text)
    macros = {m.group(1) or m.group(2): m.group(3).strip() for m in _SIMPLE_MACRO.finditer(text)}

    def literal(s):
        s = re.sub(r"\\([a-zA-Z]+)\s*(?:\{\})?", lambda m: macros.get(m.group(1), m.group(0)), s).strip()
        return None if "\\" in s else s

    pos = 0
    while True:
        m = _IFTHEN.search(text, pos)
        if not m:
            return text
        parts, i = [], m.end()               # (inner start, inner end, end of the argument)
        for k in range(3):
            i = skip_ws(text, i)
            if text[i:i + 1] == "{":
                e = balanced_end(text, i)
                if e < 0:
                    break
                parts.append((i + 1, e - 1, e))
                i = e
            elif k == 2:
                # TeX reads an undelimited argument: a lone token is one too (\iftoggle{t}{A}\subsection{B})
                tok = re.compile(r"\\[a-zA-Z@]+|\\.|[^\s{}]").match(text, i)
                if tok:
                    parts.append((i, tok.end(), tok.end()))
            else:
                break
        pos = m.end()
        # "#1" in the test: the conditional is the body of a macro definition, to be decided at each use
        if len(parts) < 3 or "#" in text[parts[0][0]:parts[0][1]]:
            continue
        test = text[parts[0][0]:parts[0][1]].strip()
        value = None
        if m.group(1) == "ifthenelse":
            eq = re.fullmatch(r"\\equal\s*\{([^{}]*)\}\s*\{([^{}]*)\}", test)
            flag = re.fullmatch(r"\\boolean\s*\{([^{}]*)\}", test)
            if eq:
                a, b = literal(eq.group(1)), literal(eq.group(2))
                value = None if a is None or b is None else a == b
            elif flag:
                value = _flag(flag.group(1), text, m.start())
        else:
            value = _flag(test, text, m.start())
            if value is not None and m.group(1).startswith("not"):
                value = not value
        if value is None:
            continue
        start, end, _ = parts[1] if value else parts[2]
        text = text[:m.start()] + text[start:end] + text[parts[2][2]:]
        pos = m.start()


_CHECKLIST = re.compile(r"\\answer(Yes|No|NA|TODO)(?![a-zA-Z@])\s*(?=\{)")
_CHECKLIST_LABEL = {"Yes": "[Yes]", "No": "[No]", "NA": "[N/A]", "TODO": "[TODO]"}


def checklist_answers(text: str) -> str:
    r"""NeurIPS paper checklist: ``\answerYes{We included the code}`` is "[Yes] We included the code" in the PDF. The macro
    lives in the conference style file, so pandoc drops it with its justification."""
    body_at = text.find("\\begin{document}")

    def handler(m, t):
        if m.start() < body_at or _DEFINES_ON_LINE.search(t, t.rfind("\n", 0, m.start()) + 1, m.start()):
            return None
        e = balanced_end(t, m.end())
        if e < 0:
            return None
        return f"{_CHECKLIST_LABEL[m.group(1)]} {t[m.end() + 1:e - 1].strip()}".rstrip() + " ", e
    return rewrite_outside_math(text, _CHECKLIST, handler, passes=1)
