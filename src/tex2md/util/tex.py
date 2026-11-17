"""TeX source parsing helpers (balanced groups, comments, macro unwrapping)."""
from __future__ import annotations

import re


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def group(text, i, open_="{", close="}"):
    """Return (content, end) of the balanced group starting at text[i] == open_."""
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    return text[i + 1:], len(text)


def skip_ws(text, i):
    while i < len(text) and text[i] in " \t\n":
        i += 1
    return i


def clean_num(s):
    s = re.sub(r"\\(relax|textup|textnormal|mbox|text|ensuremath|hbox)\b", "", s)
    s = re.sub(r"\\[a-zA-Z@]+\s*", "", s)
    return re.sub(r"[{}\s]", "", s)


def unwrap(text, cmd, nargs_drop, fn=None):
    """\\cmd{a}{b}{BODY} -> BODY (or fn(BODY)), dropping the first nargs_drop groups."""
    out, i = [], 0
    pat = re.compile(r"\\" + cmd + r"\*?\s*")
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        ok = True
        for _ in range(nargs_drop):
            j = skip_ws(text, j)
            if j < len(text) and text[j] == "[":
                _, j = group(text, j, "[", "]")
                j = skip_ws(text, j)
            if j >= len(text) or text[j] != "{":
                ok = False
                break
            _, j = group(text, j)
        j = skip_ws(text, j)
        if not ok or j >= len(text) or text[j] != "{":
            out.append(text[m.start():m.end()])
            i = m.end()
            continue
        body, j = group(text, j)
        out.append(fn(body) if fn else body)
        i = j
    return "".join(out)


def balanced_end(text, i):
    """Index just past the {...} group starting at text[i] (== "{"), or -1."""
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


MATH_ENVS = r"(?:equation|align|alignat|flalign|gather|multline|eqnarray|displaymath|math|IEEEeqnarray|dmath|empheq|mathpar)\*?"
# (the comment environment of the comment package is read verbatim and typeset as nothing: braces in it do not count)
VERBATIM_ENVS = r"(?:verbatim|Verbatim|BVerbatim|LVerbatim|lstlisting|minted|spverbatim|comment)\*?"

_PROTECT = re.compile(
    r"\\begin\{(?P<env>" + MATH_ENVS + "|" + VERBATIM_ENVS + r")\}"
    r"|\\verb\*?(?P<delim>[^a-zA-Z\s*])"
    r"|\\(?P<open>[(\[])"
    r"|\\[\s\S]"                 # any other escape (\$, \%, \\, the first letter of a control word)
    r"|(?P<dollar>\$\$?)"
    r"|%[^\n]*")
_DOLLAR_END = re.compile(r"\\[\s\S]|(?P<close>\$\$?)|(?P<par>\n[ \t]*\n)")


def protected_spans(text, math=True):
    """Sorted (start, end) of the regions a text-mode rewrite must not touch: math (``$..$``, ``$$..$$``,
    ``\\(..\\)``, ``\\[..\\]``, equation-like environments) and verbatim (``\\verb``, listings).
    With ``math=False`` only the verbatim regions are returned."""
    spans, i = [], 0
    while True:
        m = _PROTECT.search(text, i)
        if not m:
            return spans
        i = m.end()
        if m.group("env"):
            if not math and re.fullmatch(MATH_ENVS, m.group("env")):
                continue
            close = "\\end{" + m.group("env") + "}"
            end = text.find(close, i)
            if end >= 0:
                i = end + len(close)
                spans.append((m.start(), i))
        elif m.group("delim"):
            end = text.find(m.group("delim"), i)
            if end >= 0 and "\n" not in text[i:end]:
                i = end + 1
                spans.append((m.start(), i))
        elif m.group("open") and math:
            end = text.find("\\)" if m.group("open") == "(" else "\\]", i)
            if end >= 0:
                i = end + 2
                spans.append((m.start(), i))
        elif m.group("dollar") and math:
            display = m.group("dollar") == "$$"
            j = i
            while True:
                c = _DOLLAR_END.search(text, j)
                if not c or (c.group("par") and not display):
                    break    # no closing dollar in this paragraph: a literal "$"
                j = c.end()
                if c.group("close") and (not display or c.group("close") == "$$"):
                    # "$a$$b$": the first dollar of "$$" closes the inline formula
                    i = c.start() + (2 if display else 1)
                    spans.append((m.start(), i))
                    break


def span_at(spans, pos):
    """The protected span containing ``pos``, or None (``spans`` as returned by ``protected_spans``)."""
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][1] <= pos:
            lo = mid + 1
        else:
            hi = mid
    return spans[lo] if lo < len(spans) and spans[lo][0] <= pos else None


def rewrite_outside_math(text, pattern, handler, passes=6):
    """Rewrite what ``pattern`` finds outside math and verbatim. ``handler(match, text)`` returns
    ``(replacement, end)`` or None to leave the match alone. Runs again while something changed, so a
    construct nested in a rewritten one is reached too."""
    for _ in range(passes):
        spans = protected_spans(text)
        out, pos = [], 0
        for m in pattern.finditer(text):
            if m.start() < pos or span_at(spans, m.start()):
                continue
            res = handler(m, text)
            if res is None:
                continue
            rep, end = res
            out.append(text[pos:m.start()] + rep)
            pos = end
        if not out:
            return text
        out.append(text[pos:])
        text = "".join(out)
    return text


def bracket_end(text, i):
    """Index just past the [...] starting at text[i] (== "["), or -1. A "]" inside a {...} group does not close it:
    ``\\caption[short {list]} entry]{...}``."""
    j = i + 1
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
        elif c == "{":
            j = balanced_end(text, j)
            if j < 0:
                return -1
        elif c == "]":
            return j + 1
        else:
            j += 1
    return -1


def skip_optional(text, i):
    """Index after any run of ``[...]`` optional arguments (and the blanks around them) starting at i."""
    while True:
        j = skip_ws(text, i)
        if j < len(text) and text[j] == "[":
            end = bracket_end(text, j)
            if end < 0:
                return j
            i = end
        else:
            return j
