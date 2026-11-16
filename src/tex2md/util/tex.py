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
