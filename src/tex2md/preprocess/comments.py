r"""Comment stripping that leaves verbatim text alone.

``latexpand`` strips comments line by line and knows nothing about verbatim: ``if i % 2 == 1:`` inside a
listing is cut at the percent sign. The source is therefore flattened with ``--keep-comments`` and the comments
are emptied here, with the same result as ``latexpand --empty-comments`` everywhere except inside verbatim
environments, ``\verb`` and ``\lstinline``.
"""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end, protected_spans, span_at

VERBATIM_NAMES = ("verbatim", "verbatim*", "Verbatim", "Verbatim*", "BVerbatim", "LVerbatim", "lstlisting",
                  "minted", "spverbatim", "alltt")

_DEFINED = re.compile(r"\\(?:DefineVerbatimEnvironment|lstnewenvironment|newtcblisting|NewTCBListing|newminted)"
                      r"\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}")
# what may hold a literal "%" on a line of running text
_TOKEN = re.compile(r"\\verb\*?(?P<d>[^a-zA-Z\s*]).*?(?P=d)"
                    r"|\\lstinline\s*(?:\[[^\]]*\])?(?P<l>[^a-zA-Z\s\[{]).*?(?P=l)"
                    r"|\\(?:url|href|nolinkurl|path)\s*\{[^{}]*\}"
                    r"|\\.|(?P<comment>%)")
_URL_PERCENT = re.compile(r"(?<!\\)%")


def verbatim_environments(text: str) -> set[str]:
    """Names of the verbatim-like environments of a document: the standard ones plus the ones it defines."""
    names = set(VERBATIM_NAMES)
    for m in _DEFINED.finditer(text):
        names.add(m.group(1).strip())
    return names


def _empty_comment(line: str) -> str:
    """``text % comment`` -> ``text %`` (the percent sign stays: it still swallows the line break)."""
    out: list[str] = []
    pos = 0
    for m in _TOKEN.finditer(line):
        if m.group("comment"):
            return "".join(out) + line[pos:m.start()] + "%"
        if m.group(0).startswith(("\\url", "\\href", "\\nolinkurl", "\\path")):
            # a "%" in a URL is literal; escaped, the later stages cannot take it for a comment
            out.append(line[pos:m.start()] + _URL_PERCENT.sub(r"\\%", m.group(0)))
            pos = m.end()
    return "".join(out) + line[pos:]


def empty_comments(text: str) -> str:
    """Empty every TeX comment outside verbatim and drop what follows ``\\end{document}``."""
    names = verbatim_environments(text)
    begin = re.compile(r"\\begin\{(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\}")
    out, inside = [], None
    for raw in text.splitlines(keepends=True):
        line, eol = (raw[:-1], "\n") if raw.endswith("\n") else (raw, "")
        if inside:
            out.append(raw)
            if "\\end{" + inside + "}" in line:
                inside = None
            continue
        line = _empty_comment(line)
        out.append(line + eol)
        opened = list(begin.finditer(line))
        if opened and "\\end{" + opened[-1].group(1) + "}" not in line[opened[-1].end():]:
            inside = opened[-1].group(1)
        if re.match(r"[ \t]*\\end\{document\}", line):
            break
    return "".join(out)


_LST_NEW = re.compile(r"\\lstnewenvironment\s*\{([^{}]+)\}\s*(?:\[(\d)\])?\s*(?:\[[^\]]*\])?\s*(?=\{)")
_VERB_NEW = re.compile(r"\\DefineVerbatimEnvironment\s*\{([^{}]+)\}\s*\{[^{}]*\}\s*(?=\{)")
_TOKEN_MACRO = re.compile(r"\\[A-Za-z]+Tok\{")
_HIGHLIGHT_ESCAPES = (("\\textless{}", "<"), ("\\textgreater{}", ">"), ("\\textquotesingle{}", "'"),
                      ("\\textquotedbl{}", '"'), ("\\textasciigrave{}", "`"), ("\\textasciitilde{}", "~"),
                      ("\\textasciicircum{}", "^"), ("\\textbar{}", "|"), ("{-}", "-"))


def _plain_highlighting(body: str) -> str:
    r"""A ``Highlighting`` block written by pandoc (``\KeywordTok{def}\NormalTok{ f():}``) -> the code itself."""
    while True:
        m = _TOKEN_MACRO.search(body)
        if not m:
            break
        e = balanced_end(body, m.end() - 1)
        if e < 0:
            break
        body = body[:m.start()] + body[m.end():e - 1] + body[e:]
    for escaped, char in _HIGHLIGHT_ESCAPES:
        body = body.replace(escaped, char)
    body = re.sub(r"\\([{}_&%#$])", r"\1", body)
    return body.replace("\\textbackslash{}", "\\")


def plain_verbatim_environments(text: str) -> str:
    r"""Verbatim environments of the document's own (``\DefineVerbatimEnvironment``, ``\lstnewenvironment``) and
    ``spverbatim`` -> ``verbatim`` / ``lstlisting``: pandoc reads an environment it does not know as ordinary
    LaTeX, which runs the lines of a listing together and drops every ``%`` comment in it."""
    renames = {"spverbatim": ("verbatim", 0, False)}
    for pattern, target, groups in ((_LST_NEW, "lstlisting", 2), (_VERB_NEW, "verbatim", 1)):
        while True:
            m = pattern.search(text)
            if not m:
                break
            end = m.end()
            for _ in range(groups):
                while end < len(text) and text[end] in " \t\n":
                    end += 1
                e = balanced_end(text, end) if end < len(text) and text[end] == "{" else -1
                if e < 0:
                    break
                end = e
            nargs = int(m.group(2)) if pattern is _LST_NEW and m.group(2) else 0
            # commandchars: the block holds commands (pandoc's own highlighted code), not literal text
            renames[m.group(1).strip()] = (target, nargs, "commandchars" in text[m.end():end])
            text = text[:m.start()] + text[end:]
    for name, (target, nargs, highlighted) in renames.items():
        esc = re.escape(name)
        if highlighted:
            text = re.sub(r"\\begin\{" + esc + r"\}(?:\[[^\]\n]*\])?(.*?)\\end\{" + esc + r"\}",
                          lambda m: "\\begin{verbatim}" + _plain_highlighting(m.group(1)) + "\\end{verbatim}",
                          text, flags=re.S)
            continue
        # the arguments of the environment (a caption, a style) are layout: drop them with the name
        args = r"(?:[ \t]*(?:\[[^\]\n]*\]|\{[^{}\n]*\})){0,%d}" % nargs if nargs else r"(?:[ \t]*\[[^\]\n]*\])?"  # noqa: UP031
        text = re.sub(r"\\begin\{" + esc + r"\}" + args, "\\\\begin{" + target + "}", text)
        text = re.sub(r"\\end\{" + esc + r"\}", "\\\\end{" + target + "}", text)
    return text


_LISTING_FILE = re.compile(
    r"\\(?P<cmd>lstinputlisting|verbatiminput\*?|inputminted|VerbatimInput|BVerbatimInput|LVerbatimInput|inputcode)"
    r"(?![a-zA-Z@])\s*(?:\[[^\]]*\])*\s*(?:\{(?P<lang>[^{}]*)\}\s*)?\{(?P<file>[^{}]+)\}")


def inline_listing_files(text: str, folder: str, read=None) -> str:
    r"""``\lstinputlisting{data/prompt.md}`` and the like -> a ``verbatim`` block with the file's text. pandoc would
    read the file itself, but it runs in an empty directory (so that it does not parse the paper's .sty files): the
    prompts and samples that papers keep in side files were missing from the Markdown."""
    import os

    def load(path):
        for candidate in (path, path + ".tex", path + ".txt"):
            full = os.path.join(folder, candidate)
            if os.path.isfile(full):
                with open(full, encoding="utf-8", errors="replace") as fh:
                    return fh.read()
        return None
    read = read or load

    def one(m):
        name = m.group("file").strip()
        lang = m.group("lang")
        if m.group("cmd") != "inputminted":
            lang = None
        content = read(name)
        if content is None or "\\end{verbatim}" in content:
            return m.group(0)
        env = "lstlisting" if lang else "verbatim"
        return "\\begin{" + env + "}" + (f"[language={lang}]" if lang else "") + "\n" + content.rstrip("\n") + "\n\\end{" + env + "}"
    return _LISTING_FILE.sub(one, text)


_INPUT = re.compile(r"\\(?:input|include)(?![a-zA-Z@])\s*\{([^{}]+)\}")


def expand_remaining_inputs(text: str, folder: str, depth: int = 0) -> str:
    r"""``\input{a} & \input{b}``: latexpand expands one \input per line and leaves the other for pandoc, which runs in
    an empty directory and drops the whole construct (the generation examples of a results table disappeared with
    it). The ones that are left and exist on disk are put in here, comments emptied, as many levels deep as needed."""
    import os
    if depth > 8:
        return text
    spans = protected_spans(text, math=False)

    def one(m):
        if span_at(spans, m.start()):
            return m.group(0)
        name = m.group(1).strip()
        for candidate in (name, name + ".tex"):
            path = os.path.join(folder, candidate)
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    body = empty_comments(fh.read())
                return "\n" + expand_remaining_inputs(body, folder, depth + 1) + "\n"
        return m.group(0)
    return _INPUT.sub(one, text)
