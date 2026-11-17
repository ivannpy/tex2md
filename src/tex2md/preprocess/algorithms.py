"""algorithm/algorithmic environments to plain lines."""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end

ALGO_CMDS = {
    # name: (nargs, kind, opening text, closing text)
    # kind: stmt = plain line, open = line then indent, mid = dedent+line+indent, end = dedent+line
    "state": (0, "stmt", "", ""), "statex": (0, "stmt", "", ""),
    "require": (0, "stmt", r"\textbf{Require:} ", ""), "ensure": (0, "stmt", r"\textbf{Ensure:} ", ""),
    "input": (0, "stmt", r"\textbf{Input:} ", ""), "output": (0, "stmt", r"\textbf{Output:} ", ""),
    "return": (0, "stmt", r"\textbf{return} ", ""), "print": (0, "stmt", r"\textbf{print} ", ""),
    "if": (1, "open", r"\textbf{if} ", r" \textbf{then}"),
    "elsif": (1, "mid", r"\textbf{else if} ", r" \textbf{then}"), "elseif": (1, "mid", r"\textbf{else if} ", r" \textbf{then}"),
    "else": (0, "mid", r"\textbf{else}", ""), "endif": (0, "end", r"\textbf{end if}", ""),
    "for": (1, "open", r"\textbf{for} ", r" \textbf{do}"), "forall": (1, "open", r"\textbf{for all} ", r" \textbf{do}"),
    "endfor": (0, "end", r"\textbf{end for}", ""),
    "while": (1, "open", r"\textbf{while} ", r" \textbf{do}"), "endwhile": (0, "end", r"\textbf{end while}", ""),
    "repeat": (0, "open", r"\textbf{repeat}", ""), "until": (1, "end", r"\textbf{until} ", ""),
    "loop": (0, "open", r"\textbf{loop}", ""), "endloop": (0, "end", r"\textbf{end loop}", ""),
    "procedure": (2, "open", r"\textbf{procedure} ", ""), "function": (2, "open", r"\textbf{function} ", ""),
    "endprocedure": (0, "end", r"\textbf{end procedure}", ""), "endfunction": (0, "end", r"\textbf{end function}", ""),
}


ALGO_INLINE = {"comment": 1, "call": 2}


def _brace_args(text, i, n):
    """Read n {..} arguments starting at text[i]; returns (args, end), or (None, i) if they are not there."""
    args = []
    for _ in range(n):
        while i < len(text) and text[i] in " \t\n":
            i += 1
        if i >= len(text) or text[i] != "{":
            return None, i
        e = balanced_end(text, i)
        if e < 0:
            return None, i
        args.append(text[i + 1:e - 1])
        i = e
    return args, i


def algorithmic_to_lines(body):
    """Rewrite the body of an algorithmic environment (\\STATE/\\State, \\IF/\\If ... styles) as one
    paragraph per line, keywords in bold and nesting shown with \\quad; pandoc would otherwise run
    all the lines together or drop the unknown commands."""
    cmd = re.compile(r"\\([A-Za-z]+)")

    def inline(text):
        # \Comment{x} / \Call{f}{a} inside a statement
        out, pos = "", 0
        for m in cmd.finditer(text):
            nm = m.group(1).lower()
            if m.start() < pos or nm not in ALGO_INLINE:
                continue
            args, e = _brace_args(text, m.end(), ALGO_INLINE[nm])
            if args is None:
                continue
            out += text[pos:m.start()]
            out += (r"\hfill\textit{// " + args[0] + "}") if nm == "comment" else (r"\textsc{" + args[0] + "}(" + args[1] + ")")
            pos = e
        return out + text[pos:]

    pieces, pos, scan = [], 0, 0
    while True:
        m = cmd.search(body, scan)
        if not m:
            break
        nm = m.group(1).lower()
        if nm not in ALGO_CMDS:
            scan = m.end()
            continue
        nargs, kind, pre, post = ALGO_CMDS[nm]
        args, e = _brace_args(body, m.end(), nargs)
        if args is None:
            scan = m.end()
            continue
        pieces.append((body[pos:m.start()], None))
        pieces.append((nm, (kind, pre, post, args)))
        pos = scan = e
        # algorithmic.sty-style "\IF{c}{ ... }": the braces after a block head only group; drop them
        if kind in ("open", "mid") and nm not in ("procedure", "function"):
            j = pos
            while j < len(body) and body[j] in " \t\n":
                j += 1
            if j < len(body) and body[j] == "{":
                end = balanced_end(body, j)
                if end > 0:
                    body = body[:j] + " " + body[j + 1:end - 1] + " " + body[end:]
                    pos = scan = j
    pieces.append((body[pos:], None))

    lines, cur, depth = [], "", 0

    def flush():
        nonlocal cur
        if cur.strip():
            lines.append((depth, cur.strip()))
        cur = ""

    for text, info in pieces:
        if info is None:
            cur += inline(text)
            continue
        kind, pre, post, args = info
        flush()
        if kind in ("mid", "end"):
            depth = max(depth - 1, 0)
        if len(args) == 2:  # procedure / function name(args)
            head = pre + r"\textsc{" + args[0] + "}(" + args[1] + ")"
        else:
            head = pre + "".join(args)
        if kind == "stmt":
            cur = head + post + " "
        else:
            lines.append((depth, (head + post).strip()))
            if kind in ("open", "mid"):
                depth += 1
    flush()
    return "\n\n".join(r"\quad" * d + (r"\ " if d else "") + t for d, t in lines)


def rewrite_algorithmic(text):
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{algorithmic\}(\[[^\]]*\])?(.*?)\\end\{algorithmic\}", text, re.S):
        out.append(text[pos:m.start()])
        out.append("\\begin{algorithmic}\n\n" + algorithmic_to_lines(m.group(2)) + "\n\n\\end{algorithmic}")
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def _caption_key(tex):
    """The start of a caption the way ``parse_floats`` keeps the .aux captions (letters and digits only)."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\\[a-zA-Z@]+\s*", " ", tex).lower())[:40]


class _AlgorithmNumbers:
    """Number of each algorithm caption: its own label, else the matching .aux caption, else a running count."""

    def __init__(self, labels, floats):
        self.labels = labels
        self.entries = [f for f in (floats or []) if f.get("kind") == "algorithm"]
        self.used: set[int] = set()
        self.count = 0

    def number(self, caption, label):
        self.count += 1
        lab = self.labels.get(label) if label else None
        # a \label after the caption may belong to a line of the algorithm: only an algorithm label counts
        if lab and lab.get("num") and lab.get("type") in ("algorithm", "") and (lab["type"] or lab["num"].isdigit()):
            self._take(caption)
            return lab["num"]
        return self._take(caption) or str(self.count)

    def _take(self, caption):
        key = _caption_key(caption)
        for n in (40, 24, 12):
            prefix = key[:n]
            if not prefix:
                break
            for i, entry in enumerate(self.entries):
                if i not in self.used and entry["text"].startswith(prefix):
                    self.used.add(i)
                    return entry["num"]
        return None


def rewrite_algorithm_captions(text, labels, floats=None):
    """pandoc drops caption inside algorithm environments; keep it as a bold "Algoritmo N." paragraph, with the
    number the compiled document gives it (``\\captionof{algorithm}`` outside the environment included)."""
    numbers = _AlgorithmNumbers(labels, floats)
    out, pos = [], 0
    spans = [(m.start(), m.end(), m) for m in re.finditer(r"\\begin\{algorithm\*?\}(\[[^\]]*\])?(.*?)\\end\{algorithm\*?\}", text, re.S)]
    loose = re.compile(r"\\captionof\s*\{algorithm\}\s*(?:\[[^\]]*\])?\s*(?=\{)")

    def titled(cap, after):
        lab = re.search(r"\\label\{([^}]*)\}", cap) or re.match(r"\s*\\label\{([^}]*)\}", after)
        cap = re.sub(r"\\label\{[^}]*\}", "", cap).strip()
        return "\n\n\\textbf{Algoritmo " + numbers.number(cap, lab.group(1) if lab else None) + ".} " + cap + "\n\n"

    def loose_captions(chunk):
        res, at = [], 0
        for c in loose.finditer(chunk):
            e = balanced_end(chunk, c.end())
            if e < 0 or c.start() < at:
                continue
            res.append(chunk[at:c.start()] + titled(chunk[c.end() + 1:e - 1], chunk[e:e + 200]))
            at = e
        return "".join(res) + chunk[at:]

    for start, end, m in spans:
        out.append(loose_captions(text[pos:start]))
        body = m.group(2)
        cm = re.search(r"\\caption\*?(?:\[[^\]]*\])?(?=\{)", body)
        if cm:
            e = balanced_end(body, cm.end())
            if e > 0:
                body = body[:cm.start()] + titled(body[cm.end() + 1:e - 1], body[e:e + 200]) + body[e:]
        else:
            numbers.count += 1     # an algorithm without caption still takes a number
        out.append(text[start:m.start(2)] + body + text[m.end(2):end])
        pos = end
    out.append(loose_captions(text[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- algorithm2e

A2E_MARKERS = re.compile(r"\\(?:KwIn|KwOut|KwData|KwResult|Input|Output|For|ForEach|ForAll|While|If|uIf|eIf|lIf|Repeat|Return|tcp|tcc)\b|\\;")
A2E_HEADS = {  # name: (number of {..} args that follow, text before, text after the first argument)
    "For": (2, "for ", " do"), "ForEach": (2, "for each ", " do"), "ForAll": (2, "for all ", " do"),
    "While": (2, "while ", " do"), "If": (2, "if ", " then"), "uIf": (2, "if ", " then"),
    "ElseIf": (2, "else if ", " then"), "uElseIf": (2, "else if ", " then"),
    # the one-line forms (\lIf{c}{stmt}) read the same once written out
    "lFor": (2, "for ", " do"), "lForEach": (2, "for each ", " do"), "lForAll": (2, "for all ", " do"),
    "lWhile": (2, "while ", " do"), "lIf": (2, "if ", " then"), "lElseIf": (2, "else if ", " then"),
}
A2E_BARE = {"Else": "else", "uElse": "else", "lElse": "else"}
A2E_LABELS = {"KwIn": "Input", "KwOut": "Output", "KwData": "Data", "KwResult": "Result", "Input": "Input", "Output": "Output"}


def _a2e_inline(s):
    """Keyword macros that may appear inside a header or an input list."""
    s = re.sub(r"\\KwTo\b", r" \\textbf{to} ", s)
    s = re.sub(r"\\(?:Return|KwRet)\b\s*", r"\\textbf{return} ", s)
    # "\;" ends a statement; inside a formula it is a thin space and stays
    s = re.sub(r"(\$[^$]*\$)|\\;", lambda m: m.group(1) or "", s)
    return re.sub(r" {2,}", " ", s).strip()


def _a2e(text, depth, lines):
    r"""algorithm2e source -> list of (depth, tex line). Statements end at \; ; blocks are {..} groups."""
    cur, i, in_math = [], 0, False

    def flush():
        s = re.sub(r" {2,}", " ", "".join(cur)).strip()
        if s:
            lines.append((depth, s))
        cur.clear()

    while i < len(text):
        c = text[i]
        if c == "$":      # a formula: its "\;", "\\" and macros are not algorithm2e's
            n = 2 if text.startswith("$$", i) else 1
            in_math = not in_math
            cur.append(text[i:i + n])
            i += n
            continue
        if c == "\\":
            m = re.compile(r"\\([A-Za-z]+)\*?|\\;|\\\\").match(text, i)
            if not m or in_math:
                step = m.end() - i if m else 2
                cur.append(text[i:i + step])
                i += step
                continue
            name = m.group(1)
            if m.group(0) in (r"\;", r"\\"):
                flush()
                i = m.end()
            elif name in A2E_LABELS:
                flush()
                args, e = _brace_args(text, m.end(), 1)
                if args is not None:
                    lines.append((depth, r"\textbf{" + A2E_LABELS[name] + r":} " + _a2e_inline(args[0])))
                    i = e
                else:
                    i = m.end()
            elif name in A2E_HEADS:
                n, pre, post = A2E_HEADS[name]
                args, e = _brace_args(text, m.end(), n)
                if args is None:
                    cur.append(m.group(0))
                    i = m.end()
                    continue
                flush()
                lines.append((depth, r"\textbf{" + pre.strip() + "} " + _a2e_inline(args[0]) + (r" \textbf{" + post.strip() + "}")))
                _a2e(args[1], depth + 1, lines)
                i = e
            elif name == "eIf":     # \eIf{condition}{then}{else}
                args, e = _brace_args(text, m.end(), 3)
                if args is None:
                    cur.append(m.group(0))
                    i = m.end()
                    continue
                flush()
                lines.append((depth, r"\textbf{if} " + _a2e_inline(args[0]) + r" \textbf{then}"))
                _a2e(args[1], depth + 1, lines)
                lines.append((depth, r"\textbf{else}"))
                _a2e(args[2], depth + 1, lines)
                i = e
            elif name in A2E_BARE:
                args, e = _brace_args(text, m.end(), 1)
                flush()
                lines.append((depth, r"\textbf{" + A2E_BARE[name] + "}"))
                if args is not None:
                    _a2e(args[0], depth + 1, lines)
                    i = e
                else:
                    i = m.end()
            elif name == "Repeat":
                args, e = _brace_args(text, m.end(), 2)
                if args is None:
                    i = m.end()
                    continue
                flush()
                lines.append((depth, r"\textbf{repeat}"))
                _a2e(args[0], depth + 1, lines)
                lines.append((depth, r"\textbf{until} " + _a2e_inline(args[1])))
                i = e
            elif name in ("tcp", "tcc"):
                j = m.end()
                opt = re.compile(r"\*?(?:\[[^\]]*\])?").match(text, j)
                args, e = _brace_args(text, opt.end() if opt else j, 1)
                if args is not None:
                    cur.append(r"\hfill\textit{// " + args[0].strip() + "}")
                    i = e
                else:
                    i = m.end()
            elif name in ("Return", "KwRet"):
                cur.append(r"\textbf{return} ")
                i = m.end()
            elif name == "KwTo":
                cur.append(r" \textbf{to} ")
                i = m.end()
            else:
                cur.append(m.group(0))
                i = m.end()
        else:
            cur.append(c)
            i += 1
    flush()


def algorithm2e_to_lines(body):
    lines = []
    _a2e(body, 0, lines)
    return "\n\n".join(r"\quad" * d + (r"\ " if d else "") + t for d, t in lines)


def rewrite_algorithm2e(text):
    """``algorithm`` environments written with algorithm2e (\\KwIn, \\For{..}{..}, \\;): pandoc keeps only
    the caption, so rewrite the body as one paragraph per line (caption and label stay where they are)."""
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{algorithm\*?\}(\[[^\]]*\])?(.*?)\\end\{algorithm\*?\}", text, re.S):
        body = m.group(2)
        if "\\begin{algorithmic}" in body or not A2E_MARKERS.search(body):
            continue
        keep, rest, last = [], [], 0
        for cm in re.finditer(r"\\(?:caption|label)\*?(?:\[[^\]]*\])?(?=\{)", body):
            if cm.start() < last:
                continue
            e = balanced_end(body, cm.end())
            if e < 0:
                continue
            rest.append(body[last:cm.start()])
            keep.append(body[cm.start():e])
            last = e
        rest.append(body[last:])
        new = "".join(keep) + "\n\n" + algorithm2e_to_lines("".join(rest)) + "\n\n"
        out.append(text[pos:m.start(2)] + new)
        pos = m.end(2)
    out.append(text[pos:])
    return "".join(out)
