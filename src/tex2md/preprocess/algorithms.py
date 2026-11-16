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


def rewrite_algorithm_captions(text, labels):
    """pandoc drops caption inside algorithm environments; keep it as a bold "Algoritmo N." paragraph."""
    out, pos, count = [], 0, 0
    for m in re.finditer(r"\\begin\{algorithm\*?\}(\[[^\]]*\])?(.*?)\\end\{algorithm\*?\}", text, re.S):
        body = m.group(2)
        cm = re.search(r"\\caption\*?(?:\[[^\]]*\])?(?=\{)", body)
        count += 1
        if cm:
            e = balanced_end(body, cm.end())
            if e > 0:
                cap = body[cm.end() + 1:e - 1]
                lab = re.search(r"\\label\{([^}]*)\}", cap) or re.search(r"\\label\{([^}]*)\}", body[e:e + 200])
                cap = re.sub(r"\\label\{[^}]*\}", "", cap).strip()
                num = str(count)
                if lab and re.fullmatch(r"[1-9]\d*", str((labels.get(lab.group(1)) or {}).get("num", ""))):
                    num = labels[lab.group(1)]["num"]
                body = body[:cm.start()] + "\n\n\\textbf{Algoritmo " + num + ".} " + cap + "\n\n" + body[e:]
        out.append(text[pos:m.start()] + text[m.start():m.start(2)] + body + text[m.end(2):m.end()])
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)
