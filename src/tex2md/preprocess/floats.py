"""Figure/table environments: normalise them so pandoc keeps every caption."""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end


def tables_without_tabular(text):
    """A \\begin{table} holding only an image (or text) and a caption is dropped by pandoc:
    turn it into a figure and mark the caption so the filter still calls it "Tabla"."""
    envre = re.compile(r"\\begin\{table(\*?)\}")
    out, pos = [], 0
    while True:
        m = envre.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        endm = re.compile(r"\\end\{table" + re.escape(m.group(1)) + r"\}").search(text, m.end())
        if not endm:
            out.append(text[pos:])
            return "".join(out)
        body = text[m.end():endm.start()]
        if (not re.search(r"\\begin\{(?:tabular[x*y]?|longtable|array|NiceTabular\*?|tabulary|subtable|table)", body)
                and "\\caption" in body and "\\includegraphics" in body
                and not re.search(r"\\begin\{(?:tcolorbox|verbatim|lstlisting|minted|algorithm|itemize|enumerate|framed|mdframed)|\\input", body)):
            body = re.sub(r"\\caption(\*?)(\[[^\]]*\])?\s*\{", lambda c: "\\caption" + c.group(1) + (c.group(2) or "") + "{TEX2MDTABLE ", body, count=1)
            out.append(text[pos:m.start()] + "\\begin{figure}" + body + "\\end{figure}")
        elif (not re.search(r"\\begin\{(?:tabular[x*y]?|longtable|array|NiceTabular\*?|tabulary|subtable|table)", body)
                and re.search(r"\\caption\*?(?:\[[^\]]*\])?\s*\{", body)):
            # text-only table (boxes, listings): keep the content, give the caption its own float
            cm = re.search(r"\\caption\*?(?:\[[^\]]*\])?\s*(?=\{)", body)
            ce = balanced_end(body, cm.end())
            if ce > 0:
                cap = body[cm.end() + 1:ce - 1]
                out.append(text[pos:m.start()] + "\\begin{figure}\\caption{TEX2MDTABLE " + cap + "}\\end{figure}\n"
                           + f"\\begin{{table{m.group(1)}}}" + body[:cm.start()] + body[ce:] + text[endm.start():endm.end()])
            else:
                out.append(text[pos:endm.end()])
        else:
            out.append(text[pos:endm.end()])
        pos = endm.end()


def retype_floats(text):
    """A figure whose only caption is \\captionof{table} (wrapfigure holding a table) is a table."""
    envre = re.compile(r"\\begin\{(table|figure)(\*?)\}")
    out, pos = [], 0
    while True:
        m = envre.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        endm = re.compile(r"\\end\{" + m.group(1) + re.escape(m.group(2)) + r"\}").search(text, m.end())
        if not endm:
            out.append(text[pos:])
            return "".join(out)
        body = text[m.end():endm.start()]
        kinds = re.findall(r"\\captionof\s*\{(table|figure)\}", body)
        if len(kinds) == 1 and kinds[0] != m.group(1) and not re.search(r"\\begin\{(?:table|figure)", body):
            k = kinds[0]
            out.append(text[pos:m.start()] + f"\\begin{{{k}}}{body}\\end{{{k}}}")
        else:
            out.append(text[pos:endm.end()])
        pos = endm.end()


def wrap_loose_captions(text):
    """\\captionof{figure|table}{...} outside any float (a teaser under \\maketitle, a table in a
    minipage on its own): pandoc ignores it. Give it a float of its own."""
    cap = re.compile(r"\\captionof\s*\{(table|figure)\}\s*(?:\[[^\]]*\])?\s*(?=\{)")
    tok = re.compile(r"\\(begin|end)\{(?:table|figure)\*?\}")
    pos = 0
    while True:
        m = cap.search(text, pos)
        if not m:
            return text
        depth = sum(1 if t.group(1) == "begin" else -1 for t in tok.finditer(text, 0, m.start()))
        e = balanced_end(text, m.end())
        if depth > 0 or e < 0:
            pos = m.end()
            continue
        start, stop = m.start(), e
        tb = re.compile(r"\s*(\\begin\{(tabular[x*]?|longtable|tabulary)\})")
        wrap = re.compile(r"\\begin\{adjustbox\}(?:\{[^{}]*\})?|\\(?:resizebox|scalebox)\s*\{[^{}]*\}\s*(?:\{[^{}]*\})?\s*\{")
        skip = re.compile(r"(?:\s|\\label\{[^}]*\}|\\vspace\*?\{[^}]*\}|\\centering|\\small|\\footnotesize|%[^\n]*|" + wrap.pattern + r")*").match(text, e)
        nm = tb.match(text, skip.end())
        if nm:  # the table follows its caption
            en = text.find(f"\\end{{{nm.group(2)}}}", nm.end())
            if en > 0:
                stop = en + len(f"\\end{{{nm.group(2)}}}")
                closers = [w.group(0) for w in wrap.finditer(text, e, skip.end())]
                for w in reversed(closers):
                    c = re.compile(r"\s*(?:\\end\{adjustbox\}|\})" if w.startswith("\\begin") is False else r"\s*\\end\{adjustbox\}").match(text, stop)
                    if c:
                        stop = c.end()
        else:   # the table precedes its caption
            pm = re.search(r"\\end\{(tabular[x*]?|longtable|tabulary)\}\s*$", text[:m.start()])
            if pm:
                bg = text.rfind(f"\\begin{{{pm.group(1)}}}", 0, pm.start())
                if bg >= 0:
                    start = bg
        inner = text[start:m.start()] + "\\caption" + text[m.end():e] + text[e:stop]
        rep = f"\\begin{{{m.group(1)}}}{inner}\\end{{{m.group(1)}}}"
        text = text[:start] + rep + text[stop:]
        pos = start + len(rep)


def split_captioned_blocks(text):
    """A float holding several side-by-side minipage/parbox blocks, each with its own \\caption:
    pandoc keeps one caption for the lot. Make one float per block."""
    envre = re.compile(r"\\begin\{(table|figure)(\*?)\}(\[[^\]]*\])?")
    out, pos = [], 0
    while True:
        m = envre.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        endm = re.compile(r"\\end\{" + m.group(1) + re.escape(m.group(2)) + r"\}").search(text, m.end())
        if not endm:
            out.append(text[pos:])
            return "".join(out)
        body = text[m.end():endm.start()]
        spans = []
        for bm in re.finditer(r"\\begin\{minipage\}(?:\[[^\]]*\])?\s*\{[^{}]*\}(.*?)\\end\{minipage\}", body, re.S):
            spans.append((bm.start(), bm.end(), bm.group(1)))
        for bm in re.finditer(r"\\parbox\s*(?:\[[^\]]*\])*\s*\{[^{}]*\}\s*(?=\{)", body):
            e = balanced_end(body, bm.end())
            if e > 0:
                spans.append((bm.start(), e, body[bm.end() + 1:e - 1]))
        spans.sort()
        top, last = [], -1
        for st, en, inner in spans:   # keep only outermost blocks
            if st >= last:
                top.append((st, en, inner))
                last = en
        blocks = [inner for _, _, inner in top]
        capt = [b for b in blocks if "\\caption" in b]
        out.append(text[pos:m.start()])
        # only when the float is nothing but those blocks (plus spacing): never drop content
        rest = body
        for st, en, _ in reversed(top):
            rest = rest[:st] + rest[en:]
        rest = re.sub(r"%[^\n]*", "", rest)
        rest = re.sub(r"\\(?:centering|hfill|hspace|vspace|vskip|hskip|noindent|small|footnotesize|scriptsize|tiny|large|normalsize|quad|qquad|label|smallskip|medskip|bigskip|setlength|captionsetup|par|raggedright|raggedleft|relax|linewidth|textwidth|fill)\*?(?:\[[^\]]*\])?(?:\s*\{[^{}]*\})?", "", rest)
        rest = re.sub(r"-?[0-9.]+\s*(?:in|pt|cm|mm|em|ex|bp)\b|\\[;,:! ]|[\s{}\[\]~]", "", rest)
        clean = len(rest) < 4 and not re.search(r"\\begin\{(?:table|figure)", body)
        if len(capt) >= 2 and len(capt) == len(blocks) and clean:
            env_kind = m.group(1)

            def kind_of(b, default=env_kind):
                k = re.search(r"\\captionof\s*\{(table|figure)\}", b)
                return k.group(1) if k else default
            out.append("".join(f"\\begin{{{kind_of(b)}}}{b}\\end{{{kind_of(b)}}}\n" for b in capt))
        else:
            out.append(text[m.start():endm.end()])
        pos = endm.end()


def flatten_subfloats(text):
    """subcaption's subtable/subfigure: pandoc keeps only one caption for the whole float and
    loses the outer one. Unwrap them and turn each sub-caption into an "(a) ..." line."""
    beg = re.compile(r"\\begin\{(subtable|subfigure)\*?\}(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})?")
    end = re.compile(r"\\end\{(?:subtable|subfigure)\*?\}")
    cap = re.compile(r"\\caption\*?(?:\[[^\]]*\])?\s*(?=\{)")
    flt = re.compile(r"\\begin\{(?:table|figure)\*?\}")
    pos = 0
    while True:
        m = beg.search(text, pos)
        if not m:
            return text
        e = end.search(text, m.end())
        if not e:
            return text
        body = text[m.end():e.start()]
        cm = cap.search(body)
        if cm:
            ce = balanced_end(body, cm.end())
            if ce > 0:
                starts = [x.end() for x in flt.finditer(text, 0, m.start())]
                n = len(re.findall(r"\\par\\textit\{\([a-z0-9]+\) ", text[starts[-1]:m.start()])) if starts else 0
                lab = chr(ord("a") + n) if n < 26 else str(n)
                body = body[:cm.start()] + "\\par\\textit{(" + lab + ") " + body[cm.end() + 1:ce - 1] + "}\\par " + body[ce:]
        text = text[:m.start()] + body + text[e.end():]
        pos = m.start() + len(body)


def unnest_tables(text):
    """\\begin{table} wrapping minipages that hold their own \\begin{table}: drop the outer one."""
    tok = re.compile(r"\\(begin|end)\{table\*?\}(?:\[[^\]]*\])?")
    while True:
        stack, hit = [], None
        for m in tok.finditer(text):
            if m.group(1) == "begin":
                if stack:
                    hit = (stack[0], m)
                    break
                stack.append(m)
            elif stack:
                stack.pop()
        if not hit:
            return text
        outer = hit[0]
        depth, end = 0, None
        for m in tok.finditer(text, outer.start()):
            depth += 1 if m.group(1) == "begin" else -1
            if depth == 0:
                end = m
                break
        if end is None:
            return text
        text = text[:outer.start()] + text[outer.end():end.start()] + text[end.end():]


def wrap_envs_to_plain(text):
    """wraptable/wrapfigure (with their width arguments) -> table/figure."""
    text = re.sub(r"\\begin\{(wraptable|wrapfigure)\}(\{[^}]*\}|\[[^\]]*\])*", lambda m: "\\begin{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    text = re.sub(r"\\end\{(wraptable|wrapfigure)\}", lambda m: "\\end{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    return text


def strip_float_layout(text):
    """minipage / parbox / SCfigure wrappers around float contents."""
    text = re.sub(r"\\(?:begin|end)\{minipage\}(?:\[[^\]]*\])?(?:\s*\{[^{}]*\})?", "", text)
    text = re.sub(r"\\parbox\s*(?:\[[^\]]*\])*\s*\{[^{}]*\}\s*(?=\{)", "", text)  # \\parbox{w}{...} -> {...}
    text = re.sub(r"\\(begin|end)\{SC(figure|table)\}(?:\s*\[[^\]]*\])*", r"\\\1{\2}", text)
    return text


def starred_floats_to_plain(text):
    """table* / figure* -> table / figure."""
    return re.sub(r"\\(begin|end)\{(table|figure)\*\}", r"\\\1{\2}", text)


def captionof_to_caption(text):
    return re.sub(r"\\captionof\{(table|figure)\}", r"\\caption", text)
