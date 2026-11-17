"""Figure/table environments: normalise them so pandoc keeps every caption."""
from __future__ import annotations

import re

from tex2md.util.tex import balanced_end


def unwrap_algorithm_floats(text):
    r"""A ``table``/``figure`` used only to put two ``algorithm`` environments side by side has no caption of
    its own: drop it, or the float rules below take the caption of the first algorithm for the float's."""
    envre = re.compile(r"\\begin\{(table|figure)(\*?)\}(?:\[[^\]]*\])?")
    out, pos = [], 0
    while True:
        m = envre.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        close = "\\end{" + m.group(1) + m.group(2) + "}"
        end = text.find(close, m.end())
        if end < 0:
            out.append(text[pos:])
            return "".join(out)
        body = text[m.end():end]
        own = re.sub(r"\\begin\{algorithm\*?\}.*?\\end\{algorithm\*?\}", "", body, flags=re.S)
        if own != body and "\\caption" not in own and not re.search(r"\\begin\{(?:table|figure|tabular)", own):
            out.append(text[pos:m.start()] + body)
        else:
            out.append(text[pos:end + len(close)])
        pos = end + len(close)


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


_CAPTIONOF_KIND = re.compile(r"\\captionof\s*\{(table|figure)\}")
# (\captionsetup[sub]{type=...} is about the sub-captions, not the caption of the float)
_SETUP_TYPE = re.compile(r"\\captionsetup\s*(?:\[(?:table|figure)\])?\s*\{[^{}]*?\btype\s*=\s*(table|figure)\b")


def declared_kind(body):
    r"""The kind a block says its caption is of: the first ``\captionof{table|figure}``, else the ``type=`` of the
    ``\captionsetup`` ahead of its ``\caption`` (when they all agree); None when it says nothing."""
    k = _CAPTIONOF_KIND.search(body)
    if k:
        return k.group(1)
    setup = set(_SETUP_TYPE.findall(body))
    return setup.pop() if len(setup) == 1 else None


def retype_floats(text):
    r"""A float whose only caption is declared to be of the other kind is of that kind: a figure whose caption is
    ``\captionof{table}`` (wrapfigure holding a table), or a figure with ``\captionsetup{type=table}`` before its
    ``\caption`` (the compiled document numbers it as a table, and the numbering below follows the .aux)."""
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
        only_caption = len(re.findall(r"\\caption(?:of)?(?![a-zA-Z@])", body)) == 1       # the only caption the float has
        declared = declared_kind(body) if only_caption else None
        if declared and only_caption and declared != m.group(1) and not re.search(r"\\begin\{(?:table|figure)", body):
            out.append(text[pos:m.start()] + f"\\begin{{{declared}}}{body}\\end{{{declared}}}")
        else:
            out.append(text[pos:endm.end()])
        pos = endm.end()


def dissolve_multi_caption_floats(text):
    r"""A float holding several captions (a table and a figure sharing one ``wrapfigure``, or two ``\caption`` one after
    the other in the same ``figure``): pandoc keeps one caption for the lot. Drop the float itself so each caption
    becomes a loose ``\captionof`` that ``wrap_loose_captions`` can pair with its own content. Floats made of
    minipage/parbox blocks are left to ``split_captioned_blocks``, those with sub-floats to ``flatten_subfloats``."""
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
        n_of = len(re.findall(r"\\captionof\s*\{", body))
        n_caption = len(re.findall(r"\\caption(?![a-zA-Z@*])", body))
        nested = re.search(r"\\begin\{(?:table|figure|minipage|subfigure|subtable|subfloat|wrapfigure|wraptable)|\\parbox"
                           r"|\\subfloat|\\subfigure|\\subcaptionbox", body)
        if n_of + n_caption >= 2 and (n_of >= 2 or n_caption >= 2) and not nested:
            # each \caption left alone in the body would be taken for "the" caption of the float: make them all loose
            kind = m.group(1)
            body = re.sub(r"\\caption(?![a-zA-Z@*])", lambda _m, kind=kind: "\\captionof{" + kind + "}", body)
            out.append(text[pos:m.start()] + body)
        else:
            out.append(text[pos:endm.end()])
        pos = endm.end()


_TABLE_ENVS = r"tabular[x*y]?|longtable|NiceTabular\*?|NiceTabularX"
_LAYOUT = (r"(?:\s|%[^\n]*|\\vspace\*?\{[^{}]*\}|\\label\{[^{}]*\}|\\centering|\\small|\\footnotesize|\\scriptsize|\\tiny|\\normalsize"
           r"|\\setlength\s*\{[^{}]*\}\s*\{[^{}]*\}|\\captionsetup\s*(?:\[[^\]]*\])?\s*\{[^{}]*\})")
_CLOSER = r"\\end\{(center|adjustbox|small|footnotesize|scriptsize|tiny|flushleft|flushright)\}"
# the opening of a group that wraps a tabular: \resizebox{\textwidth}{!}{ ... } or a bare brace
_OPEN_GROUP = r"(?:\\(?:resizebox\*?|scalebox|adjustbox|makebox|raisebox)\s*(?:\[[^\]]*\]\s*)*(?:\{[^{}]*\}\s*){0,2}\{|\{)"


def _table_before(text, caption_at):
    """Where the tabular that a loose caption belongs to begins, when it sits right before the caption (only
    layout in between, even a ``center`` or a ``\\resizebox{..}{!}{`` that closes after the tabular): the index of
    its ``\\begin`` (and of the wrappers around it), or None."""
    pm = re.search(r"\\end\{(" + _TABLE_ENVS + r")\}((?:" + _LAYOUT + "|" + _CLOSER + r"|\})*)$", text[:caption_at])
    if not pm:
        return None
    bg = text.rfind(f"\\begin{{{pm.group(1)}}}", 0, pm.start())
    if bg < 0:
        return None
    # (the braces of \vspace{..} and the like are layout, not groups that close: take those out first)
    for closer in re.finditer(r"\}|" + _CLOSER, re.sub(_LAYOUT, "", pm.group(2))):
        if closer.group(0) == "}":
            opener = re.search(_OPEN_GROUP + r"(?:" + _LAYOUT + r")*$", text[:bg])
        else:
            opener = re.search(r"\\begin\{" + closer.group(1) + r"\}(?:\[[^\]]*\])?(?:\{[^{}]*\})?(?:" + _LAYOUT + r")*$", text[:bg])
        if not opener:
            return None            # a closer with no opener in front of the tabular: it belongs to something bigger
        bg = opener.start()
    return bg


def _caption_follows(text, i):
    """True when only layout (and the braces of ``\\resizebox``) lie between ``i`` and the next loose ``\\captionof``."""
    return bool(re.compile(r"(?:" + _LAYOUT + "|" + _CLOSER + r"|\})*\\captionof\s*\{(?:table|figure)\}").match(text, i))


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
        tb = re.compile(r"\s*(\\begin\{(" + _TABLE_ENVS + r")\})")
        wrap = re.compile(r"\\begin\{adjustbox\}(?:\{[^{}]*\})?|\\(?:resizebox|scalebox)\s*\{[^{}]*\}\s*(?:\{[^{}]*\})?\s*\{")
        skip = re.compile(r"(?:\s|\\label\{[^}]*\}|\\vspace\*?\{[^}]*\}|\\centering|\\small|\\footnotesize|\\scriptsize|\\tiny|\\normalsize|\\setlength\s*\{[^{}]*\}\s*\{[^{}]*\}|\\captionsetup\s*\{[^{}]*\}|%[^\n]*|" + wrap.pattern + r")*").match(text, e)
        nm = tb.match(text, skip.end())
        before = _table_before(text, m.start())
        en = text.find(f"\\end{{{nm.group(2)}}}", nm.end()) if nm else -1
        if nm and en > 0 and before is not None and _caption_follows(text, en + len(f"\\end{{{nm.group(2)}}}")):
            nm = None          # "table, caption, table, caption": the table that follows belongs to the caption after it
        if nm:  # the table follows its caption
            if en > 0:
                stop = en + len(f"\\end{{{nm.group(2)}}}")
                closers = [w.group(0) for w in wrap.finditer(text, e, skip.end())]
                for w in reversed(closers):
                    c = re.compile(r"\s*(?:\\end\{adjustbox\}|\})" if w.startswith("\\begin") is False else r"\s*\\end\{adjustbox\}").match(text, stop)
                    if c:
                        stop = c.end()
        else:   # the table precedes its caption
            start = start if before is None else before          # (0 is a position, not "nothing")
        inner = text[start:m.start()] + "\\caption" + text[m.end():e] + text[e:stop]
        rep = f"\\begin{{{m.group(1)}}}{inner}\\end{{{m.group(1)}}}"
        text = text[:start] + rep + text[stop:]
        pos = start + len(rep)


def _picture_only(block):
    """True when a block holds nothing but pictures and layout (so dropping it loses no text)."""
    rest = re.sub(r"%[^\n]*|\\includegraphics\*?\s*(?:\[(?:[^\[\]{}]|\{[^{}]*\})*\])?\s*\{[^{}]*\}|\\label\s*\{[^{}]*\}", "", block)
    rest = re.sub(r"\\(?:centering|hfill|hspace\*?|vspace\*?|vskip|hskip|noindent|par|smallskip|medskip|bigskip|quad|qquad|raggedright|linewidth|textwidth)"
                  r"(?:\s*\{[^{}]*\})?|-?[0-9.]+\s*(?:in|pt|cm|mm|em|ex|bp)\b|[\s{}\[\]~]", "", rest)
    return not rest


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
        others = [b for b in blocks if "\\caption" not in b]
        out.append(text[pos:m.start()])
        # only when the float is nothing but those blocks (plus spacing): never drop content
        rest = body
        for st, en, _ in reversed(top):
            rest = rest[:st] + rest[en:]
        rest = re.sub(r"%[^\n]*", "", rest)
        rest = re.sub(r"\\(?:centering|hfill|hspace|vspace|vskip|hskip|noindent|small|footnotesize|scriptsize|tiny|large|normalsize|quad|qquad|label|smallskip|medskip|bigskip|setlength|captionsetup|par|raggedright|raggedleft|relax|linewidth|textwidth|fill)\*?(?:\[[^\]]*\])?(?:\s*\{[^{}]*\})?", "", rest)
        rest = re.sub(r"-?[0-9.]+\s*(?:in|pt|cm|mm|em|ex|bp)\b|\\[;,:! ]|[\s{}\[\]~]", "", rest)
        clean = len(rest) < 4 and not re.search(r"\\begin\{(?:table|figure)", body)
        # blocks with no caption may be the pictures the captions below them describe: nothing to keep in them
        if len(capt) >= 2 and all(_picture_only(b) for b in others) and clean:
            env_kind = m.group(1)

            def kind_of(b, default=env_kind):
                return declared_kind(b) or default
            out.append("".join(f"\\begin{{{kind_of(b)}}}{b}\\end{{{kind_of(b)}}}\n" for b in capt))
        else:
            out.append(text[m.start():endm.end()])
        pos = endm.end()


def _aux_key(tex):
    """The start of a caption as ``parse_floats`` keeps it: letters and digits only, 40 of them."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\\[a-zA-Z@]+\s*", " ", tex).lower())[:40]


def _parent_has_caption(text, at):
    """True when the float that holds the sub-float starting at ``at`` has a caption of its own (outside the sub-floats)."""
    opened = []
    for t in re.finditer(r"\\(begin|end)\{(figure|table)(\*?)\}", text[:at]):
        if t.group(1) == "begin":
            opened.append(t)
        elif opened:
            opened.pop()
    if not opened:
        return False
    parent = opened[-1]
    close = text.find("\\end{" + parent.group(2) + parent.group(3) + "}", at)
    body = text[parent.end():close if close >= 0 else len(text)]
    # (what is inside the sub-floats, and inside floats already made out of earlier ones, is not the float's own)
    body = re.sub(r"\\begin\{(sub(?:figure|table)|figure|table)(\*?)\}.*?\\end\{\1\2\}", "", body, flags=re.S)
    return bool(re.search(r"\\caption(?![a-zA-Z@*])", body))


def flatten_subfloats(text, floats=None):
    """subcaption's subtable/subfigure: pandoc keeps only one caption for the whole float and
    loses the outer one. Unwrap them and turn each sub-caption into an "(a) ..." line.

    A sub-caption that the compiled document numbers like a caption of its own (a "Figure 15" in the .aux whose text
    is that of the sub-caption: the style makes the panels figures) stays a caption: it becomes a float."""
    numbered = {(f["kind"], f["text"][:20]) for f in (floats or []) if f["kind"] in ("figure", "table")}
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
                caption = body[cm.end() + 1:ce - 1]
                kind = "table" if m.group(1) == "subtable" else "figure"
                if (kind, _aux_key(caption)[:20]) in numbered and _aux_key(caption) and not _parent_has_caption(text, m.start()):
                    # a figure of its own: its content and caption in a float, which the enclosing one gives way to
                    text = (text[:m.start()] + "\\begin{" + kind + "}" + body + "\\end{" + kind + "}" + text[e.end():])
                    pos = m.start() + len(body) + 2 * len(kind) + 12
                    continue
                starts = [x.end() for x in flt.finditer(text, 0, m.start())]
                n = len(re.findall(r"\\par\\textit\{\([a-z0-9]+\) ", text[starts[-1]:m.start()])) if starts else 0
                lab = chr(ord("a") + n) if n < 26 else str(n)
                body = body[:cm.start()] + "\\par\\textit{(" + lab + ") " + caption + "}\\par " + body[ce:]
        text = text[:m.start()] + body + text[e.end():]
        pos = m.start() + len(body)


def unnest_tables(text, env="table"):
    """\\begin{table} wrapping minipages that hold their own \\begin{table}: drop the outer one.

    ``env`` is "table" or "figure" (a figure whose minipages hold figures of their own)."""
    tok = re.compile(r"\\(begin|end)\{" + env + r"\*?\}(?:\[[^\]]*\])?")
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


def unnest_floats(text):
    """Drop the outer float when it only wraps floats of the same kind (tables and figures)."""
    return unnest_tables(unnest_tables(text, "table"), "figure")


def wrap_envs_to_plain(text):
    """wraptable/wrapfigure (with their width arguments) -> table/figure."""
    text = re.sub(r"\\begin\{(wraptable|wrapfigure)\}(\{[^}]*\}|\[[^\]]*\])*", lambda m: "\\begin{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    text = re.sub(r"\\end\{(wraptable|wrapfigure)\}", lambda m: "\\end{" + ("table" if m.group(1) == "wraptable" else "figure") + "}", text)
    return text


def strip_float_layout(text):
    """minipage / parbox / SCfigure wrappers around float contents."""
    # only \begin takes arguments: a group after \end{minipage} is text ("\end{minipage}{\small Source: ...}")
    text = re.sub(r"\\begin\{minipage\}(?:\s*\[[^\]]*\])*(?:\s*\{[^{}]*\})?|\\end\{minipage\}", "", text)
    text = re.sub(r"\\parbox\s*(?:\[[^\]]*\])*\s*\{[^{}]*\}\s*(?=\{)", "", text)  # \\parbox{w}{...} -> {...}
    text = re.sub(r"\\(begin|end)\{SC(figure|table)\}(?:\s*\[[^\]]*\])*", r"\\\1{\2}", text)
    return text


def starred_floats_to_plain(text):
    """table* / figure* -> table / figure."""
    return re.sub(r"\\(begin|end)\{(table|figure)\*\}", r"\\\1{\2}", text)


def captionof_to_caption(text):
    return re.sub(r"\\captionof\{(table|figure)\}", r"\\caption", text)


_LIST = re.compile(r"\\begin\{(itemize|enumerate|description)\}(?:\s*\[(?:[^\[\]{}]|\{[^{}]*\})*\])?(.*?)\\end\{\1\}", re.S)


def _inline_tabulars(arg):
    r"""A tabular inside a caption (a legend of symbols, say): its cells read one after the other, "; " apart."""
    pos = 0
    head = re.compile(r"\\begin\{tabular\}\s*(?:\[[a-z]\])?\s*(?=\{)")
    while True:
        m = head.search(arg, pos)
        if not m:
            return arg
        spec_end = balanced_end(arg, m.end())
        end = arg.find("\\end{tabular}", spec_end) if spec_end > 0 else -1
        if end < 0:
            return arg
        cells = re.split(r"&|\\\\(?:\[[^\]]*\])?", arg[spec_end:end])
        text = "; ".join(c.strip() for c in cells if c.strip())
        arg = arg[:m.start()] + text + arg[end + len("\\end{tabular}"):]
        pos = m.start() + len(text)


def _inline_lists(arg):
    r"""A list inside a caption (pandoc cannot read block environments there): its items become " - item" runs."""
    arg = _inline_tabulars(arg)
    def one(m):
        count = 0
        body = m.group(2)

        def item(im):
            nonlocal count
            count += 1
            label = im.group(1)
            if label:
                return f" {label[1:-1]} "
            return f" ({count}) " if m.group(1) == "enumerate" else " \u2022 "
        return re.sub(r"\\item(?![a-zA-Z@])\s*(\[[^\]]*\])?", item, body)
    return _LIST.sub(one, arg)


_CAPTION_HEAD = re.compile(r"\\(?:caption|captionof\s*\{[a-z]+\}|subcaption)\*?\s*(?:\[(?:[^\[\]{}]|\{[^{}]*\})*\])?\s*(?=\{)")


def single_paragraph_captions(text):
    r"""A paragraph break (a blank line, ``\par``) inside ``\caption{...}``: pandoc fails with "unexpected ()" and the whole
    figure or table, with its numbers, is lost. The caption is printed as one paragraph anyway, so the break becomes a
    blank."""
    out, pos = [], 0
    while True:
        m = _CAPTION_HEAD.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        e = balanced_end(text, m.end())
        if e < 0:
            out.append(text[pos:])
            return "".join(out)
        arg = _inline_lists(text[m.end():e])
        fixed = re.sub(r"\n[ \t]*(?:\n[ \t]*)+|\\par(?![a-zA-Z@])\s*", " ", arg)
        out.append(text[pos:m.end()] + fixed)
        pos = e
