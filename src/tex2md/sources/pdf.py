"""PDF -> Markdown body with marker, run in its own isolated uv project (see ``tex2md.config``)."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from tex2md.config import MARKER, MODELS


def marker_body(pdf, mode):
    env = dict(os.environ, MODEL_CACHE_DIR=MODELS,
               HF_HOME=os.path.join(MODELS, "hf"))
    with tempfile.TemporaryDirectory() as tmp:
        # marker 1.x (pure PyTorch; 2.x needs an external llama.cpp server).
        # balanced = OCR every page (better math); fast = OCR only where the text layer is bad.
        cmd = ["uv", "run", "--project", MARKER, "marker_single", pdf, "--output_dir", tmp,
               "--disable_image_extraction"] + (["--force_ocr"] if mode == "balanced" else [])
        p = subprocess.run(cmd, cwd=MARKER, env=env, capture_output=True, text=True)
        mds = [os.path.join(dp, f) for dp, _, fns in os.walk(tmp) for f in fns if f.endswith(".md")]
        if p.returncode != 0 or not mds:
            raise SystemExit("marker failed: " + p.stderr[-800:])
        body = Path(mds[0]).read_text(encoding="utf-8")
    # marker leaves empty image links when image extraction is off
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", body)
    body = re.sub(r"<span id=\"page-\d+-\d+\"></span>", "", body)
    body = re.sub(r"(?m)^(#+) \*\*(.+?)\*\*\s*$", r"\1 \2", body)

    def level(m):
        # marker guesses heading levels; numbered headings get theirs from the number
        text = re.sub(r"\*\*([0-9A-Z][0-9.]*)\*\*", r"\1", m.group(2)).strip()
        n = re.match(r"^([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)+)\.?\s+\S", text)
        depth = n.group(1).count(".") + 1 if n else None
        if depth:
            return "#" * min(depth, 4) + " " + text
        if re.match(r"(?i)^(abstract|references|bibliography|acknowledge?ments?|appendix)\b", text):
            return "# " + text
        return m.group(1) + " " + text
    body = re.sub(r"(?m)^(#+) (.+)$", level, body)
    return tidy_marker(body)


_ROMAN = re.compile(r"^(?=[IVXL]+\.)(L?X{0,3})(IX|IV|V?I{0,3})\.\s+\S")


def _false_heading(text):
    """Text marker set as a heading because it is large or bold: a list item, a row of single letters (a sample of a
    source), or a whole sentence."""
    words = text.split()
    single = sum(len(w) == 1 for w in words)
    return text[:1] in "•·-" or (len(words) >= 4 and single / len(words) >= 0.6) or len(words) > 14


_SUP = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")


def _script(m):
    """``<sup>x</sup>`` / ``<sub>x</sub>`` of a table cell in Markdown: Unicode when every character has a raised or
    lowered form, a formula otherwise."""
    tag, x = m.group(1), m.group(2)
    table, mark = (_SUP, "^") if tag == "sup" else (_SUB, "_")
    plain = x.translate(table)
    if all(c != o for c, o in zip(plain, x)):
        return plain
    if re.search(r"[$\\{}]", x):
        return m.group(0)
    return "$" + mark + "{\\text{" + x + "}}$"


def _table_html(line):
    """The HTML marker writes inside a table row: bold, italics and underline as Markdown, scripts as text."""
    line = re.sub(r"<br\s*/>", "<br>", line)
    line = re.sub(r"<b>(.*?)</b>", r"**\1**", line)
    line = re.sub(r"<i>(.*?)</i>", r"*\1*", line)
    line = re.sub(r"<u>(.*?)</u>", r"\1", line)
    # (a script right after a formula goes into it: "$x$<sup>2</sup>" -> "$x^{2}$", not "$x$$^{..}$")
    line = re.sub(r"\$([^$]+)\$<(su[pb])>([^<$\\]*)</\2>",
                  lambda m: "$" + m.group(1) + ("^" if m.group(2) == "sup" else "_") + "{" + m.group(3) + "}$", line)
    return re.sub(r"<(su[pb])>([^<]*)</\1>", _script, line)


def tidy_marker(body):
    """What marker 1.x gets wrong in its Markdown, fixable without running it again (also on converted .md):
    ``<sup>&</sup>lt;sup>1</sup>`` (a footnote mark escaped twice), the cover page JSTOR adds to its PDFs, heading
    levels taken from the font size where the sections are numbered I., II. with subsections A., B., false
    headings (a list item, a sample of letters, a sentence), and HTML tags in tables (``<b>``, ``<sup>``...; a
    ``<br>`` stays: it is how a Markdown table cell breaks a line)."""
    body = re.sub(r"<(su[pb])>&</\1>lt;\1>", r"<\1>", body)
    head = body[:4000]
    end = head.rfind("JSTOR is an independent not-for-profit")
    if "Stable URL" in head and end >= 0:
        stop = body.find("\n\n", end)
        body = body[stop if stop >= 0 else len(body):].lstrip("\n")
    lines = body.split("\n")
    texts = [re.sub(r"^#+ ", "", ln) for ln in lines]
    roman = sum(1 for ln, t in zip(lines, texts) if ln.startswith("#") and _ROMAN.match(t))
    out, in_roman = [], False
    for ln, t in zip(lines, texts):
        if ln.startswith("|"):
            ln = _table_html(ln)
        if ln.startswith("#") and re.match(r"^#+ ", ln):
            if _false_heading(t):
                ln = t                          # a paragraph again
            elif roman >= 2 and _ROMAN.match(t):
                ln, in_roman = "# " + t, True
            elif roman >= 2 and in_roman and re.match(r"^[A-H]\.\s+\S", t):
                ln = "## " + t
            elif re.match(r"(?i)^(references|bibliography|acknowledge?ments?)\b", t):
                in_roman = False
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))



def tidy_marker_md(text):
    """``tidy_marker`` on a converted .md: only the body marker wrote, after the front matter and the header this
    package adds (title, authors, the "Convertido desde el PDF con marker" note); and without the repeated title the
    JSTOR cover hid from the first conversion."""
    m = re.search(r"(?m)^> Convertido desde el PDF con marker.*\n(?:>.*\n)*\n", text)
    if not m:
        return text
    title = re.search(r'^title: "((?:[^"\\]|\\.)*)"$', text, re.M)
    body = tidy_marker(text[m.end():])
    if title:
        body = re.sub(r"\A\s*#+ " + re.escape(title.group(1)) + r"\s*\n", "", body, flags=re.I)
    return text[:m.end()] + body.lstrip("\n")


def cmd_fixmarker(a):
    for f in a.files:
        t = Path(f).read_text(encoding="utf-8")
        new = tidy_marker_md(t)
        if new != t:
            Path(f).write_text(new, encoding="utf-8")
            print("fixed", f)
    return 0
