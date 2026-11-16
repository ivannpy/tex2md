"""arXiv LaTeX source -> Markdown, as a sequence of explicit stages.

    fetch -> compile (.aux/.bbl) -> flatten -> preprocess -> pandoc -> postprocess -> write

Every stage reads and fills a ``Workspace`` and the JSON quality ``report``; a stage that
cannot continue sets ``report["error"]`` and the conversion stops there.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass

from tex2md.build.aux import parse_aux, parse_floats
from tex2md.build.latex import find_main
from tex2md.config import CACHE, FILTER
from tex2md.library.frontmatter import frontmatter
from tex2md.postprocess.markdown import (
    cap_heading_levels, fix_text_math, isolate_display_math, normalize_math_macros)
from tex2md.preprocess.citations import parse_bbl
from tex2md.preprocess.macros import local_style_macros
from tex2md.preprocess.pipeline import preprocess
from tex2md.preprocess.tables import escape_citation_ampersands
from tex2md.sources.arxiv import arxiv_meta, fetch_source
from tex2md.util.io import read, run

ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$")

# pandoc's Markdown dialect: plain text, GitHub-style pipe tables, no attribute noise
PANDOC_FORMAT = ("markdown-raw_attribute-header_attributes-fenced_divs-bracketed_spans-native_divs"
                 "-native_spans-link_attributes-inline_code_attributes-simple_tables-multiline_tables"
                 "-grid_tables-smart-auto_identifiers-implicit_figures")


@dataclass
class Workspace:
    """Directories and files of one paper inside the cache."""
    work: str
    src: str
    build: str
    main: str = ""
    bdir: str = ""
    base: str = ""


def parse_arxiv_arg(arg):
    m = ARXIV_ID.match(arg)
    if not m:
        raise SystemExit("not an arXiv id: " + arg)
    return m.group(1), m.group(2)


def fetch(aid, ver, report) -> Workspace | None:
    work = os.path.join(CACHE, (aid + ver).replace("/", "_"))
    os.makedirs(work, exist_ok=True)
    if not fetch_source(aid + ver, work):
        report["error"] = "arXiv has no LaTeX source (PDF only)"
        return None
    ws = Workspace(work=work, src=os.path.join(work, "src"), build=os.path.join(work, "build"))
    ws.main = find_main(ws.src)
    if not ws.main:
        report["error"] = "no main .tex found"
        return None
    if not os.path.isdir(ws.build):
        shutil.copytree(ws.src, ws.build)
    rel_main = os.path.relpath(ws.main, ws.src)
    ws.bdir = os.path.join(ws.build, os.path.dirname(rel_main))
    ws.base = os.path.splitext(os.path.basename(rel_main))[0]
    return ws


def compile_latex(ws: Workspace, report):
    """Compile once for the .aux/.bbl (errors tolerated: a partial .aux still helps)."""
    if not os.path.exists(os.path.join(ws.bdir, ws.base + ".aux")):
        rc, _, _ = run(["latexmk", "-pdf", "-f", "-interaction=nonstopmode", "-silent", ws.base + ".tex"], ws.bdir, 600)
        report["compile_rc"] = rc
    report["compiled_pdf"] = os.path.exists(os.path.join(ws.bdir, ws.base + ".pdf"))


def locate_bbl(ws: Workspace):
    candidates = (os.path.join(ws.bdir, ws.base + ".bbl"), os.path.join(os.path.dirname(ws.main), ws.base + ".bbl"))
    bbl_path = next((p for p in candidates if os.path.exists(p)), None)
    if not bbl_path:
        # the arXiv .bbl may be named after another file (e.g. neurips_2026.bbl next to atula.tex)
        bbls = [os.path.join(dp, f) for dp, _, fns in os.walk(ws.src) for f in fns if f.endswith(".bbl")]
        bbl_path = bbls[0] if len(bbls) == 1 else None
    return bbl_path


def flatten(ws: Workspace):
    rc, flat, _ = run(["latexpand", "--empty-comments", ws.base + ".tex"], ws.bdir, 120)
    if rc != 0 or not flat.strip():
        flat = read(os.path.join(ws.bdir, ws.base + ".tex"))
    return flat


def load_bibliography(flat, bbl_path):
    inline = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", flat, re.S)
    return parse_bbl(inline.group(0) if inline else read(bbl_path)) if (inline or bbl_path) else None


def run_pandoc(ws: Workspace, tex, labels, bib, report):
    """Write the preprocessed LaTeX and the numbering tables, run pandoc; returns (body, stderr)."""
    # pandoc runs in an empty directory: it would otherwise try to parse local .sty files
    pdir = os.path.join(ws.work, "pandoc")
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "_tex2md_flat.tex"), "w") as fh:
        fh.write(tex)
    lab_path = os.path.join(ws.work, "labels.json")
    with open(lab_path, "w") as fh:
        json.dump(labels, fh)
    floats_path = os.path.join(ws.work, "floats.json")
    with open(floats_path, "w") as fh:
        json.dump(parse_floats(ws.build), fh)
    cmd = ["pandoc", "_tex2md_flat.tex", "-f", "latex", "-t", PANDOC_FORMAT, "--wrap=none",
           "--lua-filter", FILTER, "-M", "tex2md_labels=" + lab_path,
           "-M", "tex2md_floats=" + floats_path]
    bibfiles = [os.path.join(dp, f) for dp, _, fns in os.walk(ws.build) for f in fns if f.endswith(".bib")]
    if not bib and bibfiles:
        cmd += ["--citeproc"] + [a for b in bibfiles for a in ("--bibliography", b)]
        report["citations"] = "citeproc"
    rc, body, err = run(cmd, pdir, 600)
    if rc != 0 or not body.strip():
        report["error"] = "pandoc failed: " + err.strip()[-500:]
        return None, err
    return body, err


def postprocess(body):
    # leftovers of \maketitle/\thanks redefinitions
    body = re.sub(r"(?m)^[a-z ]*\bmaketitle\b[a-z ]*$", "", body)
    body = isolate_display_math(fix_text_math(normalize_math_macros(cap_heading_levels(body))))
    return re.sub(r"\n{3,}", "\n\n", body)


def render_document(meta, aid, ver, mn, body):
    head = [frontmatter(meta, ver, mn), "# " + meta["title"], "",
            "**Autores:** " + "; ".join(meta["authors"]), ""]
    if meta["journal_ref"]:
        head += ["**Publicado en:** " + meta["journal_ref"], ""]
    head += [f"> arXiv:{aid}{ver} · https://arxiv.org/abs/{aid}",
             "> Convertido desde la fuente LaTeX de arXiv con tex2md (pandoc). Numeración de secciones, "
             "figuras, tablas y ecuaciones tomada de la compilación del artículo; las figuras se reducen a su leyenda. "
             "Ante cualquier duda, el PDF es la referencia.", ""]
    if "## Abstract" not in body[:3000]:
        head += ["## Abstract", "", meta["abstract"], ""]
    return "\n".join(head) + "\n" + body.strip() + "\n"


def convert(arg, out, mn):
    """Convert one arXiv paper (id with optional version) to ``out``; returns the quality report."""
    aid, ver = parse_arxiv_arg(arg)
    meta = arxiv_meta(aid)
    ver = ver or meta["latest"]
    report = {"arxiv": aid + ver, "out": out, "ok": False}

    ws = fetch(aid, ver, report)
    if ws is None:
        return report
    compile_latex(ws, report)
    labels = parse_aux(ws.build)
    report["labels"] = len(labels)
    flat = flatten(ws)
    bib = load_bibliography(flat, locate_bbl(ws))
    report["bib_entries"] = len(bib) if bib else 0

    stats = {"unresolved_refs": 0, "unresolved_cites": 0}
    defs = local_style_macros(flat, ws.bdir)
    if defs:
        at = flat.find("\\usepackage")
        flat = flat[:at] + defs + "\n" + flat[at:] if at >= 0 else defs + "\n" + flat
    tex = escape_citation_ampersands(preprocess(flat, labels, bib, stats))
    report.update(stats)
    report["tables_in_tex"] = len(re.findall(r"\\begin\{table\*?\}", tex))

    body, err = run_pandoc(ws, tex, labels, bib, report)
    if body is None:
        return report
    mm = re.search(r"tex2md-stats figures=(\d+) tables=(\d+)", err)
    if mm:
        report["figures_out"], report["tables_out"] = int(mm.group(1)), int(mm.group(2))
    body = postprocess(body)
    report["unresolved_left"] = len(re.findall(r"\[[a-zA-Z]+:[^\]\s]+\]", body)) + body.count("??")

    with open(out, "w") as fh:
        fh.write(render_document(meta, aid, ver, mn, body))
    report["ok"] = True
    with open(os.path.join(ws.work, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report
