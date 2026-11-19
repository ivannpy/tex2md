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

from tex2md.build.aux import parse_aux, parse_floats, parse_sections, use_long_captions
from tex2md.build.latex import declared_toplevel, find_main, main_candidates
from tex2md.config import CACHE, FILTER, UNWRAP_FILTER
from tex2md.diagnostics import (
    Diagnostic,
    counts,
    from_pandoc_stderr,
    main_file_diagnostics,
    scan_lost_text,
    scan_markdown,
    scan_missing_sections,
)
from tex2md.library.frontmatter import frontmatter
from tex2md.pandoc_recovery import convert_with_recovery
from tex2md.pandoc_recovery import diagnostics as dropped_diagnostics
from tex2md.postprocess.markdown import (
    cap_heading_levels,
    drop_counter_hacks,
    fix_text_math,
    isolate_display_math,
    mathml_to_tex,
    normalize_math_macros,
    protect_code,
    resolve_ifthenelse,
    restore_code,
    restore_math_text,
    tidy_math,
)
from tex2md.preprocess.braces import BraceRepair, repair_braces
from tex2md.preprocess.citations import parse_bbl
from tex2md.preprocess.comments import empty_comments, expand_remaining_inputs, inline_listing_files
from tex2md.preprocess.equations import label_numbered_equations
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
    candidates: tuple[str, ...] = ()     # complete documents of the tree when no 00README names the main one
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
    cands = main_candidates(ws.src)
    ws.main = find_main(ws.src, cands)
    if not declared_toplevel(ws.src):
        ws.candidates = tuple(os.path.relpath(c, ws.src) for c in cands)
    if not ws.main:
        report["error"] = "no main .tex found (no file with \\documentclass and \\begin{document}: plain TeX and " \
                          "LaTeX 2.09 sources are not supported)"
        return None
    if not os.path.isdir(ws.build):
        shutil.copytree(ws.src, ws.build)
    rel_main = os.path.relpath(ws.main, ws.src)
    ws.bdir = os.path.join(ws.build, os.path.dirname(rel_main))
    ws.base = os.path.splitext(os.path.basename(rel_main))[0]
    report["main"] = rel_main
    return ws


EQUATION_MARK = ".tex2md-equation-labels"


def label_equations(ws: Workspace):
    """Give every unlabelled numbered equation a label of ours (once per cached build), then drop the stale
    auxiliary files so the next compile writes all the equation numbers to the .aux."""
    mark = os.path.join(ws.build, EQUATION_MARK)
    if os.path.exists(mark):
        return
    counter = 1
    for dp, _, fns in sorted(os.walk(ws.build)):
        for fn in sorted(fns):
            if fn.endswith(".tex"):
                path = os.path.join(dp, fn)
                # surrogateescape: a latin-1 source goes back to disk byte for byte, only the labels are added
                with open(path, encoding="utf-8", errors="surrogateescape") as fh:
                    text = fh.read()
                new, counter = label_numbered_equations(text, start=counter)
                if new != text:
                    with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
                        fh.write(new)
    for dp, _, fns in os.walk(ws.build):
        for fn in fns:
            if fn.endswith((".aux", ".pdf", ".fdb_latexmk", ".fls")):
                os.remove(os.path.join(dp, fn))
    with open(mark, "w") as fh:
        fh.write(str(counter - 1))


def latexmk_command(base: str) -> list[str]:
    """latexmk call for one paper. The source is untrusted: ``-no-shell-escape`` stops ``\\write18`` from running
    even the commands TeX Live whitelists (``shell_escape=p``)."""
    return ["latexmk", "-pdf", "-f", "-interaction=nonstopmode", "-silent",
            "-pdflatex=pdflatex -no-shell-escape -interaction=nonstopmode %O %S", base + ".tex"]


def compile_latex(ws: Workspace, report):
    """Compile once for the .aux/.bbl (errors tolerated: a partial .aux still helps)."""
    if not os.path.exists(os.path.join(ws.bdir, ws.base + ".aux")):
        rc, _, _ = run(latexmk_command(ws.base), ws.bdir, 600)
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
    """One .tex with every \\input resolved and the comments emptied. latexpand would strip "%" inside verbatim
    too (``if i % 2 == 1:`` in a listing), so it keeps the comments and ``empty_comments`` removes them."""
    rc, flat, _ = run(["latexpand", "--keep-comments", "--empty-comments", ws.base + ".tex"], ws.bdir, 120)
    if rc != 0 or not flat.strip():
        flat = read(os.path.join(ws.bdir, ws.base + ".tex"))
    return expand_remaining_inputs(empty_comments(flat), ws.bdir)


def load_bibliography(flat, bbl_path):
    inline = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", flat, re.S)
    return parse_bbl(inline.group(0) if inline else read(bbl_path)) if (inline or bbl_path) else None


def run_pandoc(ws: Workspace, tex, labels, floats, sections, bib, report):
    """Write the numbering tables and run pandoc (retrying without the environments it cannot parse).
    Returns (body or None, last stderr, dropped environments)."""
    # pandoc runs in an empty directory: it would otherwise try to parse local .sty files
    pdir = os.path.join(ws.work, "pandoc")
    os.makedirs(pdir, exist_ok=True)
    lab_path = os.path.join(ws.work, "labels.json")
    with open(lab_path, "w") as fh:
        json.dump(labels, fh)
    floats_path = os.path.join(ws.work, "floats.json")
    with open(floats_path, "w") as fh:
        json.dump(floats, fh)
    sections_path = os.path.join(ws.work, "sections.json")
    with open(sections_path, "w") as fh:
        json.dump(sections, fh)
    cmd = ["pandoc", "_tex2md_flat.tex", "-f", "latex", "-t", PANDOC_FORMAT, "--wrap=none",
           "--syntax-highlighting=none", "--lua-filter", FILTER, "-M", "tex2md_labels=" + lab_path,
           "-M", "tex2md_floats=" + floats_path, "-M", "tex2md_sections=" + sections_path]
    bibfiles = [os.path.join(dp, f) for dp, _, fns in os.walk(ws.build) for f in fns if f.endswith(".bib")]
    if not bib and bibfiles:
        # citeproc adds its bibliography <div> after our filter: unwrap it with a second pass
        cmd += ["--citeproc"] + [a for b in bibfiles for a in ("--bibliography", b)] + ["--lua-filter", UNWRAP_FILTER]
        report["citations"] = "citeproc"

    def run_once(text):
        with open(os.path.join(pdir, "_tex2md_flat.tex"), "w") as fh:
            fh.write(text)
        return run(cmd, pdir, 600)

    body, err, dropped, error = convert_with_recovery(tex, run_once)
    report["dropped_environments"] = [vars(d) for d in dropped]
    if body is None:
        if error.startswith("pandoc produced an empty document") and "\\includepdf" in tex:
            error += " (the arXiv source only wraps a PDF with \\includepdf: convert the PDF with `tex2md pdf`)"
        report["error"] = error
        return None, err, dropped
    return body, err, dropped


def latex_reference_diagnostics(stats) -> list[Diagnostic]:
    """\\ref / \\cite keys the preprocessor could not find in the .aux / .bbl."""
    out = []
    if stats.get("unresolved_refs"):
        out.append(Diagnostic("unresolved-latex-reference", "error",
                              "\\ref/\\cref to a label that is not in the .aux", stats["unresolved_refs"]))
    if stats.get("unresolved_cites"):
        out.append(Diagnostic("unresolved-latex-citation", "warning",
                              "\\cite of a key that is not in the bibliography", stats["unresolved_cites"]))
    return out


def brace_diagnostics(repairs: list[BraceRepair]) -> list[Diagnostic]:
    """Braces of the paper's own LaTeX that did not pair up and were repaired so that pandoc could read it."""
    if not repairs:
        return []
    return [Diagnostic("unbalanced-braces", "warning",
                       "braces that do not pair up in the LaTeX source: a group was closed at the end of its paragraph "
                       "or a stray } was dropped (check the text around it)", len(repairs),
                       [f"{r.kind} at line {r.line}: {r.context}" for r in repairs[:3]])]


def postprocess(body):
    # leftovers of \maketitle/\thanks redefinitions
    body = re.sub(r"(?m)^[a-z ]*\bmaketitle\b[a-z ]*$", "", body)
    # MathML first: it sits in HTML tables, whose backticks must not be taken for inline code below
    body, code = protect_code(mathml_to_tex(cap_heading_levels(body)))     # "$" inside code is not math
    body = fix_text_math(normalize_math_macros(resolve_ifthenelse(body)))
    body = drop_counter_hacks(restore_math_text(body))
    # tidy first: it joins "$a$$b$", which the display-math pass would otherwise read as the start of a block
    body = restore_code(isolate_display_math(tidy_math(body)), code)
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


def convert(arg, out, mn, title=None):
    """Convert one arXiv paper (id with optional version) to ``out``; returns the quality report. ``title`` takes
    the place of arXiv's (the published version's title, a hand-corrected one)."""
    aid, ver = parse_arxiv_arg(arg)
    meta = arxiv_meta(aid)
    if title:
        meta = dict(meta, title=title)
    ver = ver or meta["latest"]
    report = {"arxiv": aid + ver, "out": out, "ok": False}

    ws = fetch(aid, ver, report)
    if ws is None:
        return report
    label_equations(ws)
    compile_latex(ws, report)
    labels = parse_aux(ws.build)
    report["labels"] = len(labels)
    flat = flatten(ws)
    bib = load_bibliography(flat, locate_bbl(ws))
    report["bib_entries"] = len(bib) if bib else 0

    stats = {"unresolved_refs": 0, "unresolved_cites": 0}
    flat = inline_listing_files(flat, ws.bdir)
    defs = local_style_macros(flat, ws.bdir)
    if defs:
        at = flat.find("\\usepackage")
        flat = flat[:at] + defs + "\n" + flat[at:] if at >= 0 else defs + "\n" + flat
    floats = use_long_captions(parse_floats(ws.build), flat)
    sections = parse_sections(ws.bdir, ws.base)
    tex, repairs = repair_braces(escape_citation_ampersands(preprocess(flat, labels, bib, stats, floats)))
    report.update(stats)
    report["tables_in_tex"] = len(re.findall(r"\\begin\{table\*?\}", tex))

    body, err, dropped = run_pandoc(ws, tex, labels, floats, sections, bib, report)
    if body is None:
        return report
    mm = re.search(r"tex2md-stats figures=(\d+) tables=(\d+)", err)
    if mm:
        report["figures_out"], report["tables_out"] = int(mm.group(1)), int(mm.group(2))
    body = postprocess(body)
    report["unresolved_left"] = len(re.findall(r"\[[a-zA-Z]+:[^\]\s]+\]", body)) + body.count("??")

    diags = (scan_markdown(body) + from_pandoc_stderr(err) + latex_reference_diagnostics(stats)
             + dropped_diagnostics(dropped) + scan_lost_text(tex, body) + brace_diagnostics(repairs)
             + scan_missing_sections(sections, body)
             + main_file_diagnostics(os.path.relpath(ws.main, ws.src), list(ws.candidates)))
    report["diagnostics"] = [d.to_dict() for d in diags]
    report["diagnostic_counts"] = counts(diags)

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_document(meta, aid, ver, mn, body))
    report["ok"] = True
    with open(os.path.join(ws.work, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return report
