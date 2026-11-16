"""Command line: ``tex2md <command>``.

    arxiv        arXiv LaTeX source -> Markdown
    jats         publisher / PubMed Central JATS XML -> Markdown
    pdf          PDF -> Markdown (marker, OCR)
    frontmatter  prepend YAML front matter to a .md
    add-mn       add a MarginNote document to a .md front matter
    check        list .md whose MarginNote PDF no longer exists
    index        rebuild INDEX.md of a library directory
    fixmath      re-run the math clean-up on converted .md
    doctor       verify the external tools and data directories
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from tex2md import __version__
from tex2md.library import papers


def _mn(values):
    return [tuple(x.split("::", 1)) for x in values]


def cmd_arxiv(a):
    from tex2md.convert import convert
    print(json.dumps(convert(a.arxiv, os.path.abspath(a.out), _mn(a.mn)), ensure_ascii=False))
    return 0


def cmd_external(a):
    """jats / pdf: papers without LaTeX source; metadata from Crossref or the flags."""
    from tex2md.sources.external import crossref, external_frontmatter, external_header
    if a.doi:
        meta = crossref(a.doi)
    else:
        if not (a.title and a.authors and a.year):
            raise SystemExit("without --doi, --title, --authors and --year are required")
        meta = {"title": a.title, "authors": [x.strip() for x in a.authors.split(";")], "year": a.year}
    if a.venue:
        meta["venue"] = a.venue
    mn = _mn(a.mn)
    if a.command == "jats":
        from tex2md.sources.jats import jats_body
        body = jats_body(a.src)
        source = "JATS XML (" + (a.src if re.match(r"https?://", a.src) else os.path.basename(a.src)) + "), pandoc"
        note = "Convertido desde el XML JATS de la editorial con pandoc. Las figuras se reducen a su leyenda. Ante cualquier duda, el PDF es la referencia."
    else:
        from tex2md.sources.pdf import marker_body
        body = marker_body(a.src, a.mode)
        source = f"PDF, marker ({a.mode})"
        note = ("Convertido desde el PDF con marker (OCR + modelo de diseño): puede haber errores de reconocimiento, "
                "sobre todo en ecuaciones y tablas. Sin figuras. Ante cualquier duda, el PDF es la referencia.")
    # marker repeats the title as the first heading
    body = re.sub(r"\A\s*#+ " + re.escape(meta["title"]) + r"\s*\n", "", body, flags=re.I)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(external_frontmatter(meta, mn, source) + "\n" + external_header(meta, note) + "\n" + body.strip() + "\n")
    print(json.dumps({"out": a.out, "words": len(body.split()), "title": meta["title"]}, ensure_ascii=False))
    return 0


def cmd_fixmath(a):
    from tex2md.postprocess.fixmath import cmd_fixmath as run
    return run(a)


def cmd_doctor(a):
    from tex2md.doctor import run_doctor
    return run_doctor()


def build_parser():
    ap = argparse.ArgumentParser(prog="tex2md", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version="tex2md " + __version__)
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("arxiv", help="arXiv LaTeX source -> Markdown")
    p.add_argument("arxiv", help="arXiv id, optionally with version (2603.23507v1)")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--mn", action="append", default=[], metavar="BOOKMD5::PATH",
                   help="MarginNote document id and its path under the MN Documents folder")
    p.set_defaults(fn=cmd_arxiv)

    for kind, text in (("jats", "publisher / PubMed Central JATS XML -> Markdown"),
                       ("pdf", "PDF -> Markdown with marker (OCR)")):
        p = sub.add_parser(kind, help=text)
        p.add_argument("src", help="XML file or URL" if kind == "jats" else "PDF file")
        p.add_argument("-o", "--out", required=True)
        p.add_argument("--doi")
        p.add_argument("--title")
        p.add_argument("--authors", help='"First Last; First Last"')
        p.add_argument("--year")
        p.add_argument("--venue")
        p.add_argument("--mn", action="append", default=[], metavar="BOOKMD5::PATH")
        if kind == "pdf":
            p.add_argument("--mode", default="balanced", choices=["fast", "balanced"])
        p.set_defaults(fn=cmd_external)

    p = sub.add_parser("index", help="rebuild INDEX.md of a library directory")
    p.add_argument("dir")
    p.set_defaults(fn=papers.cmd_index)
    p = sub.add_parser("check", help="list .md whose MarginNote PDF no longer exists")
    p.add_argument("dir")
    p.set_defaults(fn=papers.cmd_check)
    p = sub.add_parser("frontmatter", help="prepend YAML front matter to a .md")
    p.add_argument("file")
    p.add_argument("--arxiv", required=True)
    p.add_argument("--mn", action="append", default=[])
    p.add_argument("--source", default="arXiv LaTeX, pandoc")
    p.set_defaults(fn=papers.cmd_frontmatter)
    p = sub.add_parser("add-mn", help="add a MarginNote document to a .md front matter")
    p.add_argument("file")
    p.add_argument("entry", metavar="BOOKMD5::PATH")
    p.set_defaults(fn=papers.cmd_add_mn)
    p = sub.add_parser("fixmath", help="re-run the math clean-up on converted .md")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_fixmath)
    p = sub.add_parser("doctor", help="verify external tools and data directories")
    p.set_defaults(fn=cmd_doctor)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    rc = a.fn(a)
    return rc if isinstance(rc, int) else 0


def legacy(command, argv=None):
    """Entry point of the old scripts (tex2md.py, papers.py, doc2md.py): same arguments as before."""
    argv = sys.argv[1:] if argv is None else argv
    return main(([command] if command else []) + argv)


if __name__ == "__main__":
    raise SystemExit(main())
