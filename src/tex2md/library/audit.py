"""``tex2md audit DIR``: compare the figures and tables of each converted .md with the compiled PDF.

For every .md converted from arXiv LaTeX, the numbers of the real captions (read from the .aux of its
cached build) are compared with the ``**Figura N.**`` / ``**Tabla N.**`` lines of the Markdown.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from tex2md.build.aux import parse_floats
from tex2md.config import CACHE

KINDS = (("table", "Tabla"), ("figure", "Figura"))


@dataclass
class Audit:
    file: str
    missing: list[str] = field(default_factory=list)     # in the PDF, not in the .md
    unknown: list[str] = field(default_factory=list)     # in the .md, not in the .aux
    unnumbered: int = 0                                  # "**Figura.**" / "**Tabla.**" lines
    no_data: bool = False                                # no cached build / no caption lines


def audit_file(path: str | Path) -> Audit | None:
    text = Path(path).read_text(encoding="utf-8")
    if 'source: "arXiv LaTeX' not in text:
        return None
    aid = re.search(r'^arxiv: "([^"]+)"', text, re.M)
    ver = re.search(r'^arxiv_version: "([^"]+)"', text, re.M)
    result = Audit(file=os.path.basename(str(path)))
    result.unnumbered = len(re.findall(r"^\*\*(?:Tabla|Figura)\.\*\*", text, re.M))
    if not aid:
        result.no_data = True
        return result
    build = os.path.join(CACHE, (aid.group(1) + (ver.group(1) if ver else "")).replace("/", "_"), "build")
    floats = parse_floats(build) if os.path.isdir(build) else []
    if not floats:
        result.no_data = True
        return result
    for kind, word in KINDS:
        expected = [f["num"] for f in floats if f["kind"] == kind]
        got = set(re.findall(r"^\*\*" + word + r" ([A-Z0-9.]+)\.\*\*", text, re.M))
        if not expected:
            continue
        result.missing += [f"{word} {n}" for n in expected if n not in got]
        result.unknown += [f"{word} {n}" for n in sorted(got) if n not in expected]
    return result


def cmd_audit(a):
    problems = 0
    for name in sorted(os.listdir(a.dir)):
        if not name.endswith(".md") or name in ("INDEX.md",):
            continue
        r = audit_file(os.path.join(a.dir, name))
        if r is None or r.no_data:
            continue
        if r.missing or r.unknown:
            problems += 1
            print(f"{r.file:<46} missing: {', '.join(r.missing) or '-':<30} not in .aux: {', '.join(r.unknown) or '-'}")
    print(f"{problems} file(s) with differences")
    return 1 if problems else 0
