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
from tex2md.diagnostics import SEVERITIES, Diagnostic, scan_markdown

KINDS = (("table", "Tabla"), ("figure", "Figura"))


@dataclass
class Audit:
    file: str
    missing: list[str] = field(default_factory=list)     # in the PDF, not in the .md
    unknown: list[str] = field(default_factory=list)     # in the .md, not in the .aux
    unnumbered: int = 0                                  # "**Figura.**" / "**Tabla.**" lines
    no_data: bool = False                                # no cached build / no caption lines
    diagnostics: list[Diagnostic] = field(default_factory=list)   # what scan_markdown finds in the body


def audit_file(path: str | Path) -> Audit | None:
    text = Path(path).read_text(encoding="utf-8")
    if 'source: "arXiv LaTeX' not in text:
        return None
    aid = re.search(r'^arxiv: "([^"]+)"', text, re.M)
    ver = re.search(r'^arxiv_version: "([^"]+)"', text, re.M)
    result = Audit(file=os.path.basename(str(path)))
    result.unnumbered = len(re.findall(r"^\*\*(?:Tabla|Figura)\.\*\*", text, re.M))
    result.diagnostics = scan_markdown(text.split("\n---\n", 1)[-1])
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
    """Float numbers against the PDF, plus the typed diagnostics found in each body.
    Exit status 1 if a float is missing/unknown or any diagnostic of severity ``error`` appears."""
    shown = SEVERITIES[:SEVERITIES.index(a.level) + 1]
    float_problems = error_files = 0
    totals: dict[str, int] = {}
    for name in sorted(os.listdir(a.dir)):
        if not name.endswith(".md") or name in ("INDEX.md",):
            continue
        r = audit_file(os.path.join(a.dir, name))
        if r is None:
            continue
        lines = []
        if not r.no_data and (r.missing or r.unknown):
            float_problems += 1
            lines.append(f"missing: {', '.join(r.missing) or '-':<30} not in .aux: {', '.join(r.unknown) or '-'}")
        for d in r.diagnostics:
            totals[d.code] = totals.get(d.code, 0) + d.count
            if d.severity in shown:
                lines.append(f"{d.severity:<7} {d.code} x{d.count}  {'; '.join(d.examples[:3])}"[:150])
        if any(d.severity == "error" for d in r.diagnostics):
            error_files += 1
        for i, line in enumerate(lines):
            print(f"{r.file if i == 0 else '':<46} {line}")
    print(f"{float_problems} file(s) with float differences, {error_files} with errors")
    if totals:
        print("totals: " + ", ".join(f"{c} {n}" for c, n in sorted(totals.items())))
    return 1 if (float_problems or error_files) else 0
