"""``tex2md doctor``: check the external tools and the data directories."""
from __future__ import annotations

import os
import re
import shutil
import subprocess

from tex2md.config import CACHE, MARKER, MODELS

# name, executable, how to get it, what needs it
TOOLS = [
    ("pandoc", "pandoc", "brew install pandoc", "every conversion (>= 3.0 recommended)"),
    ("latexmk", "latexmk", "install MacTeX or TeX Live", "arxiv: compiles the source for the real numbering"),
    ("pdflatex", "pdflatex", "install MacTeX or TeX Live", "arxiv"),
    ("latexpand", "latexpand", "tlmgr install latexpand", "arxiv: flattens \\input/\\include"),
    ("uv", "uv", "brew install uv", "pdf: runs marker in its own environment"),
]


def _version(exe):
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"\d+(?:\.\d+)+", out)
    return m.group(0) if m else ""


def run_doctor() -> int:
    problems = 0
    print("External tools")
    for name, exe, hint, why in TOOLS:
        path = shutil.which(exe)
        if path:
            print(f"  ok       {name:<9} {_version(exe) or '?':<9} {path}")
        else:
            problems += 1
            print(f"  MISSING  {name:<9} {why}\n           -> {hint}")
    print("Data directories")
    for label, path, _need in (("cache", CACHE, False), ("models", MODELS, False), ("marker env", MARKER, False)):
        state = "ok      " if os.path.isdir(path) else "absent  "
        print(f"  {state} {label:<10} {path}")
    if not os.path.exists(os.path.join(MARKER, "pyproject.toml")):
        print("  note     the marker project is missing: the pdf command will not work")
    print("Environment variables: TEX2MD_CACHE, TEX2MD_MODELS, TEX2MD_MARKER")
    print("OK" if not problems else f"{problems} problem(s) found")
    return 1 if problems else 0
