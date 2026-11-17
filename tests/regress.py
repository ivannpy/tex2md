#!/usr/bin/env python3
"""Regression harness: reconvert papers with the current code and compare with a frozen baseline.

    python3 tests/regress.py BASELINE_DIR [--jobs 4] [--only NAME ...]

Each baseline .md carries its own arXiv id, version and MarginNote links in the front matter, so the
conversion is replayed exactly (from the local cache, no network needed once downloaded). The
comparison ignores the ``converted:`` date. Exit status 1 if any file differs.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def command_for(md_text: str, out: str) -> list[str]:
    fm = md_text.split("---\n", 2)[1]
    aid = re.search(r'^arxiv: "([^"]+)"', fm, re.M).group(1)
    ver = re.search(r'^arxiv_version: "([^"]+)"', fm, re.M)
    cmd = [sys.executable, "-m", "tex2md", "arxiv", aid + (ver.group(1) if ver else ""), "-o", out]
    for b, p in re.findall(r'bookmd5: (\w+)\s+path: ("(?:[^"\\]|\\.)*")', fm):
        cmd += ["--mn", f"{b}::{json.loads(p)}"]      # the path is a JSON string: decode its escapes (\u001b, \")
    return cmd


def normalise(text: str) -> str:
    return re.sub(r"^converted: .*$", "converted: X", text, flags=re.M)


def check(path: str, outdir: str):
    name = os.path.basename(path)
    base = Path(path).read_text()
    out = os.path.join(outdir, name)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    p = subprocess.run(command_for(base, out), capture_output=True, text=True, env=env, cwd=outdir)
    if p.returncode != 0 or not os.path.exists(out):
        return name, "FAILED", (p.stderr or p.stdout)[-400:]
    new = Path(out).read_text()
    # the front matter of the baseline keeps the original paths verbatim; compare the body and the stable keys
    nb, nn = normalise(base), normalise(new)
    if nb.split("---\n", 2)[2] == nn.split("---\n", 2)[2]:
        return name, "same", ""
    diff = "".join(list(difflib.unified_diff(nb.splitlines(True), nn.splitlines(True), "baseline", "new", n=0))[:30])
    return name, "DIFF", diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    files = sorted(os.path.join(a.baseline, f) for f in os.listdir(a.baseline) if f.endswith(".md"))
    if a.only:
        files = [f for f in files if any(o in f for o in a.only)]
    bad = 0
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.jobs) as ex:
        for name, status, detail in ex.map(lambda f: check(f, tmp), files):
            if status != "same":
                bad += 1
                print(f"{status:7} {name}\n{detail}")
    print(f"{len(files) - bad}/{len(files)} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
