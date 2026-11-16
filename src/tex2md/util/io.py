"""Small I/O helpers: file reads, subprocesses, HTTP, YAML scalars."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request

from tex2md.config import UA


def http_get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def read(path):
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", errors="replace")


def run(cmd, cwd, timeout):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def yaml_str(s):
    return json.dumps(s, ensure_ascii=False)
