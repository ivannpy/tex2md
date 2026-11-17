"""Small I/O helpers: file reads, subprocesses, HTTP, YAML scalars."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request

from tex2md.config import UA


def _retry_after(error):
    """Seconds a server asks to wait (``Retry-After``), capped; None when it does not say."""
    value = error.headers.get("Retry-After") if getattr(error, "headers", None) else None
    try:
        return min(int(value), 120) if value is not None else None
    except ValueError:
        return None


def http_get(url, tries=4):
    """GET with retries for what can pass (timeouts, 5xx, 429: waiting as long as the server asks). A 4xx answer is
    final: retrying a "not found" only makes the command wait half a minute before it says so."""
    for i in range(tries):
        wait = 5 * (i + 1)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429 or i == tries - 1:
                raise
            wait = _retry_after(e) or wait
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(wait)


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
