"""arXiv API metadata and e-print download."""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import tarfile
import time
import xml.etree.ElementTree as ET

from tex2md.config import ATOM, CACHE
from tex2md.util.io import http_get


def arxiv_meta(aid):
    """arXiv API metadata, cached in cache/meta/<id>.json (the API is rate limited)."""
    path = os.path.join(CACHE, "meta", aid.replace("/", "_") + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            meta = json.load(fh)
        return dict(meta, title=plain_title(meta["title"]))      # (cached before titles were cleaned)
    meta = arxiv_meta_batch([aid]).get(aid)
    if not meta:
        raise SystemExit(f"arXiv has no record for {aid}")
    return meta


API_GAP = 3.0       # seconds between two requests to the arXiv API: its terms of use, and it answers 429 otherwise


def _pace_api():
    """Wait until ``API_GAP`` seconds have passed since the last API request of any run (a loop that converts one
    paper after another used to be refused after a dozen)."""
    path = os.path.join(CACHE, "meta", ".last_request")
    try:
        with open(path) as fh:
            last = float(fh.read())
    except (OSError, ValueError):
        last = 0.0
    wait = last + API_GAP - time.time()
    if wait > 0:
        time.sleep(wait)
    with open(path, "w") as fh:
        fh.write(str(time.time()))


def arxiv_meta_batch(ids):
    """Fetch and cache metadata for many ids in one API call per 50 ids."""
    out = {}
    os.makedirs(os.path.join(CACHE, "meta"), exist_ok=True)
    for i in range(0, len(ids), 50):
        _pace_api()
        xml = http_get("https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids[i:i + 50]))
        for e in ET.fromstring(xml).findall("a:entry", ATOM):
            if e.find("a:title", ATOM) is None:
                continue
            m = _entry(e)
            out[m["id"]] = m
            with open(os.path.join(CACHE, "meta", m["id"].replace("/", "_") + ".json"), "w") as fh:
                json.dump(m, fh, ensure_ascii=False)
    return out


def plain_title(title):
    """``The Diffusion Duality, Chapter II: $Ψ$-Samplers`` -> ``... Ψ-Samplers``: dollars around plain text (no TeX
    command, sub- or superscript) are only there to typeset it; a title with real math keeps them."""
    return re.sub(r"\$([^$]*)\$", lambda m: m.group(1) if not re.search(r"[\\^_{}]", m.group(1)) else m.group(0), title)


def _entry(e):
    jr = e.find("arxiv:journal_ref", ATOM)
    return {
        "id": re.search(r"abs/(.+?)(v\d+)?$", e.find("a:id", ATOM).text).group(1),
        "latest": re.search(r"(v\d+)$", e.find("a:id", ATOM).text).group(1),
        "title": plain_title(" ".join(e.find("a:title", ATOM).text.split())),
        "authors": [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)],
        "published": e.find("a:published", ATOM).text[:10],
        "abstract": " ".join(e.find("a:summary", ATOM).text.split()),
        "journal_ref": " ".join(jr.text.split()) if jr is not None and jr.text else None,
    }


def fetch_source(aid, dest):
    """Download and unpack the e-print into dest/src. Returns False if arXiv only has a PDF."""
    src = os.path.join(dest, "src")
    if os.path.isdir(src) and os.listdir(src):
        return True
    data = http_get("https://arxiv.org/e-print/" + aid)
    if data[:4] == b"%PDF":
        return False
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    # unpack beside the final place and rename: an interrupted or refused extraction must not leave a
    # half-filled src/ that the next run would take for the whole source
    partial = src + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    os.makedirs(partial)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            tf.extractall(partial, filter="data")
    except tarfile.ReadError:
        with open(os.path.join(partial, "main.tex"), "wb") as fh:
            fh.write(raw)
    shutil.rmtree(src, ignore_errors=True)
    os.replace(partial, src)
    return True
