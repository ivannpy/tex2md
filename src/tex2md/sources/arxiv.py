"""arXiv API metadata and e-print download."""
from __future__ import annotations

import gzip
import io
import json
import os
import re
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
            return json.load(fh)
    meta = arxiv_meta_batch([aid]).get(aid)
    if not meta:
        raise SystemExit(f"arXiv has no record for {aid}")
    return meta


def arxiv_meta_batch(ids):
    """Fetch and cache metadata for many ids in one API call per 50 ids."""
    out = {}
    os.makedirs(os.path.join(CACHE, "meta"), exist_ok=True)
    for i in range(0, len(ids), 50):
        if i:
            time.sleep(3)
        xml = http_get("https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids[i:i + 50]))
        for e in ET.fromstring(xml).findall("a:entry", ATOM):
            if e.find("a:title", ATOM) is None:
                continue
            m = _entry(e)
            out[m["id"]] = m
            with open(os.path.join(CACHE, "meta", m["id"].replace("/", "_") + ".json"), "w") as fh:
                json.dump(m, fh, ensure_ascii=False)
    return out


def _entry(e):
    jr = e.find("arxiv:journal_ref", ATOM)
    return {
        "id": re.search(r"abs/(.+?)(v\d+)?$", e.find("a:id", ATOM).text).group(1),
        "latest": re.search(r"(v\d+)$", e.find("a:id", ATOM).text).group(1),
        "title": " ".join(e.find("a:title", ATOM).text.split()),
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
    os.makedirs(src, exist_ok=True)
    if data[:4] == b"%PDF":
        return False
    raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            tf.extractall(src, filter="data")
    except tarfile.ReadError:
        with open(os.path.join(src, "main.tex"), "wb") as fh:
            fh.write(raw)
    return True
