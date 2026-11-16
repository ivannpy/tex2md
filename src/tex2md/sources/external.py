"""Metadata and front matter for papers that are not on arXiv (Crossref / command line)."""
from __future__ import annotations

import datetime
import json
import urllib.parse

from tex2md.util.io import http_get, yaml_str


def crossref(doi):
    data = json.loads(http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    authors = [" ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name", "")
               for a in data.get("author", [])]
    date = (data.get("published") or data.get("issued") or {}).get("date-parts", [[None]])[0]
    venue = (data.get("container-title") or [None])[0]
    return {"title": " ".join((data.get("title") or [""])[0].split()), "authors": authors,
            "year": date[0], "doi": doi, "venue": venue}


def external_frontmatter(meta, mn, source):
    lines = ["---", "title: " + yaml_str(meta["title"]),
             "authors: [" + ", ".join(yaml_str(a) for a in meta["authors"]) + "]",
             f"year: {meta['year']}"]
    if meta.get("doi"):
        lines.append("doi: " + yaml_str(meta["doi"]))
    if meta.get("venue"):
        lines.append("venue: " + yaml_str(meta["venue"]))
    if mn:
        lines.append("marginnote:")
        for b, p in mn:
            lines += ["  - bookmd5: " + b, "    path: " + yaml_str(p)]
    lines += ["source: " + yaml_str(source), "converted: " + datetime.date.today().isoformat(), "---", ""]
    return "\n".join(lines)


def external_header(meta, note):
    h = ["# " + meta["title"], "", "**Autores:** " + "; ".join(meta["authors"]), ""]
    if meta.get("venue"):
        h += [f"**Publicado en:** {meta['venue']}, {meta['year']}", ""]
    ref = f"> DOI: https://doi.org/{meta['doi']}" if meta.get("doi") else "> Sin DOI."
    h += [ref, "> " + note, ""]
    return "\n".join(h)
