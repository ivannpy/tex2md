"""YAML front matter written at the top of every converted paper."""
from __future__ import annotations

import datetime

from tex2md.config import VERSION
from tex2md.util.io import yaml_str


def frontmatter(meta, version, mn):
    lines = ["---", "title: " + yaml_str(meta["title"]),
             "authors: [" + ", ".join(yaml_str(a) for a in meta["authors"]) + "]",
             "year: " + meta["published"][:4],
             "arxiv: " + yaml_str(meta["id"]),
             "arxiv_version: " + yaml_str(version)]
    if mn:
        lines.append("marginnote:")
        for b, p in mn:
            lines += ["  - bookmd5: " + b, "    path: " + yaml_str(p)]
    lines += ["source: " + yaml_str(f"arXiv LaTeX, tex2md {VERSION}"),
              "converted: " + datetime.date.today().isoformat(), "---", ""]
    return "\n".join(lines)
