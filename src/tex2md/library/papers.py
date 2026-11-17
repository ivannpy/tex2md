"""Maintain the paper library: front matter of each .md, MarginNote links and INDEX.md."""
from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

from tex2md.config import MN_ROOT
from tex2md.sources.arxiv import arxiv_meta
from tex2md.util.io import yaml_str


def split_frontmatter(text):
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, text
    return text[4:end], text[end + 5:]


def parse_frontmatter(fm):
    """Minimal reader for the front matter these tools write."""
    data, mn, cur = {}, [], None
    for line in fm.splitlines():
        if line.startswith("  - bookmd5:"):
            cur = {"bookmd5": line.split(":", 1)[1].strip()}
            mn.append(cur)
        elif line.startswith("    path:") and cur is not None:
            cur["path"] = json.loads(line.split(":", 1)[1].strip())
        elif re.match(r"^[a-z_]+:", line):
            k, v = line.split(":", 1)
            v = v.strip()
            try:
                data[k] = json.loads(v) if v else None
            except json.JSONDecodeError:
                data[k] = v
    data["marginnote"] = mn
    return data


def build_frontmatter(meta, mn, source, version=None):
    lines = ["---", "title: " + yaml_str(meta["title"]),
             "authors: [" + ", ".join(yaml_str(a) for a in meta["authors"]) + "]",
             "year: " + meta["published"][:4],
             "arxiv: " + yaml_str(meta["id"])]
    if version:
        lines.append("arxiv_version: " + yaml_str(version))
    if mn:
        lines.append("marginnote:")
        for b, p in mn:
            lines += ["  - bookmd5: " + b, "    path: " + yaml_str(p)]
    lines += ["source: " + yaml_str(source), "converted: " + datetime.date.today().isoformat(), "---", ""]
    return "\n".join(lines)


def cmd_frontmatter(a):
    text = Path(a.file).read_text(encoding="utf-8")
    fm, _ = split_frontmatter(text)
    if fm is not None:
        raise SystemExit(f"{a.file} already has front matter")
    meta = arxiv_meta(a.arxiv)
    mn = []
    for value in a.mn:
        book, sep, path = value.partition("::")
        if not sep:
            raise SystemExit(f"--mn expects BOOKMD5::PATH, got: {value}")
        mn.append((book, path))
    new = build_frontmatter(meta, mn, a.source) + "\n" + text
    tmp = a.file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(new)
    assert Path(tmp).read_text(encoding="utf-8").endswith(text)
    os.replace(tmp, a.file)


def cmd_add_mn(a):
    text = Path(a.file).read_text(encoding="utf-8")
    fm, body = split_frontmatter(text)
    if fm is None:
        raise SystemExit(f"{a.file} has no front matter")
    b, sep, p = a.entry.partition("::")
    if not sep:
        raise SystemExit(f"expected BOOKMD5::PATH, got: {a.entry}")
    if b in fm:
        return
    entry = "  - bookmd5: " + b + "\n    path: " + yaml_str(p)
    if "\nmarginnote:" in "\n" + fm:
        lines = fm.splitlines()
        i = lines.index("marginnote:") + 1
        while i < len(lines) and lines[i].startswith("  "):
            i += 1
        lines.insert(i, entry)
        fm = "\n".join(lines)
    else:
        fm = re.sub(r"(?m)^(source:)", "marginnote:\n" + entry + r"\n\1", fm, count=1)
    with open(a.file, "w", encoding="utf-8") as fh:
        fh.write("---\n" + fm + "\n---\n" + body)


def cmd_check(a):
    """Report .md files whose MarginNote paths no longer exist (deleted, moved or renamed in MN)."""
    bad = 0
    for f in sorted(os.listdir(a.dir)):
        if not f.endswith(".md") or f in ("INDEX.md", "CLAUDE.md", "README.md"):
            continue
        fm, _ = split_frontmatter(Path(os.path.join(a.dir, f)).read_text(encoding="utf-8"))
        if fm is None:
            print(f"NO FRONT MATTER  {f}")
            bad += 1
            continue
        mn = parse_frontmatter(fm)["marginnote"]
        if not mn:
            print(f"NO MARGINNOTE    {f}")
            bad += 1
        for m in mn:
            # an entry without path would join to the MarginNote folder itself, which always exists
            if not m.get("path") or not os.path.exists(os.path.join(MN_ROOT, m["path"])):
                print(f"MISSING PDF      {f}  {m['bookmd5'][:12]}  {m.get('path')}")
                bad += 1
    print(f"{bad} problems")


def cmd_index(a):
    rows = []
    for f in sorted(os.listdir(a.dir)):
        if not f.endswith(".md") or f in ("INDEX.md", "CLAUDE.md", "README.md"):
            continue
        fm, _ = split_frontmatter(Path(os.path.join(a.dir, f)).read_text(encoding="utf-8"))
        if fm is None:
            continue
        rows.append((f, parse_frontmatter(fm)))
    rows.sort(key=lambda r: (str(r[1].get("year", "")), r[0]))
    out = ["# Índice de artículos", "",
           "Generado con `python3 ~/Documents/tex2md/papers.py index <esta carpeta>` a partir del encabezado YAML de cada `.md`. "
           "No editar a mano: editar el encabezado del `.md` y regenerar.", "",
           f"{len(rows)} artículos. `bookmd5` identifica el PDF en MarginNote (origen único de los PDF).", "",
           "| Archivo | Título | Año | arXiv / DOI | MarginNote (bookmd5 · ruta) |", "|---|---|---|---|---|"]
    def esc(s):
        return str(s).replace("|", "\\|")
    for f, d in rows:
        mn = "<br>".join(f"`{m['bookmd5'][:12]}…` · {esc(m.get('path', ''))}" for m in d["marginnote"]) or "—"
        ref = d.get("arxiv") or (f"doi:{d['doi']}" if d.get("doi") else "—")
        out.append(f"| [{f}]({f}) | {esc(d.get('title', ''))} | {d.get('year', '')} | {ref} | {mn} |")
    with open(os.path.join(a.dir, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"INDEX.md: {len(rows)} entries")
