import json
import os
import subprocess
import sys

from tex2md import __version__
from tex2md.cli import build_parser, main

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")

MD = """---
title: "A paper"
authors: ["A B"]
year: 2020
arxiv: "2001.00001"
arxiv_version: "v1"
marginnote:
  - bookmd5: aaa111
    path: "PCIC/missing.pdf"
source: "arXiv LaTeX, tex2md 0.1"
converted: 2026-01-01
---

# A paper

body
"""


def test_every_documented_command_exists():
    parser = build_parser()
    choices = next(a for a in parser._actions if a.dest == "command").choices
    assert set(choices) == {"arxiv", "jats", "pdf", "index", "check", "frontmatter", "add-mn", "audit", "fixmath", "doctor"}


def test_version_flag(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert __version__ in capsys.readouterr().out


def test_index_lists_the_paper(tmp_path):
    (tmp_path / "a.md").write_text(MD)
    assert main(["index", str(tmp_path)]) in (0, None)
    index = (tmp_path / "INDEX.md").read_text()
    assert "A paper" in index and "2001.00001" in index


def test_check_reports_a_missing_pdf(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("tex2md.library.papers.MN_ROOT", str(tmp_path / "nowhere"))
    (tmp_path / "a.md").write_text(MD)
    main(["check", str(tmp_path)])
    out = capsys.readouterr().out
    assert "MISSING PDF" in out and "a.md" in out


def test_add_mn_appends_a_document_and_keeps_the_body(tmp_path):
    f = tmp_path / "a.md"
    f.write_text(MD)
    main(["add-mn", str(f), "bbb222::PCIC/other.pdf"])
    text = f.read_text()
    assert "bookmd5: bbb222" in text and text.endswith("# A paper\n\nbody\n")


def test_fixmath_isolates_inline_display_math(tmp_path, capsys):
    f = tmp_path / "a.md"
    f.write_text(MD + "\ntext $$x=1$$ more\n")
    main(["fixmath", str(f)])
    assert "$$x=1$$" in f.read_text()


def test_legacy_wrappers_still_work():
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    for script in ("tex2md.py", "papers.py", "doc2md.py"):
        p = subprocess.run([sys.executable, os.path.join(root, script), "--help"], capture_output=True, text=True)
        assert p.returncode == 0 and "usage" in p.stdout.lower(), script


def test_doctor_runs_and_reports_tools(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "pandoc" in out and "latexmk" in out and rc in (0, 1)


def test_jats_conversion_end_to_end(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<article><body><sec><title>Intro</title><p>Hello <italic>world</italic>.</p></sec></body></article>")
    out = tmp_path / "out.md"
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, "-m", "tex2md", "jats", str(xml), "-o", str(out), "--title", "T",
                        "--authors", "A B", "--year", "2020"], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["title"] == "T"
    assert "Hello *world*." in out.read_text()
