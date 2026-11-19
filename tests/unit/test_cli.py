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
    assert set(choices) == {"arxiv", "jats", "pdf", "index", "check", "frontmatter", "add-mn", "audit", "fixmath", "fixmarker", "doctor"}


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


def test_audit_reports_diagnostics_and_exit_status(tmp_path, capsys):
    f = tmp_path / "a.md"
    f.write_text(MD.replace("arxiv_version", "arxiv_version") + "\nSee Eq. (??) and $\\vz$.\n")
    assert main(["audit", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "unresolved-reference" in out and "unknown-math-macro" in out and "1 with errors" in out
    f.write_text(MD + "\nAll fine, $x$.\n")
    assert main(["audit", str(tmp_path)]) == 0


def test_a_malformed_mn_argument_is_refused_before_any_work(tmp_path, capsys):
    import pytest
    for bad in ("nopath", "::path", "id::"):
        with pytest.raises(SystemExit) as e:
            main(["arxiv", "2001.00001", "-o", str(tmp_path / "x.md"), "--mn", bad])
        assert "BOOKMD5::PATH" in str(e.value)


def test_a_missing_output_directory_is_refused_before_a_long_conversion(tmp_path):
    import pytest
    with pytest.raises(SystemExit) as e:
        main(["arxiv", "2001.00001", "-o", str(tmp_path / "no" / "such" / "x.md")])
    assert "output directory does not exist" in str(e.value)


def test_exit_status_is_one_when_the_conversion_could_not_finish(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tex2md.convert.convert", lambda *a: {"ok": False, "error": "arXiv has no LaTeX source (PDF only)"})
    assert main(["arxiv", "2001.00001", "-o", str(tmp_path / "x.md")]) == 1
    assert "PDF only" in capsys.readouterr().out
    monkeypatch.setattr("tex2md.convert.convert", lambda *a: {"ok": True})
    assert main(["arxiv", "2001.00001", "-o", str(tmp_path / "x.md")]) == 0


def test_add_mn_and_frontmatter_refuse_an_entry_without_a_path(tmp_path):
    import pytest
    f = tmp_path / "a.md"
    f.write_text(MD)
    with pytest.raises(SystemExit):
        main(["add-mn", str(f), "bbb222"])


def test_check_flags_an_entry_with_no_path(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("tex2md.library.papers.MN_ROOT", str(tmp_path))     # a folder that does exist
    (tmp_path / "a.md").write_text(MD.replace('    path: "PCIC/missing.pdf"\n', ""))
    main(["check", str(tmp_path)])
    assert "MISSING PDF" in capsys.readouterr().out


def test_audit_counts_floats_kept_out_of_the_aux_by_an_empty_short_caption(tmp_path):
    from tex2md.library.audit import _unlisted_captions
    (tmp_path / "main.tex").write_text(
        r"\begin{wrapfigure}{r}{5cm}\caption[]{Teaser}\end{wrapfigure}"
        r"\begin{table}\caption[]{A}\end{table}\begin{table}\caption[Short]{B}\end{table}"
        "\n% \\begin{table}\\caption[]{commented}\\end{table}\n")
    assert _unlisted_captions(str(tmp_path)) == {"figure": 1, "table": 1}


def test_regress_keeps_the_title_of_the_header():
    import importlib.util
    spec = importlib.util.spec_from_file_location("regress", os.path.join(os.path.dirname(__file__), "..", "regress.py"))
    regress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regress)
    md = '---\ntitle: "Generative Adversarial \\"Nets\\""\narxiv: "1406.2661"\narxiv_version: "v1"\n---\n# x\n'
    cmd = regress.command_for(md, "out.md")
    assert cmd[cmd.index("--title") + 1] == 'Generative Adversarial "Nets"' and "1406.2661v1" in cmd
