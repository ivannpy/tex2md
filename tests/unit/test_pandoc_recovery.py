import shutil
import subprocess

import pytest

from tex2md.pandoc_recovery import (
    convert_with_recovery,
    diagnostics,
    drop_environment,
    error_location,
    innermost_environment,
)

ERR = 'Error at "_tex2md_flat.tex" (line 2, column 19):\nunexpected \\end\n'


def test_error_location_reads_line_and_column():
    assert error_location(ERR) == (2, 19)
    assert error_location("something else") is None


def test_innermost_environment_picks_the_smallest_enclosing_one():
    tex = "a\n\\begin{table}\n\\begin{tabular}{c} x \\end{tabular}\n\\end{table}\n"
    start, end, name = innermost_environment(tex, 3, 20)
    assert name == "tabular" and tex[start:end].startswith("\\begin{tabular}")
    assert innermost_environment(tex, 1, 1) is None


def test_document_is_never_dropped():
    tex = "\\begin{document}\nbroken\n\\end{document}"
    assert innermost_environment(tex, 2, 3) is None


def test_drop_environment_leaves_a_visible_note():
    tex = "before\n\\begin{center}\nbad\n\\end{center}\nafter"
    new, info = drop_environment(tex, innermost_environment(tex, 3, 1), "unexpected \\end")
    assert "before" in new and "after" in new and "bad" not in new and "se omitió el entorno «center»" in new
    assert (info.name, info.first_line, info.last_line) == ("center", 2, 4)


def test_retry_drops_one_environment_at_a_time_and_gives_up_when_nothing_is_left():
    calls = []

    def run_once(tex):
        calls.append(tex)
        if "BAD" in tex:
            return 64, "", 'Error at "x" (line 2, column 3):\nunexpected end of input'
        return 0, "ok body", ""

    body, _, dropped, error = convert_with_recovery("a\n\\begin{foo}BAD\\end{foo}\nz", run_once)
    assert body == "ok body" and [d.name for d in dropped] == ["foo"] and error == "" and len(calls) == 2

    always = lambda tex: (64, "", 'Error at "x" (line 1, column 1):\nunexpected')  # noqa: E731
    body, _, dropped, error = convert_with_recovery("no environment here", always)
    assert body is None and dropped == [] and error.startswith("pandoc failed")


def test_retries_are_limited():
    def run_once(tex):
        return 64, "", 'Error at "x" (line 1, column 12):\nunexpected'
    nested = "\\begin{a}\\begin{b}\\begin{c}xxxxx\\end{c}\\end{b}\\end{a}"
    body, _, dropped, error = convert_with_recovery(nested, run_once, max_retries=1)
    assert body is None and len(dropped) == 1 and error.startswith("pandoc failed")


def test_diagnostic_lists_what_was_lost():
    tex = "\\begin{center}\nbad\n\\end{center}"
    _, info = drop_environment(tex, innermost_environment(tex, 2, 1), "boom")
    (d,) = diagnostics([info])
    assert d.code == "dropped-environment" and d.severity == "error" and "center" in d.message and d.count == 1
    assert diagnostics([]) == []


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_real_pandoc_recovers_from_an_unreadable_environment(tmp_path):
    tex = ("\\documentclass{article}\\begin{document}\nIntro text.\n\\begin{center} {a \\end{center}\n"
           "After text.\n\\end{document}")

    def run_once(text):
        (tmp_path / "f.tex").write_text(text)
        p = subprocess.run(["pandoc", "f.tex", "-f", "latex", "-t", "markdown"], cwd=tmp_path,
                           capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    assert run_once(tex)[0] != 0   # the construct really breaks pandoc
    body, _, dropped, error = convert_with_recovery(tex, run_once)
    assert error == "" and [d.name for d in dropped] == ["center"]
    assert "Intro text." in body and "After text." in body and "se omitió el entorno" in body
