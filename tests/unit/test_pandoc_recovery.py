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


def test_definition_spans_cover_the_statement_and_not_what_follows():
    from tex2md.pandoc_recovery import definition_spans
    tex = (r"\newcolumntype{x}[1]{>{\centering}p{#1pt}} \def\a#1#2{b #1} \newcommand\notsotiny{\@setfontsize\notsotiny{8}{9}}"
           r" \let\old\new \renewcommand{\texttt}[1]{{\ttfamily #1}} text")
    found = [(tex[a:b], name) for a, b, name in definition_spans(tex)]
    assert found == [(r"\newcolumntype{x}[1]{>{\centering}p{#1pt}}", "x"), (r"\def\a#1#2{b #1}", "a"),
                     (r"\newcommand\notsotiny{\@setfontsize\notsotiny{8}{9}}", "notsotiny"), (r"\let\old\new", "old"),
                     (r"\renewcommand{\texttt}[1]{{\ttfamily #1}}", "texttt")]


def test_an_error_inside_a_preamble_definition_drops_just_that_definition():
    tex = "\\newcolumntype{x}[1]{>{\\centering}p{#1pt}}\n\\begin{document}\nText\n\\end{document}"

    def run_once(text):
        if "newcolumntype" in text:
            return 64, "", 'Error at "x" (line 1, column 37):\nunexpected #1'
        return 0, "Text", ""
    body, _, dropped, error = convert_with_recovery(tex, run_once)
    assert body == "Text" and error == "" and [(d.kind, d.name) for d in dropped] == [("definition", "\\x")]
    (d,) = diagnostics(dropped)
    assert d.code == "dropped-definition" and d.severity == "warning"


def test_a_macro_that_expands_into_itself_loses_its_definitions():
    tex = "\\newcommand\\notsotiny{\\@setfontsize\\notsotiny{8}{9}}\n\\begin{document}\\notsotiny Text\\end{document}"

    def run_once(text):
        if "@setfontsize" in text:
            return 83, "", "Loop encountered in expanding macro notsotiny"
        return 0, "Text", ""
    body, _, dropped, _ = convert_with_recovery(tex, run_once)
    assert body == "Text" and [d.name for d in dropped] == ["\\notsotiny"]


def test_the_smaller_construct_wins_when_a_definition_sits_inside_an_environment():
    tex = "\\begin{center}\n\\newcommand{\\a}[1]{#1 #2}\nkept text\n\\end{center}"

    def run_once(text):
        if "#2" in text:
            return 64, "", 'Error at "x" (line 2, column 23):\nunexpected #2'
        return 0, "ok", ""
    body, _, dropped, _ = convert_with_recovery(tex, run_once)
    assert body == "ok" and [(d.kind, d.name) for d in dropped] == [("definition", "\\a")]


def test_pandoc_that_succeeds_with_no_text_is_reported_as_an_empty_document():
    body, _, dropped, error = convert_with_recovery("\\includepdf{x.pdf}", lambda t: (0, "\n", "tex2md-stats figures=0 tables=0"))
    assert body is None and dropped == [] and "empty document" in error and "stats" not in error


def test_a_preamble_line_pandoc_cannot_read_is_dropped_and_the_text_is_kept():
    tex = "\\documentclass{article}\n\\global\\setlength{\\fboxsep}{0pt}\n\\newcommand{\\a}{b}\n\\begin{document}\nText\n\\end{document}"

    def run_once(text):
        if "\\global" in text:
            return 64, "", 'Error at "x" (line 2, column 8):\nunexpected \\setlength'
        return 0, "Text", ""
    body, _, dropped, _ = convert_with_recovery(tex, run_once)
    assert body == "Text" and len(dropped) == 1 and dropped[0].kind == "definition" and "\\global" in dropped[0].name


def test_a_preamble_statement_that_spans_lines_is_dropped_whole():
    from tex2md.pandoc_recovery import preamble_statement
    tex = "\\documentclass{article}\n\\foo{a,\n  b,\n  c}\n\\begin{document}\nx\n\\end{document}"
    start, end, _ = preamble_statement(tex, 3, 1)
    assert tex[start:end] == "\\foo{a,\n  b,\n  c}"
    assert preamble_statement(tex, 6, 1) is None          # inside the document: not a preamble statement
