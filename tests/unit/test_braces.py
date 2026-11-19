from tex2md.preprocess.braces import repair_braces

DOC = "\\documentclass{article}\\newcommand{\\a}{\\begin{center}}\n\\begin{document}\n%s\n\\end{document}"


def test_balanced_text_is_returned_untouched():
    text = DOC % "Some {\\bf bold} text, \\{ escaped \\} braces and $x_{1}$.\n\n\\begin{verbatim}\n{ not a group\n\\end{verbatim}"
    assert repair_braces(text) == (text, [])


def test_a_group_that_never_closes_is_closed_where_its_paragraph_ends():
    text = DOC % "One paragraph {The instruction tuned models are evaluated.\n\nNext paragraph stays outside it."
    out, repairs = repair_braces(text)
    assert "evaluated.}\n\nNext paragraph stays outside it." in out
    assert [(r.kind, r.line) for r in repairs] == [("closed", 3)] and "The instruction tuned" in repairs[0].context


def test_a_closing_brace_that_closes_nothing_is_removed():
    out, repairs = repair_braces(DOC % "Text } more text.\n\nThe end.")
    assert "Text  more text." in out and [r.kind for r in repairs] == ["removed"]


def test_a_group_left_open_in_the_last_paragraph_closes_before_end_document():
    out, repairs = repair_braces(DOC % "Last paragraph {open")
    assert out.endswith("Last paragraph {open}\n\\end{document}") and len(repairs) == 1


def test_braces_inside_verbatim_and_comment_environments_do_not_count():
    text = DOC % "a\n\\begin{comment}\nsome } stray { text }\n\\end{comment}\nb \\verb|{| c\n\\begin{lstlisting}\nx = {\n\\end{lstlisting}"
    assert repair_braces(text) == (text, [])


def test_the_preamble_is_not_touched():
    text = "\\newcommand{\\open}{\\begin{center}{}\n\\begin{document}\nbody\n\\end{document}"
    assert repair_braces(text) == (text, [])


def test_a_brace_of_the_preamble_never_closed_is_dropped():
    text = "\\usepackage{listings}{\n\\lstset{language=Python}\n\\begin{document}\nbody\n\\end{document}"
    fixed, repairs = repair_braces(text)
    assert fixed == "\\usepackage{listings}\n\\lstset{language=Python}\n\\begin{document}\nbody\n\\end{document}"
    assert [(r.kind, r.line) for r in repairs] == [("dropped-open", 1)]
