from tex2md.build.aux import parse_aux, parse_floats

MAIN_AUX = r"""
\relax
\@input{ch1.aux}
\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}
\newlabel{eq:loss}{{2}{3}{Loss}{equation.2}{}}
\@writefile{lof}{\contentsline {figure}{\numberline {1}{\ignorespaces \textbf  {Overview} of the model.}}{2}{figure.caption.1}\protect \ifthenelse {\boolean {@tempswa}}{}{}}
\@writefile{lot}{\contentsline {table}{\numberline {1}{\ignorespaces Results on $x$.}}{4}{table.caption.2}\protect \ifthenelse {\boolean {@tempswa}}{}{}}
"""
CH1_AUX = r"""
\@writefile{lof}{\contentsline {figure}{\numberline {2}{\ignorespaces Appendix plot.}}{9}{figure.caption.3}}
"""


def write(tmp_path):
    (tmp_path / "main.aux").write_text(MAIN_AUX)
    (tmp_path / "ch1.aux").write_text(CH1_AUX)


def test_parse_aux_reads_section_and_equation_labels(tmp_path):
    write(tmp_path)
    labels = parse_aux(str(tmp_path))
    assert labels["sec:intro"]["num"] == "1" and labels["sec:intro"]["type"] == "section"
    assert labels["eq:loss"]["num"] == "2"


def test_parse_floats_follows_input_order_and_keeps_numbers(tmp_path):
    write(tmp_path)
    floats = parse_floats(str(tmp_path))
    # \@input{ch1.aux} comes first in main.aux, so the included chapter's float is first
    assert [(f["kind"], f["num"]) for f in floats] == [("figure", "2"), ("figure", "1"), ("table", "1")]
    assert floats[0]["text"].startswith("appendixplot") and floats[1]["text"].startswith("overview")


def test_parse_floats_is_empty_without_caption_lines(tmp_path):
    (tmp_path / "main.aux").write_text(r"\relax \newlabel{a}{{1}{1}}")
    assert parse_floats(str(tmp_path)) == []
