"""The pandoc filter, run on small LaTeX documents with hand-made labels.json / floats.json."""
import json
import re
import shutil
import subprocess

import pytest

from tex2md.config import FILTER

pytestmark = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")

FMT = ("markdown-raw_attribute-header_attributes-fenced_divs-bracketed_spans-native_divs"
       "-native_spans-link_attributes-inline_code_attributes-simple_tables-multiline_tables"
       "-grid_tables-smart-auto_identifiers-implicit_figures")


def convert(tmp_path, body, labels=None, floats=None, sections=None):
    (tmp_path / "in.tex").write_text("\\documentclass{article}\\usepackage{graphicx}\\begin{document}\n" + body + "\n\\end{document}")
    (tmp_path / "labels.json").write_text(json.dumps(labels or {}))
    (tmp_path / "floats.json").write_text(json.dumps(floats or []))
    (tmp_path / "sections.json").write_text(json.dumps(sections or []))
    p = subprocess.run(["pandoc", "in.tex", "-f", "latex", "-t", FMT, "--wrap=none", "--lua-filter", FILTER,
                        "-M", "tex2md_labels=labels.json", "-M", "tex2md_floats=floats.json",
                        "-M", "tex2md_sections=sections.json"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    return p.stdout


def captions(md):
    return re.findall(r"^\*\*((?:Figura|Tabla)(?: [A-Z0-9.]+)?)\.\*\*", md, re.M)


def fl(kind, num, text):
    return {"kind": kind, "num": num, "text": re.sub(r"[^a-z0-9]", "", text.lower())[:40]}


FIG = r"\begin{figure}\includegraphics{a.png}\caption{%s}\end{figure}"
TAB = r"\begin{table}\caption{%s}\begin{tabular}{c}1\end{tabular}\end{table}"


def test_figures_and_tables_get_the_number_of_the_aux(tmp_path):
    md = convert(tmp_path, TAB % "Results on data" + FIG % "Overview of the model",
                 floats=[fl("figure", "7", "Overview of the model"), fl("table", "3", "Results on data")])
    assert captions(md) == ["Tabla 3", "Figura 7"]


def test_identical_consecutive_captions_are_separate_floats_when_the_aux_lists_both(tmp_path):
    md = convert(tmp_path, FIG % "Same words" + FIG % "Same words",
                 floats=[fl("figure", "14", "Same words"), fl("figure", "15", "Same words")])
    assert captions(md) == ["Figura 14", "Figura 15"]


def test_panels_of_one_float_repeat_its_number(tmp_path):
    two_tabulars = (r"\begin{table}\caption{Shared caption}\begin{tabular}{c}1\end{tabular}"
                    r"\begin{tabular}{c}2\end{tabular}\end{table}")
    md = convert(tmp_path, two_tabulars, floats=[fl("table", "4", "Shared caption")])
    assert captions(md) == ["Tabla 4", "Tabla 4"]


def test_a_float_without_its_own_caption_gets_no_number(tmp_path):
    md = convert(tmp_path, r"\begin{figure}\includegraphics{a.png}\end{figure}", floats=[fl("figure", "1", "x")])
    assert captions(md) == ["Figura"]


def test_figure_whose_caption_is_a_table_of_the_aux_is_printed_as_tabla(tmp_path):
    # a figure environment with a plot beside a table, captioned with \captionof{table} (wang2026computegap)
    cap = "Compute multiples at every budget of each variant"
    md = convert(tmp_path, FIG % cap, floats=[fl("figure", "1", "Revisiting the compute multiple"),
                                              fl("table", "2", cap)])
    assert captions(md) == ["Tabla 2"]


def test_marked_table_without_tabular_is_printed_as_tabla(tmp_path):
    md = convert(tmp_path, FIG % "TEX2MDTABLE Word lists", floats=[fl("table", "5", "Word lists")])
    assert captions(md) == ["Tabla 5"] and "TEX2MDTABLE" not in md


def test_without_aux_data_a_running_count_is_used(tmp_path):
    md = convert(tmp_path, FIG % "One" + FIG % "Two")
    assert captions(md) == ["Figura 1", "Figura 2"]


def test_label_of_the_right_kind_is_used_when_the_aux_has_no_floats(tmp_path):
    body = r"\begin{table}\caption{T}\label{tab:x}\begin{tabular}{c}1\end{tabular}\end{table}"
    md = convert(tmp_path, body, labels={"tab:x": {"num": "9", "type": "table"}})
    assert captions(md) == ["Tabla 9"]


def test_headings_follow_the_aux_labels_and_resync(tmp_path):
    body = r"\section{Intro}\label{s1}\subsection{Sub}\section{Other}\label{s3}"
    md = convert(tmp_path, body, labels={"s1": {"num": "1", "type": "section"}, "s3": {"num": "5", "type": "section"}})
    assert re.findall(r"^#+ (\S+)", md, re.M) == ["1", "1.1", "5"]


def test_appendix_marker_switches_to_letters(tmp_path):
    md = convert(tmp_path, "\\section{Main}\n\nTEX2MDAPPENDIXMARKER\n\n\\section{First}\\subsection{Sub}")
    assert re.findall(r"^#+ (\S+)", md, re.M) == ["1", "A", "A.1"]


def test_display_equation_label_becomes_tag(tmp_path):
    md = convert(tmp_path, r"\begin{equation}x=1\label{eq:a}\end{equation}", labels={"eq:a": {"num": "3", "type": "equation"}})
    assert r"\tag{3}" in md and r"\label" not in md


def test_tables_are_numbered_in_document_order_even_when_one_is_a_marked_figure(tmp_path):
    body = (FIG % "TEX2MDTABLE Strict words" + TAB % "Plain results" + TAB % "Last results")
    md = convert(tmp_path, body, floats=[fl("table", "1", "Strict words"), fl("table", "2", "Plain results"),
                                         fl("table", "3", "Last results")])
    assert captions(md) == ["Tabla 1", "Tabla 2", "Tabla 3"]


def test_caption_starting_with_a_macro_matches_by_substring(tmp_path):
    # \method{} is expanded in the .aux ("udlm outperforms ...") but dropped by pandoc
    body = TAB % "outperforms other uniform diffusion on text8" + TAB % "outperforms other uniform diffusion on LM1B"
    md = convert(tmp_path, body, floats=[fl("table", "2", "udlm outperforms other uniform diffusion on text8"),
                                         fl("table", "3", "udlm outperforms other uniform diffusion on LM1B")])
    assert captions(md) == ["Tabla 2", "Tabla 3"]


def test_a_valid_label_beats_the_loose_substring_match(tmp_path):
    # the .aux text of figure 7 is corrupted; figure 9 happens to contain the same words
    body = r"\begin{figure}\includegraphics{a.png}\caption{Validation loss improves}\label{fig:g}\end{figure}"
    md = convert(tmp_path, body, labels={"fig:g": {"num": "7", "type": "figure"}},
                 floats=[fl("figure", "7", "9106plus15minus4"), fl("figure", "9", "Pareto frontier of validation loss improves")])
    assert captions(md) == ["Figura 7"]


def test_spans_and_divs_are_unwrapped(tmp_path):
    md = convert(tmp_path, r"\section{A \textsc{Small} Title}\label{s}" "\n\n" r"\begin{minipage}{3cm}Inside the box.\end{minipage}")
    assert "<span" not in md and "<div" not in md and "Small" in md and "Inside the box." in md


def test_tables_with_spans_become_pipe_tables_with_plain_math(tmp_path):
    body = (r"\begin{table}\caption{T}\begin{tabular}{lcc}\toprule"
            r"Method & \multicolumn{2}{c}{Score $\uparrow$} \\ \midrule A & 1 & 2 \\ B & 3 & 4 \\ \bottomrule\end{tabular}\end{table}")
    md = convert(tmp_path, body)
    assert "<table" not in md and "<math" not in md
    assert r"$\uparrow$" in md and "| A " in md


def test_multi_paragraph_cells_and_two_header_rows_still_give_a_pipe_table(tmp_path):
    body = (r"\begin{table}\caption{T}\begin{tabular}{ll}\toprule \textbf{Model} & \textbf{Score} \\ "
            r"\textbf{(a)} & \textbf{(b)} \\ \midrule two\par lines & 1 \\ \bottomrule\end{tabular}\end{table}")
    md = convert(tmp_path, body)
    assert "<table" not in md and "<td" not in md and "| **Model** " in md


def test_a_figure_that_holds_a_text_box_keeps_the_text_and_its_caption(tmp_path):
    body = (r"\begin{figure}\begin{lstlisting}" "\nGiven an API, evaluate it and assign a score from 0 to 10.\n"
            r"\end{lstlisting}\caption{Evaluation prompt.}\end{figure}")
    md = convert(tmp_path, body)
    assert "Given an API, evaluate it" in md and "**Figura 1.** Evaluation prompt." in md


def test_a_figure_with_only_images_is_still_just_its_caption(tmp_path):
    md = convert(tmp_path, r"\begin{figure}\includegraphics{a.png}\caption{Only a picture.}\end{figure}")
    assert md.strip() == "**Figura 1.** Only a picture."


def test_a_figure_that_is_a_grid_of_images_stays_a_caption(tmp_path):
    body = (r"\begin{figure}\begin{tabular}{cc}\includegraphics{a.png}&\includegraphics{b.png}\\"
            r"\includegraphics{c.png}&\includegraphics{d.png}\end{tabular}\caption{Four panels.}\end{figure}")
    md = convert(tmp_path, body)
    assert md.strip() == "**Figura 1.** Four panels."


def test_the_caption_lines_of_the_panels_are_kept_with_the_figure(tmp_path):
    body = (r"\begin{figure}\includegraphics{a.png}\par\textit{(a) Sub one}\par \includegraphics{b.png}\par\textit{(b) Sub two}\par"
            r"\caption{Main cap}\end{figure}")
    md = convert(tmp_path, body)
    assert "*(a) Sub one*" in md and "*(b) Sub two*" in md and "**Figura 1.** Main cap" in md
    assert md.index("(a) Sub one") < md.index("(b) Sub two") < md.index("Main cap")


def test_sample_texts_boxed_in_a_tabular_inside_a_figure_are_kept_in_full(tmp_path):
    sample = "First paragraph of the sample text, long enough to be real text and not a label of an image.\n\nSecond paragraph of the sample, again with plenty of words in it.\n\nThird one that ends the sample text."
    body = r"\begin{figure}\begin{tabular}{c}{" + sample + r"}\\\end{tabular}\caption{Samples from X}\label{f:s}\end{figure}"
    md = convert(tmp_path, body, labels={"f:s": {"num": "17", "type": "figure"}})
    assert "First paragraph of the sample text" in md and "Third one that ends the sample text." in md
    assert captions(md) == ["Figura 17"] and "<table" not in md


def test_a_one_cell_box_around_paragraphs_is_just_those_paragraphs(tmp_path):
    body = r"\begin{table}\caption{Prompt}\begin{tabular}{c}{Please answer the question below.\par Then explain why.}\end{tabular}\end{table}"
    md = convert(tmp_path, body)
    assert md == "**Tabla 1.** Prompt\n\nPlease answer the question below.\n\nThen explain why.\n"


def test_a_tabular_nested_in_a_cell_is_part_of_that_cell_and_gets_no_caption_line(tmp_path):
    body = (r"\begin{table}\caption{Outer}\begin{tabular}{cc}\includegraphics{a} & \includegraphics{b} \\"
            r"\begin{tabular}{l}(a) Linear \\ $X_t$\end{tabular} & \begin{tabular}{l}(b) Rectified\end{tabular} \\ plain & cell \\ \end{tabular}\end{table}")
    md = convert(tmp_path, body)
    assert captions(md) == ["Tabla 1"] and "<table" not in md
    assert r"(a) Linear $X_t$" in md.replace(r"\(a\)", "(a)") and "plain" in md


def test_a_listing_keeps_its_language_and_gets_its_caption_with_the_pdf_number(tmp_path):
    body = (r"\begin{lstlisting}[language=Python, caption={\textbf{My} cap}, label={lst:x}]" "\ndef f(): pass\n" r"\end{lstlisting}"
            "\n\n" r"\begin{lstlisting}[style=prompt]" "\nA prompt $5\n" r"\end{lstlisting}")
    md = convert(tmp_path, body, labels={"lst:x": {"num": "2", "type": "lstlisting"}})
    assert "**Listado 2.** **My** cap" in md and "``` python\ndef f(): pass\n```" in md
    assert "style=" not in md and "{.python" not in md and "language=" not in md


def test_a_cell_with_line_breaks_stays_in_a_pipe_table_with_br(tmp_path):
    body = (r"\begin{table}\caption{Prompts}\begin{tabular}{ll}\toprule Use & Prompt \\ \midrule "
            r"Rewrite & Summary:\newline ``''\newline body \\ \bottomrule\end{tabular}\end{table}")
    md = convert(tmp_path, body)
    assert "<table" not in md and "| Rewrite " in md and "<br>" in md


def test_a_table_without_a_header_row_gets_its_first_row_as_header(tmp_path):
    body = (r"\begin{table}\caption{No header}\begin{tabular}{lcc} Model & A & B \\ Ours & 1 & 2 \\ \end{tabular}\end{table}"
            r"\begin{table}\caption{Spaced}\begin{tabular}{lcc} & & \\ Model & A & B \\ Ours & 1 & 2 \\ \end{tabular}\end{table}")
    md = convert(tmp_path, body)
    assert md.count("| Model |") == 2 and "|       |     |     |" not in md
    assert md.count("|:------|:---:|:---:|") == 2


def sec(level, num, title):
    return {"level": level, "num": num, "text": re.sub(r"[^a-z0-9]", "", title.lower())[:40]}


def test_headers_take_the_numbers_of_the_table_of_contents_even_below_a_part(tmp_path):
    body = (r"\part{Main}\section{Introduction}\paragraph{Setup}x\section{Method}\subsection{Setup}y"
            r"\section{Infrastructure}")
    md = convert(tmp_path, body, sections=[sec(1, "1", "Introduction"), sec(1, "2", "Method"),
                                           sec(2, "2.1", "Setup"), sec(1, "A", "Infrastructure")])
    heads = re.findall(r"^#+ (.*)$", md, re.M)
    # (the \\part itself still gets a counted number: the toc entries are sections only)
    assert heads[1:] == ["1 Introduction", "Setup", "2 Method", "2.1 Setup", "A Infrastructure"]
