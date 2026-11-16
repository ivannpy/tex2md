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


def convert(tmp_path, body, labels=None, floats=None):
    (tmp_path / "in.tex").write_text("\\documentclass{article}\\usepackage{graphicx}\\begin{document}\n" + body + "\n\\end{document}")
    (tmp_path / "labels.json").write_text(json.dumps(labels or {}))
    (tmp_path / "floats.json").write_text(json.dumps(floats or []))
    p = subprocess.run(["pandoc", "in.tex", "-f", "latex", "-t", FMT, "--wrap=none", "--lua-filter", FILTER,
                        "-M", "tex2md_labels=labels.json", "-M", "tex2md_floats=floats.json"],
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
