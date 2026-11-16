"""Float normalisation rules: each case is a real structure that made pandoc drop a caption."""
import re

from tex2md.preprocess.floats import (
    dissolve_multi_caption_floats,
    flatten_subfloats,
    retype_floats,
    split_captioned_blocks,
    tables_without_tabular,
    unnest_floats,
    unnest_tables,
    wrap_envs_to_plain,
    wrap_loose_captions,
)


def count(text, env):
    return len(re.findall(r"\\begin\{" + env + r"\}", text))


def test_split_two_captioned_minipages_into_two_floats():
    src = (r"\begin{table}[t]\begin{minipage}{.5\textwidth}\caption{A}\begin{tabular}{c}1\end{tabular}\end{minipage}"
           r"\hfill\begin{minipage}{.5\textwidth}\caption{B}\begin{tabular}{c}2\end{tabular}\end{minipage}\end{table}")
    out = split_captioned_blocks(src)
    assert count(out, "table") == 2
    assert r"\caption{A}" in out and r"\caption{B}" in out


def test_split_ignores_parbox_inside_cells():
    cell = r"\parbox{2cm}{x}"
    src = (r"\begin{table}\begin{minipage}{.5\textwidth}\caption{A}\begin{tabular}{c}" + cell + r"\end{tabular}\end{minipage}"
           r"\begin{minipage}{.5\textwidth}\caption{B}\begin{tabular}{c}2\end{tabular}\end{minipage}\end{table}")
    assert count(split_captioned_blocks(src), "table") == 2


def test_split_leaves_float_alone_when_a_block_has_no_caption():
    src = (r"\begin{figure}\begin{minipage}{.5\textwidth}\includegraphics{a}\end{minipage}"
           r"\begin{minipage}{.5\textwidth}\caption{B}\end{minipage}\end{figure}")
    assert split_captioned_blocks(src) == src


def test_split_never_drops_content_outside_the_blocks():
    src = (r"\begin{table}Some real text\begin{minipage}{.5\textwidth}\caption{A}\end{minipage}"
           r"\begin{minipage}{.5\textwidth}\caption{B}\end{minipage}\end{table}")
    assert split_captioned_blocks(src) == src


def test_split_uses_the_kind_given_by_captionof():
    src = (r"\begin{figure}\begin{minipage}{.5\textwidth}\captionof{table}{T}\begin{tabular}{c}1\end{tabular}\end{minipage}"
           r"\begin{minipage}{.5\textwidth}\captionof{figure}{F}\includegraphics{a}\end{minipage}\end{figure}")
    out = split_captioned_blocks(src)
    assert count(out, "table") == 1 and count(out, "figure") == 1


def test_flatten_subfloats_turns_subcaptions_into_lettered_lines():
    src = (r"\begin{table}\caption{Outer}\begin{subtable}{\textwidth}\caption{First}\begin{tabular}{c}1\end{tabular}\end{subtable}"
           r"\begin{subtable}{\textwidth}\caption{Second}\begin{tabular}{c}2\end{tabular}\end{subtable}\end{table}")
    out = flatten_subfloats(src)
    assert "subtable" not in out
    assert r"\textit{(a) First}" in out and r"\textit{(b) Second}" in out
    assert r"\caption{Outer}" in out


def test_wrap_loose_captionof_joins_the_following_tabular_through_adjustbox_and_label():
    src = (r"\begin{center}\captionof{table}{Licenses}\label{tab:l}"
           "\n" r"\begin{adjustbox}{max width=\textwidth}\begin{tabular}{c}x\end{tabular}\end{adjustbox}\end{center}")
    out = wrap_loose_captions(src)
    assert out.index(r"\begin{table}") < out.index(r"\begin{tabular}") < out.index(r"\end{table}")
    assert r"\end{adjustbox}" in out[out.index(r"\begin{table}"):out.index(r"\end{table}")]


def test_wrap_loose_captionof_leaves_captions_inside_floats():
    src = r"\begin{table}\captionof{table}{T}\begin{tabular}{c}1\end{tabular}\end{table}"
    assert wrap_loose_captions(src) == src


def test_retype_figure_holding_a_table_caption():
    out = retype_floats(r"\begin{figure}\captionof{table}{T}\begin{tabular}{c}1\end{tabular}\end{figure}")
    assert out.startswith(r"\begin{table}") and out.endswith(r"\end{table}")


def test_table_with_only_an_image_becomes_a_marked_figure():
    out = tables_without_tabular(r"\begin{table}\includegraphics{a.pdf}\caption{Words}\end{table}")
    assert out.startswith(r"\begin{figure}") and r"\caption{TEX2MDTABLE Words}" in out


def test_table_with_tabular_is_untouched():
    src = r"\begin{table}\caption{T}\begin{tabular}{c}1\end{tabular}\end{table}"
    assert tables_without_tabular(src) == src


def test_text_only_table_keeps_its_content_and_gives_the_caption_its_own_float():
    out = tables_without_tabular(r"\begin{table}\caption{Examples}\begin{tcolorbox}text\end{tcolorbox}\end{table}")
    assert r"\caption{TEX2MDTABLE Examples}" in out
    assert r"\begin{tcolorbox}text\end{tcolorbox}" in out
    assert out.count(r"\caption") == 1


def test_unnest_tables_drops_the_outer_table():
    out = unnest_tables(r"\begin{table}\begin{table}A\end{table}\begin{table}B\end{table}\end{table}")
    assert count(out, "table") == 2


def test_wrap_envs_to_plain():
    assert wrap_envs_to_plain(r"\begin{wrapfigure}{r}{0.4\textwidth}x\end{wrapfigure}") == r"\begin{figure}x\end{figure}"


def test_retype_keeps_a_figure_that_has_its_own_caption_next_to_a_captionof_table():
    src = (r"\begin{figure}\begin{minipage}{.5\textwidth}\caption{F}\end{minipage}"
           r"\begin{minipage}{.5\textwidth}\captionof{table}{T}\begin{tabular}{c}1\end{tabular}\end{minipage}\end{figure}")
    assert retype_floats(src) == src
    out = split_captioned_blocks(src)
    assert count(out, "figure") == 1 and count(out, "table") == 1


def test_dissolve_float_with_a_table_and_a_figure_caption():
    src = (r"\begin{figure}\captionof{table}{T}\label{t}\begin{tabular}{c}1\end{tabular}"
           r"\includegraphics{a}\captionof{figure}{F}\end{figure}")
    out = wrap_loose_captions(dissolve_multi_caption_floats(src))
    assert count(out, "table") == 1 and count(out, "figure") == 1
    assert out.index(r"\begin{table}") < out.index(r"\begin{tabular}") < out.index(r"\end{table}")


def test_dissolve_leaves_single_caption_floats_and_minipage_blocks():
    one = r"\begin{figure}\captionof{table}{T}\begin{tabular}{c}1\end{tabular}\end{figure}"
    assert dissolve_multi_caption_floats(one) == one
    blocks = (r"\begin{figure}\begin{minipage}{.5\textwidth}\captionof{table}{T}\end{minipage}"
              r"\begin{minipage}{.5\textwidth}\captionof{figure}{F}\end{minipage}\end{figure}")
    assert dissolve_multi_caption_floats(blocks) == blocks


def test_unnest_floats_also_unwraps_nested_figures():
    src = (r"\begin{figure}\begin{minipage}{.5\textwidth}\begin{figure}[H]\caption{A}\end{figure}\end{minipage}"
           r"\begin{minipage}{.5\textwidth}\begin{figure}[H]\caption{B}\end{figure}\end{minipage}\end{figure}")
    out = unnest_floats(src)
    assert count(out, "figure") == 2 and r"\caption{A}" in out and r"\caption{B}" in out
