"""Float normalisation rules: each case is a real structure that made pandoc drop a caption."""
import re

from tex2md.preprocess.floats import (
    dissolve_multi_caption_floats,
    flatten_subfloats,
    floatrow_to_floats,
    retype_floats,
    split_captioned_blocks,
    split_multi_caption_floats,
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


def test_a_group_after_the_end_of_a_minipage_is_text_and_stays():
    from tex2md.preprocess.floats import strip_float_layout
    src = "\\begin{minipage}{0.5\\textwidth}A\\end{minipage}\n{\\small Source: X.}\n\\begin{minipage}[t][3cm][c]{0.4\\linewidth}B\\end{minipage}"
    assert strip_float_layout(src) == "A\n{\\small Source: X.}\nB"


def test_a_float_that_only_places_algorithms_side_by_side_is_dropped():
    from tex2md.preprocess.floats import unwrap_algorithm_floats
    src = (r"\begin{table}\begin{minipage}{.5\textwidth}\begin{algorithm}\caption{One}x\end{algorithm}\end{minipage}"
           r"\begin{algorithm}\caption{Two}y\end{algorithm}\end{table} \begin{table}\caption{Real}\begin{tabular}{c}1\end{tabular}\end{table}")
    assert unwrap_algorithm_floats(src) == (
        r"\begin{minipage}{.5\textwidth}\begin{algorithm}\caption{One}x\end{algorithm}\end{minipage}"
        r"\begin{algorithm}\caption{Two}y\end{algorithm} "
        r"\begin{table}\caption{Real}\begin{tabular}{c}1\end{tabular}\end{table}")


def test_a_paragraph_break_inside_a_caption_becomes_a_blank():
    from tex2md.preprocess.floats import single_paragraph_captions
    src = ("\\begin{table}\\caption[]{\\small First.\n\\vspace{0.4em}\n\nOriginal: {\\footnotesize \\url{http://a.b}}\\par Last.}\n\n"
           "Body text keeps its break.\n\\captionof{figure}{A\n\nB} \\caption*{C\n \nD}\\subcaption{E\n\nF}\\end{table}")
    out = single_paragraph_captions(src)
    assert "First.\n\\vspace{0.4em} Original: {\\footnotesize \\url{http://a.b}} Last.}" in out
    assert "}\n\nBody text keeps its break." in out
    assert "{A B}" in out and "{C D}" in out and "{E F}" in out


def test_a_list_inside_a_caption_becomes_inline_items():
    from tex2md.preprocess.floats import single_paragraph_captions
    src = ("\\caption[]{Notes:\n\\begin{itemize}[leftmargin=*]\n\\item First point.\n\\item Second point.\n\\end{itemize}\n"
           "and \\begin{enumerate}\\item one \\item two\\end{enumerate} end}")
    out = single_paragraph_captions(src)
    assert "itemize" not in out and "enumerate" not in out and "\\item" not in out
    assert " \u2022 First point." in out and " \u2022 Second point." in out and " (1) one" in out and " (2) two" in out


def test_a_tabular_inside_a_caption_becomes_a_line_of_cells():
    from tex2md.preprocess.floats import single_paragraph_captions
    src = "\\caption{Models. See Fig. 2.\n\\begin{tabular}{ccc}\n* {\\scriptsize a photo} &\n$\\dagger$ {\\scriptsize a zoomed} \\\\ x & y\n\\end{tabular}\n}"
    out = single_paragraph_captions(src)
    assert "tabular" not in out and "* {\\scriptsize a photo}; $\\dagger$ {\\scriptsize a zoomed}; x; y" in out


def test_two_captions_in_one_float_become_two_floats():
    from tex2md.preprocess.floats import dissolve_multi_caption_floats, wrap_loose_captions
    src = (r"\begin{figure}[t]\centering\includegraphics{a}\caption{First.}\label{f:a}\par\bigskip"
           r"\includegraphics{b}\caption{Second.}\label{f:b}\end{figure}")
    out = wrap_loose_captions(dissolve_multi_caption_floats(src))
    assert out.count(r"\begin{figure}") == 2 and r"\caption{First.}" in out and r"\caption{Second.}" in out
    assert r"\begin{figure}[t]" not in out        # the original float is gone


def test_a_float_with_subfloats_or_a_single_caption_is_left_alone():
    from tex2md.preprocess.floats import dissolve_multi_caption_floats
    sub = r"\begin{figure}\begin{subfigure}{.5\textwidth}\caption{panel}\end{subfigure}\caption{main}\end{figure}"
    one = r"\begin{figure}\includegraphics{a}\caption{Only}\caption*{(a) panel line}\end{figure}"
    assert dissolve_multi_caption_floats(sub) == sub and dissolve_multi_caption_floats(one) == one


def test_captions_in_their_own_minipages_below_the_pictures_become_one_float_each():
    from tex2md.preprocess.floats import split_captioned_blocks
    src = (r"\begin{figure}\centering\begin{minipage}[t]{0.48\linewidth}\includegraphics[width=\linewidth]{a.png}\end{minipage}%"
           r"\hspace{.02\textwidth}\begin{minipage}[t]{0.48\linewidth}\includegraphics{b.png}\end{minipage}\par\vspace*{-5pt}"
           r"\begin{minipage}[t]{0.48\linewidth}\caption{First caption.}\label{f:1}\end{minipage}%"
           r"\hspace{.02\textwidth}\begin{minipage}[t]{0.48\linewidth}\caption{Second caption.}\label{f:2}\end{minipage}\end{figure}")
    out = split_captioned_blocks(src)
    assert out.count(r"\begin{figure}") == 2 and r"\caption{First caption.}" in out and r"\caption{Second caption.}" in out


def test_a_block_with_text_and_no_caption_keeps_the_float_whole():
    from tex2md.preprocess.floats import split_captioned_blocks
    src = (r"\begin{figure}\begin{minipage}{.5\linewidth}Some explanation text for the reader.\end{minipage}"
           r"\begin{minipage}{.5\linewidth}\caption{A}\end{minipage}\begin{minipage}{.5\linewidth}\caption{B}\end{minipage}\end{figure}")
    assert split_captioned_blocks(src) == src


def test_a_caption_after_a_centered_tabular_takes_the_tabular_with_its_center():
    from tex2md.preprocess.floats import wrap_loose_captions
    src = ("\\small\n\\begin{center}\n\\begin{tabular}{lc}\na & 1\\\\\n\\end{tabular}\n\\end{center}\n\\vspace{-.5em}\n"
           "\\captionof{table}{Error rates.}\n\\label{tab:x}\nafter")
    out = wrap_loose_captions(src)
    assert out.startswith("\\small\n\\begin{table}\\begin{center}") and "\\end{center}" in out.split("\\caption")[0]
    assert "\\caption{Error rates.}\\end{table}" in out and out.count("\\begin{center}") == out.count("\\end{center}") == 1


def test_a_caption_after_a_resized_tabular_takes_the_whole_wrapper():
    from tex2md.preprocess.floats import wrap_loose_captions
    src = ("\\begin{center}\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lc}\na & 1\\\\\n\\end{tabular}\n}\n\\end{center}\n"
           "\\vspace{-.5em}\n\\captionof{table}{Detection results.}\nafter")
    out = wrap_loose_captions(src)
    assert out.startswith("\\begin{table}\\begin{center}\n\\resizebox") and out.endswith("\\end{table}\nafter")
    inner = out[:out.index("\\caption")]
    assert inner.count("{") == inner.count("}") and "\\end{center}" in inner


def test_a_subfigure_caption_numbered_by_the_document_stays_a_caption_of_its_own():
    from tex2md.preprocess.floats import flatten_subfloats, unnest_floats
    src = (r"\begin{figure}\begin{subfigure}{.9\textwidth}\includegraphics{a}\caption{The comparison across MMLU categories.}"
           r"\end{subfigure}\begin{subfigure}{.9\textwidth}\includegraphics{b}\caption{Across MMLU-Pro categories.}\end{subfigure}"
           r"\label{f:both}\end{figure}")
    floats = [{"kind": "figure", "num": "15", "text": "thecomparisonacrossmmlucategories"},
              {"kind": "figure", "num": "16", "text": "acrossmmluprocategories"}]
    out = unnest_floats(flatten_subfloats(src, floats))
    assert out.count(r"\begin{figure}") == 2 and r"\caption{The comparison across MMLU categories.}" in out
    assert "textit" not in out
    # without a matching entry in the .aux they are plain panel captions, as before
    plain = flatten_subfloats(src, [])
    assert r"\par\textit{(a) The comparison across MMLU categories.}\par" in plain and r"\par\textit{(b) Across MMLU-Pro" in plain


def test_sub_captions_stay_panel_captions_when_the_float_has_a_caption_of_its_own():
    from tex2md.preprocess.floats import flatten_subfloats
    src = (r"\begin{figure*}\begin{subfigure}{.5\textwidth}\caption{ImageNet Top-1 Gains}\end{subfigure}"
           r"\caption{(a) Gains. (b) Quality.}\end{figure*}")
    floats = [{"kind": "figure", "num": "5", "text": "imagenettop1gainsacrossmodels"}]    # another figure that starts alike
    out = flatten_subfloats(src, floats)
    assert r"\par\textit{(a) ImageNet Top-1 Gains}\par" in out and out.count(r"\begin{figure") == 1


def test_a_figure_captioned_as_a_table_with_captionsetup_is_a_table():
    from tex2md.preprocess.floats import retype_floats
    src = (r"\begin{figure}\includegraphics{a}\captionsetup{type=table,skip=-6pt}\caption{(Induction Heads.) Models.}\label{f:i}\end{figure}"
           r"\begin{figure}\includegraphics{b}\captionsetup[sub]{type=table}\caption{Kept}\end{figure}")
    out = retype_floats(src)
    assert out.startswith(r"\begin{table}\includegraphics{a}") and r"\end{table}\begin{figure}" in out
    assert out.count(r"\begin{figure}") == 1


def test_blocks_side_by_side_keep_the_kind_their_captionsetup_gives_them():
    from tex2md.preprocess.floats import split_captioned_blocks
    src = (r"\begin{figure*}\begin{minipage}{.5\linewidth}\begin{tabular}{l}a\end{tabular}\captionsetup{type=table}\caption{Selective}"
           r"\end{minipage}\begin{minipage}{.5\linewidth}\includegraphics{b}\captionsetup{type=table,skip=-6pt}\caption{Induction}"
           r"\end{minipage}\end{figure*}")
    out = split_captioned_blocks(src)
    assert out.count(r"\begin{table}") == 2 and r"\begin{figure" not in out


def test_a_block_that_becomes_a_table_without_tabular_is_made_a_marked_figure_after_the_split():
    from tex2md.preprocess.floats import split_captioned_blocks, tables_without_tabular
    src = (r"\begin{figure*}\begin{minipage}{.5\linewidth}\includegraphics{a}\captionsetup{type=table}\caption{Induction}"
           r"\end{minipage}\begin{minipage}{.5\linewidth}\begin{tabular}{l}x\end{tabular}\captionsetup{type=table}\caption{Selective}"
           r"\end{minipage}\end{figure*}")
    out = tables_without_tabular(split_captioned_blocks(src))
    assert r"\caption{TEX2MDTABLE Induction}" in out and r"\begin{table}\begin{tabular}" in out


def test_loose_captions_each_take_the_table_before_them_when_the_tables_alternate_with_captions():
    from tex2md.preprocess.floats import wrap_loose_captions
    src = (r"\begin{NiceTabular}{ll}a&b\end{NiceTabular}\captionof{table}{First}\label{t1}"
           r"\resizebox{.5\textwidth}{!}{\begin{tabular}{ll}c&d\end{tabular}}\captionof{table}{Second}\label{t2}")
    out = wrap_loose_captions(src)
    first, second = out.split(r"\caption{Second}")[0], out
    assert out.count(r"\begin{table}") == 2
    assert r"\begin{table}\begin{NiceTabular}{ll}a&b\end{NiceTabular}\caption{First}" in out
    assert r"\begin{table}\resizebox{.5\textwidth}{!}{\begin{tabular}{ll}c&d\end{tabular}}\caption{Second}" in out
    assert first and second


def test_a_caption_above_its_table_still_takes_the_table_that_follows():
    from tex2md.preprocess.floats import wrap_loose_captions
    src = r"\captionof{table}{First}\begin{tabular}{l}a\end{tabular}\captionof{table}{Second}\begin{tabular}{l}b\end{tabular}"
    out = wrap_loose_captions(src)
    assert r"\begin{table}\caption{First}\begin{tabular}{l}a\end{tabular}\end{table}" in out
    assert r"\begin{table}\caption{Second}\begin{tabular}{l}b\end{tabular}\end{table}" in out


def test_wrapstuff_and_teaserfigure_become_floats():
    src = (r"\begin{teaserfigure}\includegraphics{t}\caption{Teaser}\end{teaserfigure}"
           r"\begin{wrapstuff}[r, width=0.5\linewidth, type=table]\caption{T}\begin{tabular}{c}1\end{tabular}\end{wrapstuff}"
           r"\begin{wrapstuff}[l]Just a boxed note.\end{wrapstuff}")
    out = wrap_envs_to_plain(src)
    assert count(out, "figure") == 1 and count(out, "table") == 1 and "wrapstuff" not in out and "teaserfigure" not in out
    assert "Just a boxed note." in out


def test_captionless_wrapfigure_inside_a_float_is_its_content():
    # allenzhu2023: a box of text beside the picture; as a float it closed the outer one before its caption
    src = (r"\begin{figure*}\begin{wrapfigure}{l}{0.3\textwidth}Side text.\end{wrapfigure}"
           r"\includegraphics{a}\caption{Main}\end{figure*}")
    out = wrap_envs_to_plain(src)
    assert out == r"\begin{figure*}Side text.\includegraphics{a}\caption{Main}\end{figure*}"


def test_a_captioned_table_under_two_captioned_minipages_is_a_float_too():
    # wang2026computegap: table and plot side by side, a second table below them in the same figure
    src = (r"\begin{figure}\begin{minipage}{0.4\linewidth}\captionof{table}{Laws}\begin{tabular}{c}1\end{tabular}\end{minipage}"
           r"\hfill\begin{minipage}{0.5\linewidth}\includegraphics{p}\captionof{figure}{Plot}\end{minipage}"
           r"\vspace{4pt}\centering\captionof{table}{Multiples}\begin{tabular}{c}2\end{tabular}\end{figure}")
    out = split_captioned_blocks(src)
    assert count(out, "table") == 2 and count(out, "figure") == 1
    assert out.index("Laws") < out.index("Plot") < out.index("Multiples")


def test_a_minipage_with_two_captioned_tables_is_split():
    # lu2025adablock: two tables stacked in the left minipage, one in the right, all in a center environment
    left = (r"\captionsetup{type=table}\caption{First}\begin{tabular}{c}1\end{tabular}"
            r"\captionsetup{type=table}\caption{Second}\begin{tabular}{c}2\end{tabular}")
    src = (r"\begin{table}\begin{center}\begin{minipage}[t]{0.6\linewidth}" + left + r"\end{minipage}\hfill"
           r"\begin{minipage}[t]{0.3\linewidth}\captionsetup{type=table}\caption{Third}\begin{tabular}{c}3\end{tabular}"
           r"\end{minipage}\end{center}\end{table}")
    out = split_captioned_blocks(src)
    tables = re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", out, re.S)
    assert [re.search(r"\\caption\{(\w+)\}", t).group(1) for t in tables] == ["First", "Second", "Third"]
    assert all(t.count(r"\caption{") == 1 for t in tables)


def test_a_float_with_two_captions_left_by_flattened_panels_is_split():
    # diffusiongemma2026: a prompt box and its caption, then two panels and theirs
    src = (r"\begin{figure}[H]\begin{quote}Prompt\end{quote}\caption{The prompt}\label{fig:p}"
           r"\includegraphics{a}\includegraphics{b}\caption{The panels}\label{fig:q}\end{figure}")
    out = split_multi_caption_floats(src)
    figures = re.findall(r"\\begin\{figure\}(.*?)\\end\{figure\}", out, re.S)
    assert len(figures) == 2
    assert "The prompt" in figures[0] and r"\label{fig:p}" in figures[0] and "Prompt" in figures[0]
    assert "The panels" in figures[1] and r"\includegraphics{a}" in figures[1]


def test_floatrow_boxes_become_floats():
    # kingma2021vdm: a table and a plot side by side, each box with its caption
    src = (r"\newfloatcommand{capbtabbox}{table}[][0.48\textwidth]" "\n"
           r"\begin{figure}[t]\begin{floatrow}\capbtabbox{\begin{tabular}{c}1\end{tabular}}{\caption{Results}\label{t}}"
           r"\ffigbox{\includegraphics{f}}{\caption{Ablation}\label{f}}\end{floatrow}\end{figure}")
    out = floatrow_to_floats(src)
    assert "floatrow" not in out.split("\n", 1)[1] and count(out, "table") == 1 and count(out, "figure") == 1
    assert re.search(r"\\begin\{table\}.*Results.*\\end\{table\}\s*\\begin\{figure\}.*Ablation.*\\end\{figure\}", out, re.S)


def test_minipages_with_subcaptions_or_algorithms_stay_one_figure():
    # wu2025fastdllmv2: \caption*{(a) ...} in each minipage; han2022ssdlm: an algorithm (and its caption) in each
    for panel in (r"\includegraphics{a}\caption*{(a) Left.}", r"\begin{algorithm}[H]\caption{Training}x\end{algorithm}"):
        src = (r"\begin{figure*}\begin{minipage}[t]{0.48\textwidth}" + panel + r"\end{minipage}\hfill"
               r"\begin{minipage}[t]{0.48\textwidth}" + panel + r"\end{minipage}\vspace{0.5em}\caption{Whole figure.}\end{figure*}")
        assert split_captioned_blocks(src) == src
        assert split_multi_caption_floats(src) == src


def test_a_minipage_with_subfigures_is_not_split():
    # ou2024radd: two captioned sub-figures in the right minipage are panels of one figure
    src = (r"\begin{figure}\begin{minipage}{0.4\linewidth}\includegraphics{n}\caption{Network}\end{minipage}"
           r"\begin{minipage}{0.55\linewidth}\begin{subfigure}{\linewidth}\includegraphics{a}\caption{A}\end{subfigure}"
           r"\begin{subfigure}{\linewidth}\includegraphics{b}\caption{B}\end{subfigure}\end{minipage}\end{figure}")
    out = split_captioned_blocks(src)
    assert "\\caption{A}" in out and "\\caption{B}" in out and out.count(r"\begin{subfigure}") == 2


def test_every_caption_of_a_panel_becomes_a_sub_caption():
    # xie2025dreamcoder: the second panel holds two pictures, each with its caption
    src = (r"\begin{figure}\begin{subfigure}{0.48\linewidth}\includegraphics{a}\caption{First}\end{subfigure}"
           r"\begin{subfigure}{0.48\linewidth}\includegraphics{b}\caption{Second}\includegraphics{c}\caption{Third}"
           r"\end{subfigure}\caption{All}\end{figure}")
    out = flatten_subfloats(src)
    assert re.findall(r"\\textit\{\((\w)\) (\w+)\}", out) == [("a", "First"), ("b", "Second"), ("c", "Third")]
    assert out.count(r"\caption{") == 1


def test_row_of_tables_then_row_of_captions_are_paired():
    from tex2md.preprocess.floats import pair_content_with_captions
    tex = (r"\begin{table*}\centering"
           r"\begin{minipage}{0.4\textwidth}\begin{tabular}{l}A\end{tabular}\end{minipage}\hfill"
           r"\begin{minipage}{0.4\textwidth}\begin{tabular}{l}B\end{tabular}\end{minipage}\\"
           r"\begin{minipage}{0.4\textwidth}\caption{CA}\end{minipage}\hfill"
           r"\begin{minipage}{0.4\textwidth}\caption{CB}\end{minipage}\end{table*}")
    out = pair_content_with_captions(tex)
    assert out.index("A\\end{tabular}") < out.index("\\caption{CA}") < out.index("B\\end{tabular}") < out.index("\\caption{CB}")


def test_teaser_drawn_by_a_redefined_maketitle():
    from tex2md.preprocess.floats import teaser_in_maketitle
    tex = (r"\renewcommand{\@maketitle}{\old@maketitle \includegraphics{t.pdf}\captionof{figure}{Teaser \label{fig:t}}}"
           "\\begin{document}\\maketitle\nText")
    out = teaser_in_maketitle(tex)
    assert "\\maketitle\n\\begin{figure}\n\\caption{Teaser \\label{fig:t}}\n\\end{figure}\n\nText" in out
