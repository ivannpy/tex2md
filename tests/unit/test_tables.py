from tex2md.preprocess.tables import normalize_tabulars, simplify_longtable_specs


def test_longtable_spec_from_pandoc_becomes_plain_columns():
    src = (r"\begin{longtable}[]{@{}" "\n" r"  >{\raggedright\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.0949}}" "\n"
           r"  >{\raggedright\arraybackslash}p{(\linewidth - 8\tabcolsep) * \real{0.2044}}@{}}" "\n" r"\caption{C}\tabularnewline")
    out = simplify_longtable_specs(src)
    assert out.startswith(r"\begin{longtable}{") and r"\real" not in out and r"\caption{C}" in out


def test_longtable_with_simple_spec_is_kept():
    assert simplify_longtable_specs(r"\begin{longtable}{lcc}x") == r"\begin{longtable}{lcc}x"


def test_longtable_repeated_header_and_foot_commands_are_dropped():
    src = (r"\begin{longtable}{ll}\caption{C}\tabularnewline\toprule A & B \\ \midrule\endfirsthead"
           r"\toprule A & B \\ \midrule\endhead\bottomrule\endlastfoot" "\n2025-01-22 & x \\\\\n\\end{longtable}")
    out = simplify_longtable_specs(src)
    assert out.count("A & B") == 1 and "endlastfoot" not in out and "endhead" not in out


def test_column_types_with_arguments_reduce_to_plain_letters():
    from tex2md.preprocess.tables import simple_colspec
    assert simple_colspec("lS[table-format=2.1]S") == "lcc"
    assert simple_colspec("l w{c}{2cm} X[2,l] c") == "lccc"
    assert simple_colspec(r"@{}l*{3}{c}@{}") == "lccc"
    assert simple_colspec("l|c||r") == "lcr"
    assert simple_colspec(r"lD{.}{.}{2}") == "lc"
    assert simple_colspec(r">{\raggedright\arraybackslash}p{3cm}Y{2cm}cc") == "lccc"


def test_a_plain_tabular_keeps_the_group_that_follows_its_spec():
    from tex2md.preprocess.tables import normalize_tabulars
    # "{\bf A}" is the first cell, not another argument: it must not be read as a column spec
    assert normalize_tabulars("\\begin{tabular}{lcc}{\\bf A} & b & c\\\\\\end{tabular}") == \
        "\\begin{tabular}{lcc}{\\bf A} & b & c\\\\\\end{tabular}"
    assert normalize_tabulars("\\begin{tabular} [t] {l|c}\na & b\\\\\\end{tabular}") == "\\begin{tabular} [t] {lc}\na & b\\\\\\end{tabular}"
    assert normalize_tabulars("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lS[x=1]}a\\end{tabular*}") == \
        "\\begin{tabular*}{\\textwidth}{lc}a\\end{tabular*}"
    assert normalize_tabulars("x & \\begin{tabular}{@{}c@{}}a\\\\b\\end{tabular} & \\makecell{c\\\\d}") == "x & a b & {c d}"


def test_an_ampersand_inside_a_citation_in_a_cell_is_not_a_column_break():
    from tex2md.preprocess.tables import escape_citation_ampersands
    src = r"\begin{tabular}{cc} Adam (Kingma & Ba (2015)) & x \\ (Wang & Cho, 2019) & y \\ a & b \\ \end{tabular}"
    assert escape_citation_ampersands(src) == (
        r"\begin{tabular}{cc} Adam (Kingma \& Ba (2015)) & x \\ (Wang \& Cho, 2019) & y \\ a & b \\ \end{tabular}")


def test_a_plain_tex_row_end_inside_a_tabular_becomes_a_latex_one():
    from tex2md.preprocess.tables import cr_to_row_end
    src = r"\begin{longtable}{lc} a & 1 \cr b & 2 \cr \end{longtable} outside \cr \begin{array}{c} x \cr y \end{array}"
    assert cr_to_row_end(src) == r"\begin{longtable}{lc} a & 1 \\ b & 2 \\ \end{longtable} outside \cr \begin{array}{c} x \\ y \end{array}"


def test_blanks_between_the_arguments_of_multicolumn_and_multirow_are_removed():
    from tex2md.preprocess.tables import tighten_span_arguments
    src = ("\\multicolumn{1}{l}\n{Text} & \\multicolumn {2} {c}\n\n {B \\& C} & \\multirow{2}\n{*}\n{D} & \\multirow[t]{2}{3cm}\n{E}")
    assert tighten_span_arguments(src) == ("\\multicolumn{1}{l}{Text} & \\multicolumn{2}{c}{B \\& C} & \\multirow{2}{*}{D} & "
                                           "\\multirow[t]{2}{3cm}{E}")
    assert tighten_span_arguments("\\multicolumn{2}{c}") == "\\multicolumn{2}{c}"      # incomplete: left alone
    assert tighten_span_arguments("\\multicolumn{1}{c}{\\multirow{2}{*}} & x") == "\\multicolumn{1}{c}{\\multirow{2}{*}{}} & x"


def test_column_spec_grows_to_the_widest_row():
    # deschenaux2026duo2: "llllc" for rows of 12 cells; pandoc dropped the table and its caption
    src = r"\begin{tabular}{llc}A & B & \multicolumn{3}{c}{FID} \\ 1 & 2 & 3 & 4 & 5 \\ \end{tabular}"
    assert normalize_tabulars(src).startswith(r"\begin{tabular}{llccc}")


def test_citation_ampersands_are_not_cells():
    src = r"\begin{tabular}{lc}BERT (Wang & Cho, 2019) & 1 \\ x \& y & 2 \\ \end{tabular}"
    assert normalize_tabulars(src).startswith(r"\begin{tabular}{lc}")


def test_rows_ended_by_tabularnewline_math_and_macro_rows_are_counted_right():
    # goodfellow2014gan (\tabularnewline), wang2026scdd ("&" of an aligned), liu2026mmadavla (rows from a macro)
    assert normalize_tabulars(r"\begin{tabular}{lc}a & b\tabularnewline c & d\tabularnewline\end{tabular}").startswith(
        r"\begin{tabular}{lc}")
    assert normalize_tabulars(r"\begin{tabular}{lc}x & $\begin{aligned} a &= b \\ c &= d\end{aligned}$ \\ \end{tabular}").startswith(
        r"\begin{tabular}{lc}")
    assert normalize_tabulars(r"\begin{tabular}{lc}A & B \\ \row{x}{1} \row{y}{2} \row{z}{3}\end{tabular}").startswith(
        r"\begin{tabular}{lc}")


def test_rows_from_a_macro_ending_inside_it_do_not_widen_the_spec():
    # liu2026mmadavla: \row{..} ends each row with "\\" inside its definition; only the last row has its own "\\"
    src = (r"\begin{tabular}{lcc}A & B & C \\ x & y & z \\ \row{a}{1} & \row{b}{2}" "\n" r"\row{c}{3} & \row{d}{4}" "\n"
           r"Ours & 9 & 9 \\ \end{tabular}")
    assert normalize_tabulars(src).startswith(r"\begin{tabular}{lcc}")


def test_tabu_becomes_tabular():
    from tex2md.preprocess.tables import tabu_to_tabular
    out = tabu_to_tabular(r"\begin{tabu} to \linewidth {@{}ll@{}}\tabucline{-} a & b \\ \end{tabu}")
    assert out == r"\begin{tabular}{@{}ll@{}}\hline a & b \\ \end{tabular}"


def test_cell_line_breaks():
    from tex2md.preprocess.tables import cell_line_breaks
    tex = "\\begin{tabular}{ll}\na & b \\\\\n\n\\vspace*{1mm}{\\bf T}\\\\\n\\multicolumn{2}{l}{{x \\\\ y $\\substack{a\\\\b}$}}\n\\end{tabular}"
    out = cell_line_breaks(tex)
    assert "\n\n" not in out and "vspace" not in out
    assert "{{x   y $\\substack{a\\\\b}$}}" in out and "a & b \\\\" in out


def test_multirow_with_bare_star_width():
    from tex2md.preprocess.tables import unwrap_cell_macros
    assert unwrap_cell_macros(r"\multirow{2}*{Channels} & x") == "Channels & x"


def test_pbox_cells_are_one_line():
    from tex2md.preprocess.tables import unwrap_cell_macros
    assert unwrap_cell_macros(r"\newcommand{\eqq}[2]{\pbox[t][5ex]{\linewidth}{{#1}\\{#2}}}") == \
        r"\newcommand{\eqq}[2]{{#1} {#2}}"
