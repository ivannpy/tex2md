from tex2md.preprocess.tables import simplify_longtable_specs


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
