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
