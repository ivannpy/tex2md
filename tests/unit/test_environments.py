from tex2md.preprocess.environments import (
    drop_picture_code,
    expand_glossary,
    restatable_to_theorem,
    tabular_in_math,
    tblr_to_tabular,
    unwrap_multicols,
)


def test_drawing_code_is_not_text():
    src = (r"a \begin{forest} for tree={grow'=east} [Root [A]] \end{forest} b \rotatebox{90}{\tikz\fill[all] (0,0) circle (.5ex); Label} "
           r"c \tikz[baseline]{\node[draw]{A {b}};} d \tikzset{x}")
    assert drop_picture_code(src) == r"a  b \rotatebox{90}{ Label} c  d \tikzset{x}"


def test_multicols_loses_its_column_count():
    assert unwrap_multicols(r"\begin{multicols}{2} text \end{multicols}") == " text "


def test_restatable_becomes_its_theorem():
    src = r"\begin{restatable}[Title]{theorem}{unifiedkl} For all $s$. \end{restatable} \begin{restatable}{lemma}{lem} L \end{restatable}"
    assert restatable_to_theorem(src) == r"\begin{theorem}[Title] For all $s$. \end{theorem} \begin{lemma} L \end{lemma}"


def test_tabularray_tables_become_plain_tabulars():
    src = r"\begin{tblr}{colspec={lcc}, hlines} Model & A & B \\ Ours & 1 & 2 \\ \end{tblr}"
    assert tblr_to_tabular(src) == r"\begin{tabular}{lcc} Model & A & B \\ Ours & 1 & 2 \\ \end{tabular}"


def test_glossary_terms_are_spelled_out_the_first_time_only():
    src = (r"\newacronym{llm}{LLM}{large language model} \begin{document} \gls{llm} and \gls{llm}, \glspl{llm}, \Gls{llm}, "
           r"\acrshort{llm}, \acrfull{llm}")
    out = expand_glossary(src)
    assert out.endswith("large language model (LLM) and LLM, LLMs, LLM, LLM, large language model (LLM)")


def test_a_tabular_used_as_a_piecewise_definition_inside_a_formula_becomes_an_array():
    src = (r"$\beta = \left\lbrace" "\n" r"\begin{tabular}{@{}p{0.5cm}p{1.8cm}} " "\n" r"$1$ & $q(x) \le p(x)$ \\" "\n"
           r"$\frac{p}{q}$ & $q(x) > p(x)$ \\" "\n" r"\end{tabular}\right. = E$ and a normal \begin{tabular}{ll} $a$ & b \end{tabular}")
    out = tabular_in_math(src)
    assert r"\left\lbrace\begin{array}{ll}" in out and r"\end{array}\right. = E$" in out
    assert "$1$" not in out and r"1 & q(x) \le p(x)" in out
    assert r"\begin{tabular}{ll} $a$ & b \end{tabular}" in out          # an ordinary table is untouched
