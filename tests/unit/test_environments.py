from tex2md.preprocess.environments import (
    ctable_to_table,
    drop_picture_code,
    expand_glossary,
    restatable_to_theorem,
    tabular_in_math,
    tblr_to_tabular,
    unwrap_multicols,
    unwrap_picture_puts,
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


def test_ctable_becomes_a_table_with_booktabs_rules():
    # chen2024dforcing
    src = ("\\usepackage{ctable}\n" r"\ctable[caption = {Results for time series.}, label = {tab:ts}, pos = ht, star]"
           r"{lcc}{}{\FL Method & A & B \ML X & 1 & 2 \NN Y & 3 & 4 \LL}")
    out = ctable_to_table(src).split("\n", 1)[1]
    assert out.startswith(r"\begin{table}\caption{Results for time series.}\label{tab:ts}\begin{tabular}{lcc}")
    assert r"\toprule" in out and r"\midrule" in out and r"\bottomrule" in out and out.endswith(r"\end{table}")
    assert r"\FL" not in out and r"\NN" not in out


def test_ctable_row_commands_are_left_alone_without_the_package():
    assert ctable_to_table(r"\newcommand{\NN}{\mathbb{N}} $x \in \NN$") == r"\newcommand{\NN}{\mathbb{N}} $x \in \NN$"


def test_picture_puts_keep_their_content():
    src = r"\begin{picture}(\linewidth, 0.6\linewidth)\put(0, 98){\includegraphics{m}}\put(0,43){\begin{tabular}{c}1\end{tabular}}\end{picture}"
    assert unwrap_picture_puts(src) == r"{\includegraphics{m}}{\begin{tabular}{c}1\end{tabular}}"


def test_tcolorboxes_defined_in_the_body_and_redefined_become_quotes():
    # tae2025tess2: each figure defines its boxes again; positions found once cut the definitions in the wrong place
    from tex2md.preprocess.boxes import tcolorbox_to_quote
    defs = (r"\newtcolorbox{cbox}[2][]{colback=white, sharp corners," "\n" r"  boxrule=0.5mm, title=#2, #1}" "\n"
            r"\newtcolorbox{cboxl}[2][]{sharp corners=west, title=#2, #1}" "\n")
    use = r"\begin{cbox}{Example}\begin{cboxl}[colframe=gray]{}User: hi\end{cboxl}\end{cbox}" "\n"
    out = tcolorbox_to_quote(defs + use + defs + use)
    assert "newtcolorbox" not in out and "sharp" not in out and "boxrule" not in out
    assert out.count(r"\begin{quote}") == 4 and out.count(r"\textbf{Example}") == 2 and out.count("User: hi") == 2


def test_resizebox_around_math_inside_display():
    from tex2md.preprocess.environments import boxes_around_math
    tex = r"where \[\resizebox{\linewidth}{!}{$\displaystyle a = b,$}\] and \resizebox{2cm}{!}{$c$} text"
    assert boxes_around_math(tex) == r"where \[\displaystyle a = b,\] and $c$ text"
    # a class file's \fitbox is read as \resizebox
    assert boxes_around_math(r"\begin{equation}\fitbox{0.9\textwidth}{$x$}\end{equation}") == \
        r"\begin{equation}x\end{equation}"
