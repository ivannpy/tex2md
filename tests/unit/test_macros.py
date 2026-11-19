from tex2md.preprocess.macros import (
    drop_primitive_spacing,
    expand_conditional_macros,
    expand_delimited_macros,
    expand_paired_delimiters,
    local_style_macros,
    text_in_macro_bodies,
)


def test_text_in_macro_bodies_becomes_textrm():
    src = r"\def\method{{\text{UDLM}}} \newcommand{\ours}{\text{Ours}} \method and \ours{} and \text{kept} here"
    out = text_in_macro_bodies(src)
    assert "\\def\\method{{\\textrm{\u2063UDLM}}}" in out and "\\newcommand{\\ours}{\\textrm{\u2063Ours}}" in out
    assert r"and \text{kept} here" in out


def test_macro_without_text_is_untouched():
    src = r"\newcommand{\R}{\mathbb{R}}"
    assert text_in_macro_bodies(src) == src


def test_macro_used_only_in_math_keeps_text():
    src = r"\newcommand{\cond}{\text{if}} and $x \cond y$"
    assert text_in_macro_bodies(src) == src


def test_macro_used_in_running_text_gets_textrm():
    src = r"\newcommand{\method}{\text{UDLM}} We propose \method{} here."
    assert "\\newcommand{\\method}{\\textrm{\u2063UDLM}}" in text_in_macro_bodies(src)


def test_paired_delimiters_are_expanded_in_all_three_forms():
    src = (r"\DeclarePairedDelimiter\parentheses{(}{)} \DeclarePairedDelimiter{\brackets}{[}{]} "
           r"$\parentheses{a}+\parentheses*{b}+\brackets[\big]{c}$")
    out = expand_paired_delimiters(src)
    assert "DeclarePairedDelimiter" not in out
    assert r"$(a)+\left( b \right)+\bigl[ c \bigr]$" in out


def test_delimited_macro_is_expanded_and_its_definition_dropped():
    src = r"\def\bb#1\ee{\begin{align*}#1\end{align*}} text \bb a &= b \ee more"
    out = expand_delimited_macros(src)
    assert "def" not in out and r"\begin{align*}a &= b \end{align*}" in out


def test_paired_delimiters_do_not_glue_a_control_word_to_the_next_letter():
    out = expand_paired_delimiters(r"\DeclarePairedDelimiter\norm{\lVert}{\rVert} $\norm{x}y+\norm*{y}+\norm[\big]{z}+\norm{\alpha}$")
    assert out.strip() == r"$\lVert x\rVert y+\left\lVert y \right\rVert+\bigl\lVert z \bigr\rVert+\lVert\alpha\rVert$"


def test_paired_delimiter_with_arguments_and_a_body_is_expanded_in_its_three_sizes():
    src = ("\\DeclarePairedDelimiterX{\\klxy}[2]{(}{)}{%\n  #1\\;\\delimsize\\|\\;#2%\n}\n"
           "\\begin{document} $\\klxy{q}{p} + \\klxy*{a}{b} + \\klxy[\\Big]{c}{d}$")
    out = expand_paired_delimiters(src)
    assert "DeclarePairedDelimiterX" not in out
    assert r"$(q\;\|\;p) + \left( a\;\middle\|\;b \right) + \Bigl( c\;\Big\|\;d \Bigr)$" in out


def test_a_delimiter_named_inside_another_macro_stays_defined():
    src = ("\\DeclarePairedDelimiterX{\\klxy}[2]{(}{)}{#1\\;\\delimsize\\|\\;#2}\n\\newcommand{\\kl}{\\operatorname{KL}\\klxy}\n"
           "\\begin{document} $\\kl{q}{p}$")
    out = expand_paired_delimiters(src)
    assert r"\newcommand{\klxy}[2]{(#1\;\|\;#2)}" in out and r"\newcommand{\kl}{\operatorname{KL}\klxy}" in out


def test_macros_that_decide_with_xstring_are_expanded_where_pandoc_cannot():
    src = (r"\newcommand{\deltaval}[1]{\textrm{\scriptsize \IfBeginWith{#1}{-}{\textcolor{red}{(#1)}}{\textcolor{green}{(#1)}}}}"
           r"\newcommand{\valdelta}[2]{#1{\deltaval{#2}}}"
           r"\begin{document} A & \valdelta{78.1}{-0.2} & \valdelta{60.0}{+5.3} \\ \IfEq{a}{a}{same}{diff}")
    out = expand_conditional_macros(src)
    assert "newcommand" not in out and "IfBeginWith" not in out
    assert r"78.1{{\textrm{\scriptsize \textcolor{red}{(-0.2)}}}}" in out and r"\textcolor{green}{(+5.3)}" in out and "same" in out


def test_a_style_file_that_ends_in_a_backslash_does_not_stop_the_scan(tmp_path):
    (tmp_path / "notation.sty").write_text("\\newcommand{\\R}{\\mathbb{R}}\n\\newcommand{\\bad}[1]\\")
    flat = "\\usepackage{notation}\n\\begin{document} $\\R$"
    assert "\\newcommand{\\R}{\\mathbb{R}}" in local_style_macros(flat, str(tmp_path))


def test_struts_are_dropped():
    # yamabe2025priming: \colorbox{orange!20}{\strut orange} in a caption made pandoc fail on the whole table
    assert drop_primitive_spacing(r"a \colorbox{o}{\strut x} \textbf{\strut z} \bigstrut") == r"a \colorbox{o}{x} \textbf{z} \bigstrut"


def test_struts_in_listings_are_kept():
    # gat2024dfm: |\colorbox{y}{\strut{}source}| escapes inside a listing
    src = "\\begin{lstlisting}\ndef bfs(|\\colorbox{y}{\\strut{}source}|)\n\\end{lstlisting}"
    assert drop_primitive_spacing(src) == src


def test_control_space_before_a_dollar_becomes_a_space():
    # chen2024dforcing: "$x=\ $Forward$(a)$" came out as "$x=\$Forward$(a)$" (an odd "$")
    assert drop_primitive_spacing(r"$x=\ $Forward$(a)$ and vs.\ $y$") == r"$x= $Forward$(a)$ and vs.\ $y$"
    assert drop_primitive_spacing(r"\newcommand{\noexis}{$\ $} a & $\ $ \\") == r"\newcommand{\noexis}{} a &  \\"
