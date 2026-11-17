from tex2md.preprocess.macros import expand_delimited_macros, expand_paired_delimiters, text_in_macro_bodies


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
