from tex2md.preprocess.structure import quad_before_digits, unwrap_twocolumn


def test_twocolumn_teaser_is_unwrapped():
    src = r"\twocolumn[{\centering \includegraphics{a}\captionof{figure}{Teaser}}]\maketitle"
    out = unwrap_twocolumn(src)
    assert out == r"\centering \includegraphics{a}\captionof{figure}{Teaser}\maketitle"


def test_twocolumn_without_inner_braces():
    assert unwrap_twocolumn(r"\twocolumn[\includegraphics{a}]rest") == r"\includegraphics{a}rest"


def test_text_without_twocolumn_is_untouched():
    assert unwrap_twocolumn(r"\begin{document}x") == r"\begin{document}x"


def test_twocolumn_with_only_title_and_authors_is_left_alone():
    src = r"\twocolumn[\icmltitle{T}\begin{icmlauthorlist}\icmlauthor{A}{x}\end{icmlauthorlist}]body"
    assert unwrap_twocolumn(src) == src


def test_quad_before_digits_is_replaced_in_text_but_not_in_math():
    assert quad_before_digits(r"\quad 16 steps & $a\quad 2$") == r"~16 steps & $a\quad 2$"
    assert quad_before_digits(r"\quad word") == r"\quad word"


def test_quad_is_kept_inside_math_environments():
    src = "\\begin{align}x \\quad 0 \\le y\\end{align} and \\[a\\qquad 1\\]"
    assert quad_before_digits(src) == src
