from tex2md.preprocess.equations import label_numbered_equations


def labels(text):
    out, _ = label_numbered_equations(text)
    return out


def test_unlabelled_equation_gets_a_label():
    assert labels(r"\begin{equation}x=1\end{equation}") == r"\begin{equation}x=1\label{tex2md-eq-1}\end{equation}"


def test_labelled_unnumbered_and_starred_equations_are_left_alone():
    for src in (r"\begin{equation}x=1\label{eq:a}\end{equation}",
                r"\begin{equation}x=1\nonumber\end{equation}",
                r"\begin{equation*}x=1\end{equation*}",
                r"\begin{align*}a&=b\\c&=d\end{align*}",
                r"\begin{equation}x=1\tag{A}\end{equation}"):
        assert labels(src) == src


def test_each_numbered_row_of_an_align_gets_its_own_label():
    out = labels(r"\begin{align}a&=b\\c&=d\nonumber\\e&=f\\g&=h\end{align}")
    assert out.count("tex2md-eq-") == 3
    assert r"a&=b\label{tex2md-eq-1}\\c&=d\nonumber\\e&=f\label{tex2md-eq-2}\\g&=h\label{tex2md-eq-3}" in out
    # rows that already have a label of their own keep it (no second label)
    assert labels(r"\begin{align}a&=b\label{x}\\c&=d\end{align}").count("tex2md-eq-") == 1


def test_rows_inside_nested_environments_are_not_split():
    src = r"\begin{align}A&=\begin{pmatrix}1&2\\3&4\end{pmatrix}\\B&=C\end{align}"
    out = labels(src)
    assert out.count("tex2md-eq-") == 2 and r"\end{pmatrix}\label{tex2md-eq-1}\\B" in out


def test_trailing_empty_row_is_not_labelled_and_counter_continues():
    out, nxt = label_numbered_equations(r"\begin{gather}a=b\\c=d\\\end{gather}\begin{equation}z\end{equation}", start=5)
    assert out.count("tex2md-eq-") == 3 and nxt == 8


def test_alignat_argument_and_multline():
    out = labels(r"\begin{alignat}{2}a&=b&c&=d\end{alignat}\begin{multline}a+\\b\end{multline}")
    assert out.count("tex2md-eq-") == 2


def test_align_with_an_author_label_and_nonumber_is_left_alone():
    src = r"\begin{align}\label{eq:a}x&=1\nonumber\\y&=2\end{align}"
    assert labels(src) == src


def test_comments_are_not_read_as_equations_labels_or_row_ends():
    src = ("%\\begin{equation}\n\\begin{align}\n a &= b % old: c \\\\\n   &+ c \\\\\n d &= e % \\label{no}\n\\end{align}\n"
           "%\\end{equation}\n\\begin{equation} z % \\nonumber\n\\end{equation}")
    out, nxt = label_numbered_equations(src)
    assert out.count("\\label{tex2md-eq-") == 3 and nxt == 4
    assert "% old: c \\\\\n   &+ c \\label{tex2md-eq-1}\\\\" in out        # the "\\" in the comment ends no row
    assert "%\\begin{equation}" in out and "%\\end{equation}" in out


def test_latex_shown_in_a_listing_is_not_labelled():
    src = "\\begin{verbatim}\n\\begin{equation} shown \\end{equation}\n\\end{verbatim}\n\\begin{equation}x\\end{equation}"
    out, nxt = label_numbered_equations(src)
    assert out.count("\\label{tex2md-eq-") == 1 and "shown \\end{equation}" in out and nxt == 2
