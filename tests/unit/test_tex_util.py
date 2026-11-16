from tex2md.util.tex import balanced_end, group, strip_comments, unwrap


def test_balanced_end_skips_nested_groups():
    text = r"\caption{a {b} c} rest"
    start = text.index("{")
    assert text[start:balanced_end(text, start)] == "{a {b} c}"


def test_balanced_end_ignores_escaped_braces():
    text = r"{a \} still} tail"
    assert text[:balanced_end(text, 0)] == r"{a \} still}"


def test_group_returns_content_and_next_index():
    content, nxt = group("{abc}def", 0)
    assert content == "abc" and "{abc}def"[nxt:] == "def"


def test_strip_comments_keeps_escaped_percent():
    assert strip_comments("50\\% done % comment\nnext") == "50\\% done \nnext"


def test_unwrap_keeps_content_and_drops_layout_arguments():
    out = unwrap(r"\resizebox{\textwidth}{!}{\begin{tabular}{c}a\end{tabular}}", "resizebox", 2)
    assert out == r"\begin{tabular}{c}a\end{tabular}"
