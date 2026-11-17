import re

from tex2md.util.tex import (
    balanced_end,
    group,
    protected_spans,
    rewrite_outside_math,
    skip_optional,
    span_at,
    strip_comments,
    unwrap,
)


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


def test_protected_spans_cover_math_and_verbatim_but_not_text():
    text = r"a $x$ b \$5 $$y$$ \(z\) \verb|$q$| \begin{align}w\end{align} c $no close" + "\n\nnext $p$$q$ \\begin{verbatim}$v$\\end{verbatim}"
    spans = [text[a:b] for a, b in protected_spans(text)]
    assert spans == ["$x$", "$$y$$", r"\(z\)", r"\verb|$q$|", r"\begin{align}w\end{align}", "$p$", "$q$",
                     r"\begin{verbatim}$v$\end{verbatim}"]
    assert [text[a:b] for a, b in protected_spans(text, math=False)] == [r"\verb|$q$|", r"\begin{verbatim}$v$\end{verbatim}"]


def test_span_at_finds_the_span_containing_a_position():
    spans = [(2, 5), (10, 12)]
    assert span_at(spans, 3) == (2, 5) and span_at(spans, 10) == (10, 12)
    assert span_at(spans, 5) is None and span_at(spans, 0) is None and span_at(spans, 99) is None


def test_rewrite_outside_math_skips_math_and_runs_until_nothing_changes():
    pattern = re.compile(r"\{\{")
    out = rewrite_outside_math("a {{ b $c {{ d$ e {{", pattern, lambda m, t: ("{", m.end()))
    assert out == "a { b $c {{ d$ e {"
    # a rewritten construct exposes the one nested in it: the second pass reaches it
    nested = re.compile(r"\\u\{([^{}]*)\}")
    assert rewrite_outside_math(r"\u{\u{a}}", nested, lambda m, t: (m.group(1), m.end())) == "a"


def test_skip_optional_steps_over_bracketed_arguments():
    text = "[a][b {x]} c] {body}"
    assert text[skip_optional(text, 0)] == "{"
