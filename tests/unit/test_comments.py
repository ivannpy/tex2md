import shutil
import subprocess

import pytest

from tex2md.preprocess.comments import empty_comments, plain_verbatim_environments

LISTINGS = "\n".join([
    r"Text 50\% done % a comment",
    r"\begin{verbatim}",
    "x = 10 % 3   # modulo",
    r"\end{verbatim}",
    r"\begin{lstlisting}[language=Python]",
    'print("%d" % n)',
    r"\end{lstlisting}",
    r"\verb|a % b| and \lstinline!c % d! end % trailing",
    r"See \url{http://x.org/a%20b} then % gone",
    r"\end{document}",
    "after the end",
])


def test_comments_are_emptied_but_not_inside_verbatim():
    out = empty_comments(LISTINGS)
    assert out.splitlines()[0] == r"Text 50\% done %"
    assert "x = 10 % 3   # modulo" in out and 'print("%d" % n)' in out
    assert r"\verb|a % b| and \lstinline!c % d! end %" in out
    assert "trailing" not in out and "gone" not in out


def test_the_percent_of_a_url_is_escaped_and_the_text_after_end_document_is_dropped():
    out = empty_comments(LISTINGS)
    assert r"\url{http://x.org/a\%20b}" in out
    assert "after the end" not in out and out.rstrip().endswith(r"\end{document}")


def test_environments_the_document_defines_are_verbatim_too():
    src = "\\DefineVerbatimEnvironment{Code}{Verbatim}{}\n\\begin{Code}\nif i % 2:\n\\end{Code}\nreal % comment\n"
    out = empty_comments(src)
    assert "if i % 2:" in out and out.endswith("real %\n")


@pytest.mark.skipif(shutil.which("latexpand") is None, reason="latexpand not installed")
def test_latexpand_keeps_what_empty_comments_would_otherwise_lose(tmp_path):
    (tmp_path / "a.tex").write_text("\\begin{document}\n\\begin{verbatim}\nif i % 2:\n\\end{verbatim}\n\\input{b}\n\\end{document}\n")
    (tmp_path / "b.tex").write_text("from b % comment\n")
    stripped = subprocess.run(["latexpand", "--empty-comments", "a.tex"], cwd=tmp_path, capture_output=True, text=True).stdout
    kept = subprocess.run(["latexpand", "--keep-comments", "--empty-comments", "a.tex"], cwd=tmp_path,
                          capture_output=True, text=True).stdout
    assert "if i %\n" in stripped                     # what latexpand alone does to a listing
    out = empty_comments(kept)
    assert "if i % 2:" in out and "from b %" in out and "comment" not in out


def test_environments_pandoc_would_read_as_latex_become_verbatim_or_listings():
    src = ("\\DefineVerbatimEnvironment{QS}{Verbatim}{fontsize=\\small}\n\\lstnewenvironment{python}[1][]{\\lstset{style=py,#1}}{}\n"
           "A\\begin{QS}[x=1]\na % b\n\\end{QS}\\begin{python}[caption=c]\nx\n\\end{python}\\begin{spverbatim}s\\end{spverbatim}")
    out = plain_verbatim_environments(src)
    assert "DefineVerbatimEnvironment" not in out and "lstnewenvironment" not in out
    assert "A\\begin{verbatim}\na % b\n\\end{verbatim}\\begin{lstlisting}\nx\n\\end{lstlisting}\\begin{verbatim}s\\end{verbatim}" in out


def test_code_pandoc_wrote_with_highlighting_commands_becomes_plain_text_again():
    src = ("\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\\\\{\\}}\n\\begin{Highlighting}[]\n"
           "\\KeywordTok{def}\\NormalTok{ f(x\\_1): }\\CommentTok{\\# a{-}b \\textless{}|im\\_start|\\textgreater{}}\n\\end{Highlighting}")
    out = plain_verbatim_environments(src)
    assert out.strip().splitlines()[1:3] == ["def f(x_1): # a-b <|im_start|>", "\\end{verbatim}"]
