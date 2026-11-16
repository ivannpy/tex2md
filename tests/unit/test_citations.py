from tex2md.preprocess.citations import first_authors, parse_bbl, resolve_cites

BBL_NUMERIC = r"""
\begin{thebibliography}{2}
\bibitem{sutskever2014}
Ilya Sutskever, Oriol Vinyals, and Quoc Le.
2014.
Sequence to sequence learning with neural networks.
In Proceedings of NIPS.

\bibitem{han2016}
Song Han and William J. Dally.
2016.
Deep compression.
\end{thebibliography}
"""

BBL_AUTHOR_YEAR = r"""
\begin{thebibliography}{1}
\bibitem[Ba and Caruana(2014)]{ba2014}
Lei Ba and Rich Caruana.
Do deep nets really need to be deep?
\end{thebibliography}
"""


def test_parse_bbl_numbers_entries_in_order():
    bib = parse_bbl(BBL_NUMERIC)
    assert [b["key"] for b in bib] == ["sutskever2014", "han2016"]
    assert [b["num"] for b in bib] == [1, 2]


def test_parse_bbl_reads_author_year_labels():
    (entry,) = parse_bbl(BBL_AUTHOR_YEAR)
    assert entry["short"] == "Ba and Caruana" and entry["year"] == "2014"


def test_parse_bbl_returns_none_for_biblatex():
    assert parse_bbl(r"\entry{a}{article}{}") is None


def test_first_authors_formats():
    assert first_authors("Ilya Sutskever, Oriol Vinyals, and Quoc Le. 2014. Title.") == "Sutskever et al."
    assert first_authors("Song Han and William J. Dally. 2016. Deep compression.") == "Han and Dally"
    assert first_authors("Yann LeCun. 1990. Optimal brain damage.") == "LeCun"
    assert first_authors("no year here") is None


def test_newcite_in_a_numbered_bibliography_keeps_the_author_name():
    bib = parse_bbl(BBL_NUMERIC)
    stats = {"unresolved_refs": 0, "unresolved_cites": 0}
    out = resolve_cites(r"\newcite{sutskever2014} used an LSTM and \cite{han2016} pruned it.", bib, stats)
    assert out.startswith("Sutskever et al. [1] used")
    assert "[2]" in out and stats["unresolved_cites"] == 0


def test_unknown_key_is_counted():
    stats = {"unresolved_refs": 0, "unresolved_cites": 0}
    resolve_cites(r"\cite{missing}", parse_bbl(BBL_NUMERIC), stats)
    assert stats["unresolved_cites"] == 1
