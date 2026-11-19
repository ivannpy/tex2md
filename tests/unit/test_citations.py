from tex2md.preprocess.citations import first_authors, parse_bbl, plain_bib, resolve_cites, simplify_citeproc

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


def test_citeproc_becomes_its_own_citation_text():
    src = r"Kimi \citeproc{ref-team2025kimi}{{[}12{]}} and \citeproc{ref-a}{{[}3, 4{]}} win"
    assert simplify_citeproc(src) == "Kimi {[}12{]} and {[}3, 4{]} win"


def test_text_without_citeproc_is_untouched():
    src = r"as shown \cite{ref-a} here"
    assert simplify_citeproc(src) == src


def test_citeproc_definition_is_not_touched():
    src = r"\NewDocumentCommand\citeproc{mm}{\begingroup\def\citeproctext{#2}\cite{#1}\endgroup} x \citeproc{k}{{[}1{]}}"
    out = simplify_citeproc(src)
    assert out.startswith(r"\NewDocumentCommand\citeproc{mm}{\begingroup") and out.endswith("x {[}1{]}")


def test_accents_and_special_letters_of_a_bibliography_become_unicode():
    from tex2md.preprocess.citations import plain_bib, tex_accents

    def bare(s):          # plain_bib removes the grouping braces that are left
        return tex_accents(s).replace("{", "").replace("}", "")
    assert bare(r"{\L}ukasz S{\o}ren Stra{\ss}e \AA ngstr\"om Jos\'{e} Mu\~noz {\AE}on") == "Łukasz Søren Straße Ångström José Muñoz Æon"
    assert bare(r"Mart{\'\i}~Mons Kydl{\'\i}{\v{c}}ek Ta{\"{\i}}ga Kavukcuo{\u{g}}lu \c Celebi Erd\H{o}s") == \
        "Martí~Mons Kydlíček Taïga Kavukcuoğlu Çelebi Erdős"
    assert plain_bib(r"{\L}ukasz Kaiser, 34:\penalty0 17981--17993, 2021. 12\textendash 15 50\% A\&B") == \
        "Łukasz Kaiser, 34:17981--17993, 2021. 12–15 50\\% A&B"


def test_aaai_and_old_acl_labels_give_the_author_and_year():
    bbl = (r"\bibitem[\protect\citeauthoryear{Anil \bgroup et al\mbox.\egroup }{2018}]{anil} A."
           r"\bibitem[\protect\citeauthoryear{Bucilua, Caruana, and Niculescu-Mizil}{Bucilua et~al.}{2006}]{buc} B."
           r"\bibitem[\protect\citename{Bahdanau \bgroup et al.\egroup }2015]{bah} C.")
    assert [(b["short"], b["year"]) for b in parse_bbl(bbl)] == \
        [("Anil et al.", "2018"), ("Bucilua et~al.", "2006"), ("Bahdanau et al.", "2015")]


def test_other_packages_citation_commands_are_resolved():
    stats = {"unresolved_refs": 0, "unresolved_cites": 0}
    author_year = parse_bbl(r"\bibitem[Ho et~al.(2020)]{ho} Ho. \bibitem[Song and Ermon(2019)]{song} Song.")
    assert resolve_cites(r"\citeyearpar{ho} \shortcite{ho,song} \citeA{song} \citeNP[p.~3]{ho}", author_year, stats) == \
        "(2020) (2020, 2019) Song and Ermon (2019) Ho et al., 2020, p.~3"
    assert resolve_cites(r"\footcite{ho} \citet{ho}", author_year, stats) == r"\footnote{Ho et al., 2020.} Ho et al. (2020)"
    numbered = parse_bbl(r"\bibitem{ho} Ho. \bibitem{song} Song.")
    assert resolve_cites(r"\citeyearpar{ho} \citenum{song} \cite{ho,song}", numbered, stats) == "[1] 2 [1, 2]"
    assert stats["unresolved_cites"] == 0


BBL_ACM = r"""
\begin{thebibliography}{2}
\bibitem[Arriola et~al\mbox{.}(2025a)]%
        {arriola2025block}
\bibfield{author}{\bibinfo{person}{Marianne Arriola}, {and} \bibinfo{person}{Volodymyr Kuleshov}.} \bibinfo{year}{2025}\natexlab{a}.
\newblock \showarticletitle{Block diffusion}.
\newblock
\urldef\tempurl%
\url{https://arxiv.org/abs/2503.09573}
\showURL{%
\tempurl}
\end{thebibliography}
"""


def test_acm_reference_format_key_on_the_line_after_the_label():
    (entry,) = parse_bbl(BBL_ACM)
    assert entry["key"] == "arriola2025block"
    assert entry["short"] == "Arriola et~al\\mbox{.}" and entry["year"] == "2025a"
    assert plain_bib(entry["body"]) == "Marianne Arriola, and Volodymyr Kuleshov. 2025a. Block diffusion. " \
                                       "https://arxiv.org/abs/2503.09573"
