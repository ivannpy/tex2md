from tex2md.preprocess.references import resolve_refs

LABELS = {"fig:a": {"num": "3a", "type": "subfigure"}, "sub@fig:a": {"num": "a", "type": ""},
          "fig:b": {"num": "4b", "type": "subfigure"}, "eq:1": {"num": "1", "type": "equation"},
          "eq:3": {"num": "3", "type": "equation"}, "sec:x": {"num": "2.1", "type": "subsection"}}


def run(text):
    stats = {"unresolved_refs": 0, "unresolved_cites": 0}
    return resolve_refs(text, LABELS, stats), stats["unresolved_refs"]


def test_subref_gives_the_letter_of_the_subfigure():
    assert run(r"See \subref{fig:a} and (\subref{fig:b})") == ("See (a) and (b)", 0)


def test_cleveref_range_labelcref_and_name_commands():
    assert run(r"\crefrange{eq:1}{eq:3}") == ("Eqs.~(1) to (3)", 0)
    assert run(r"\labelcref{eq:1,eq:3}") == ("(1) and (3)", 0)
    assert run(r"this \namecref{sec:x}, \nameCref{sec:x}") == ("this section, Section", 0)
    assert run(r"\Cref{sec:x} \cref{eq:1,eq:3}") == ("Section~2.1 Eqs.~(1) and (3)", 0)


def test_an_unknown_label_counts_and_prints_question_marks():
    assert run(r"\ref{nope} and \eqref{eq:1}") == ("?? and (1)", 1)


def test_a_key_written_with_a_text_macro_is_found_as_latex_wrote_it_to_the_aux():
    labels = {"alg:OeMDM{}-train": {"num": "1", "type": "algorithm"},
              "eq:LoMDM{}_loss": {"num": "3", "type": "equation"}}
    stats = {"unresolved_refs": 0}
    text = "\\def\\framework{OeMDM}\n\\def\\method{LoMDM}\nAlgorithm~\\ref{alg:\\framework{}-train} and \\eqref{eq:\\method{}_loss}"
    out = resolve_refs(text, labels, stats)
    assert out.endswith("Algorithm~1 and (3)") and stats["unresolved_refs"] == 0
