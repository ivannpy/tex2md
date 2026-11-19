from tex2md.build.aux import parse_aux, parse_floats, parse_sections, use_long_captions

MAIN_AUX = r"""
\relax
\@input{ch1.aux}
\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}
\newlabel{eq:loss}{{2}{3}{Loss}{equation.2}{}}
\@writefile{lof}{\contentsline {figure}{\numberline {1}{\ignorespaces \textbf  {Overview} of the model.}}{2}{figure.caption.1}\protect \ifthenelse {\boolean {@tempswa}}{}{}}
\@writefile{lot}{\contentsline {table}{\numberline {1}{\ignorespaces Results on $x$.}}{4}{table.caption.2}\protect \ifthenelse {\boolean {@tempswa}}{}{}}
"""
CH1_AUX = r"""
\@writefile{lof}{\contentsline {figure}{\numberline {2}{\ignorespaces Appendix plot.}}{9}{figure.caption.3}}
"""


def write(tmp_path):
    (tmp_path / "main.aux").write_text(MAIN_AUX)
    (tmp_path / "ch1.aux").write_text(CH1_AUX)


def test_parse_aux_reads_section_and_equation_labels(tmp_path):
    write(tmp_path)
    labels = parse_aux(str(tmp_path))
    assert labels["sec:intro"]["num"] == "1" and labels["sec:intro"]["type"] == "section"
    assert labels["eq:loss"]["num"] == "2"


def test_parse_floats_follows_input_order_and_keeps_numbers(tmp_path):
    write(tmp_path)
    floats = parse_floats(str(tmp_path))
    # \@input{ch1.aux} comes first in main.aux, so the included chapter's float is first
    assert [(f["kind"], f["num"]) for f in floats] == [("figure", "2"), ("figure", "1"), ("table", "1")]
    assert floats[0]["text"].startswith("appendixplot") and floats[1]["text"].startswith("overview")


def test_parse_floats_is_empty_without_caption_lines(tmp_path):
    (tmp_path / "main.aux").write_text(r"\relax \newlabel{a}{{1}{1}}")
    assert parse_floats(str(tmp_path)) == []


def test_algorithm_captions_are_read_from_the_list_of_algorithms(tmp_path):
    (tmp_path / "main.aux").write_text(
        "\\@writefile{loa}{\\contentsline {algorithm}{\\numberline {1}{\\ignorespaces Sampling from \\textbf  {Argmax} Flows}}{4}{algorithm.1}}\n"
        "\\@writefile{loa}{\\contentsline {algorithm}{\\numberline {A.2}{\\ignorespaces Optimizing}}{9}{algorithm.2}}\n"
        "\\@writefile{lof}{\\contentsline {figure}{\\numberline {1}{\\ignorespaces Overview.}}{2}{figure.caption.1}}\n")
    floats = parse_floats(str(tmp_path))
    assert [(f["kind"], f["num"]) for f in floats] == [("algorithm", "1"), ("algorithm", "A.2"), ("figure", "1")]
    assert floats[0]["text"].startswith("samplingfromargmaxflows")


def test_a_caption_written_with_a_size_command_keeps_its_own_text_in_the_aux(tmp_path):
    (tmp_path / "main.aux").write_text(
        "\\@writefile{lot}{\\contentsline {table}{\\numberline {2}{\\ignorespaces \\relax \\fontsize  {9}{10}\\selectfont  "
        "\\abovedisplayskip 6\\p@ plus1.5\\p@ minus4\\p@ \\abovedisplayshortskip \\z@ plus2\\p@ \\belowdisplayskip \\abovedisplayskip "
        "\\belowdisplayshortskip 3\\p@ plus2\\p@ minus2\\p@ {Unconditional CIFAR10 reverse process.}}}{6}{figure.caption.5}}\n"
        "\\@writefile{lot}{\\contentsline {table}{\\numberline {3}{\\ignorespaces \\@setfontsize \\small \\@ixpt {11}"
        "\\abovedisplayskip 8.5\\p@ plus3\\p@ minus4\\p@ {Second one.}}}{7}{table.3}}\n")
    first, second = (f["text"] for f in parse_floats(str(tmp_path)))
    assert first.startswith("unconditionalcifar10reverseprocess") and second.startswith("secondone")


def test_a_size_command_before_a_caption_without_braces_does_not_eat_the_caption(tmp_path):
    (tmp_path / "main.aux").write_text(
        "\\@writefile{lot}{\\contentsline {table}{\\numberline {5}{\\ignorespaces \\relax \\fontsize  {9}{10}\\selectfont  "
        "\\abovedisplayskip 6\\p@ plus1.5\\p@ minus4\\p@ \\abovedisplayshortskip \\z@ plus2\\p@ \\belowdisplayskip \\abovedisplayskip "
        "\\belowdisplayshortskip 3\\p@ plus2\\p@ minus2\\p@ Planning times for Redundancy-Aware vs. Unaware.}}{10}{table.caption.20}}\n")
    (floats,) = parse_floats(str(tmp_path))
    assert floats["text"].startswith("planningtimesforredundancyawarevsunaware")


def test_audit_counts_captions_that_sit_in_a_list_item_or_a_quote(tmp_path, monkeypatch):
    from tex2md.library import audit
    monkeypatch.setattr(audit, "CACHE", str(tmp_path / "cache"))
    build = tmp_path / "cache" / "2401.00001v1" / "build"
    build.mkdir(parents=True)
    (build / "main.aux").write_text(
        "\\@writefile{lof}{\\contentsline {figure}{\\numberline {1}{\\ignorespaces First figure.}}{1}{figure.caption.1}}\n"
        "\\@writefile{lof}{\\contentsline {figure}{\\numberline {2}{\\ignorespaces Second figure.}}{2}{figure.caption.2}}\n"
        "\\@writefile{lof}{\\contentsline {figure}{\\numberline {3}{\\ignorespaces Third figure.}}{3}{figure.caption.3}}\n")
    md = tmp_path / "p.md"
    md.write_text('---\nsource: "arXiv LaTeX (x)"\narxiv: "2401.00001"\narxiv_version: "v1"\n---\n\n'
                  "1.  Item\n\n    **Figura 1.** First figure.\n\n> **Figura 2.** Second figure.\n\n**Figura 3.** Third figure.\n")
    result = audit.audit_file(md)
    assert result is not None and result.missing == [] and result.unknown == []


TOC_AUX = r"""
\@writefile{toc}{\contentsline {section}{\numberline {1}Introduction}{1}{section.1}\protected@file@percent }
\@writefile{toc}{\contentsline {subsection}{\numberline {1.1}Definition of $\text {EB-C}$}{2}{subsection.1.1}\protected@file@percent }
\@writefile{toc}{\contentsline {section}{Acknowledgments}{9}{section*.4}\protected@file@percent }
\@writefile{toc}{\contentsline {section}{\numberline {A}More}{10}{appendix.A}\protected@file@percent }
"""


def test_parse_sections_reads_the_numbered_table_of_contents(tmp_path):
    (tmp_path / "paper.aux").write_text(TOC_AUX)
    (tmp_path / "other.aux").write_text(TOC_AUX.replace("Introduction", "Rebuttal"))
    assert parse_sections(str(tmp_path), "paper") == [
        {"level": 1, "num": "1", "text": "introduction"},
        {"level": 2, "num": "1.1", "text": "definitionofebc"},
        {"level": 1, "num": "A", "text": "more"},
    ]


def test_a_label_key_with_braces_is_read_whole(tmp_path):
    (tmp_path / "main.aux").write_text("\\newlabel{alg:OeMDM{}-train}{{1}{7}{Training}{algorithm.1}{}}\n")
    assert parse_aux(str(tmp_path))["alg:OeMDM{}-train"]["num"] == "1"


def test_parse_floats_keeps_only_the_caption_and_resolves_its_references(tmp_path):
    (tmp_path / "main.aux").write_text(
        r"\newlabel{fig:a}{{6}{3}{x}{figure.6}{}}" "\n"
        r"\@writefile{lot}{\contentsline {table}{\numberline {1}{\ignorespaces Caption}}{3}{table.caption.3}\protected@file@percent }" "\n"
        r"\@writefile{lof}{\contentsline {figure}{\numberline {7}{\ignorespaces We repeat \mbox {\hyperref [fig:a]{Figure~\ref *{fig:a}}} here}}{4}{figure.7}}")
    floats = parse_floats(str(tmp_path))
    assert [f["text"] for f in floats] == ["caption", "werepeatfigure6here"]


def test_long_captions_replace_a_shared_short_form():
    # \caption[Caption]{...} twice (besiroglu2024chinchilla): the .aux only has "Caption" for both tables
    floats = [{"kind": "table", "num": "1", "text": "caption"}, {"kind": "figure", "num": "1", "text": "plot"},
              {"kind": "table", "num": "3", "text": "caption"}]
    tex = (r"\caption[Caption]{Our estimates} % \caption[Caption]{commented out}" "\n"
           r"\caption{Plot} \caption[Caption]{\small Our estimates again}")
    out = use_long_captions(floats, tex)
    assert [f["text"] for f in out] == ["ourestimates", "plot", "ourestimatesagain"]
    assert floats[0]["text"] == "caption"           # the input list is not changed
