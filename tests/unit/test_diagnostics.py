from tex2md.diagnostics import (
    Diagnostic,
    counts,
    from_pandoc_stderr,
    main_file_diagnostics,
    scan_lost_text,
    scan_markdown,
    scan_missing_sections,
)


def codes(body):
    return {d.code: d for d in scan_markdown(body)}


def test_clean_markdown_has_no_diagnostics():
    assert scan_markdown("Plain text with $x_i \\in \\mathbb{R}$ and $$a = b \\tag{1}$$.") == []


def test_unknown_math_macros_are_counted_by_name():
    d = codes(r"We use $\vz + \vz$ and $$\vx \in \mathcal{X} \tag{1}$$ plus $\textsc{Foo}$.")["unknown-math-macro"]
    assert d.severity == "warning" and d.count == 4
    assert "\\vz x2" in d.examples and "\\textsc x1" in d.examples


def test_known_katex_commands_are_not_flagged():
    body = r"$\alpha \gets \frac{1}{2}\nonumber \triangleq \operatorname{KL}\left(p\middle\|q\right) \mathbb{E}$"
    assert "unknown-math-macro" not in codes(body)


def test_unresolved_reference_is_an_error_but_question_marks_in_data_are_not():
    d = codes("see Equation (??) above")["unresolved-reference"]
    assert d.severity == "error" and d.count == 1
    assert "unresolved-reference" not in codes("they said ????? loudly and $x=\\text{??}$")


def test_equation_environment_without_tag_is_reported_as_info():
    body = "$$\\begin{equation}\na=b\n\\end{equation}$$\n\n$$\\begin{equation}\na=c \\tag{2}\n\\end{equation}$$"
    d = codes(body)["equation-without-tag"]
    assert d.severity == "info" and d.count == 1


def test_html_residue_unresolved_citations_and_unnumbered_floats():
    body = "<table>\n<tr><td>x</td></tr></table>\n\nas in [@key2020].\n\n**Figura.** Panel\n"
    c = codes(body)
    assert c["html-residue"].count == 2 or c["html-residue"].count == 1
    assert c["unresolved-citation"].examples == ["[@key2020]"]
    assert c["unnumbered-float"].severity == "info"


def test_code_blocks_are_ignored():
    assert scan_markdown("```\n$\\vz$ and ?? and <div>\n```") == []


def test_pandoc_warnings_are_grouped_and_counted():
    err = ("[WARNING] Could not convert TeX math '\\foo', rendering as TeX\n"
           "[WARNING] Could not convert TeX math '\\bar', rendering as TeX\n"
           "[WARNING] Duplicate identifier 'x' at line 3\n")
    ds = from_pandoc_stderr(err)
    assert [d.code for d in ds] == ["pandoc-warning", "pandoc-warning"]
    assert ds[0].count == 2 and "Could not convert TeX math" in ds[0].message


def test_counts_by_severity():
    ds = [Diagnostic("a", "error", "m"), Diagnostic("b", "info", "m"), Diagnostic("c", "info", "m")]
    assert counts(ds) == {"error": 1, "warning": 0, "info": 2}


def test_a_paragraph_with_an_odd_number_of_dollars_is_an_error():
    d = codes("fine $a$ and $b$.\n\nbroken $x = 1 for $y$ here.\n\nescaped \\$5 and \\$6 are fine.")["unbalanced-math"]
    assert d.severity == "error" and d.count == 1 and d.examples[0].startswith("broken $x = 1")


def test_unbalanced_dollars_in_code_and_html_tables_are_not_reported():
    body = "```\ncost $5\n```\n\n    indented $5 code\n\n<table><tr><td>$x</td></tr></table>\n\nok $a$"
    assert "unbalanced-math" not in codes(body)


def test_mathtools_and_amssymb_commands_katex_knows_are_not_flagged():
    body = r"$a \coloneq b \Coloneqq c \lessapprox d \preccurlyeq e \rightsquigarrow f \vcenter{x}$"
    assert "unknown-math-macro" not in codes(body)


def test_text_of_the_latex_that_is_not_in_the_markdown_is_reported():
    from tex2md.diagnostics import scan_lost_text
    tex = ("\\begin{document}\\maketitle\\begin{abstract}one two three four five six seven eight nine ten eleven twelve "
           "thirteen fourteen fifteen sixteen\\end{abstract} \\noindent{hidden alpha beta gamma delta epsilon zeta eta theta iota "
           "kappa lambda mu nu xi omicron pi rho}\\end{document}")
    body = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    (d,) = scan_lost_text(tex, body)
    assert d.code == "lost-text" and d.count >= 20 and "hidden alpha beta" in d.examples[0]
    assert scan_lost_text(tex, body + " hidden alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho") == []


def test_lost_text_ignores_math_the_branches_the_document_does_not_take_and_definitions():
    from tex2md.diagnostics import scan_lost_text
    tex = ("\\newif\\ifchecklist \\checklistfalse \\begin{document}\\maketitle One two three four five six seven eight nine ten "
           "eleven twelve thirteen fourteen. $a b c d e f g h i j k l m n o p q r s t u v w x y z$ \\iffalse never shown text "
           "that is long enough to be a run of words in the source file \\fi \\ifchecklist checklist words that are hidden "
           "from the reader because the switch is off in this document\\fi \\newcommand{\\foo}{a long definition body with "
           "many many words that is not text of the paper at all}\\end{document}")
    body = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen."
    assert scan_lost_text(tex, body) == []


def test_text_hidden_by_a_macro_defined_as_nothing_is_not_lost_text():
    from tex2md.diagnostics import scan_lost_text
    tex = ("\\newcommand{\\eat}[1]{}\\begin{document}\\maketitle One two three four five six seven eight nine ten eleven twelve "
           "thirteen fourteen. \\eat{this long passage was cut from the paper by the authors but is still in the source file "
           "and is longer than twelve words}\\end{document}")
    assert scan_lost_text(tex, "One two three four five six seven eight nine ten eleven twelve thirteen fourteen.") == []


def test_a_caption_without_number_is_found_inside_a_list_item_too():
    out = {d.code: d for d in scan_markdown("1.  item\n\n    **Figura.** Panel (a).\n\n> **Tabla.** Quoted.\n\n**Figura.** Plain.\n")}
    assert out["unnumbered-float"].count == 3


def test_configuration_commands_with_deep_braces_are_not_lost_text():
    tex = ("\\begin{document}\n\\lstdefinelanguage{Py}{\n  language=Python,\n  literate=*\n"
           + "".join(f"    {{word{i}}}{{{{\\color{{codefunc}}word{i}\\_t}}}}{{1}}\n" for i in range(8))
           + "}\nThe results are presented in the next section and they agree with the theory.\n\\end{document}")
    body = "The results are presented in the next section and they agree with the theory."
    assert not [d for d in scan_lost_text(tex, body) if d.code == "lost-text"]


def test_missing_sections_flags_a_body_cut_in_half():
    sections = [{"level": 1, "num": str(n), "text": ""} for n in range(1, 7)]
    body = "# 1 Introduction\n\ntext\n\n# 2 Method\n\n## 2.1 Detail\n\n# 3 Results\n"
    (d,) = scan_missing_sections(sections, body)
    assert d.code == "missing-sections" and d.severity == "error" and d.examples == ["4", "5", "6"]


def test_missing_sections_is_quiet_when_every_number_has_a_heading():
    sections = [{"level": 1, "num": "1", "text": ""}, {"level": 2, "num": "II-A", "text": ""},
                {"level": 3, "num": "1.1.1", "text": ""}]
    assert scan_missing_sections(sections, "# 1 Intro\n\n## II-A Diffusion\n") == []


def test_an_ambiguous_main_file_is_reported():
    assert main_file_diagnostics("a.tex", ["a.tex"]) == []
    (d,) = main_file_diagnostics("a.tex", ["a.tex", "rebuttal.tex"])
    assert d.code == "ambiguous-main-file" and d.severity == "warning"


def test_html_residue_ignores_escaped_tags_and_code():
    body = "Wrap it:\\<table\\>\\</table\\>.\n\n    export_to<table> - 12\n\nAnd `<span>` too.\n"
    assert "html-residue" not in {d.code for d in scan_markdown(body)}
