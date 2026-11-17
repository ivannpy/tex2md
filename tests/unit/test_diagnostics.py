from tex2md.diagnostics import Diagnostic, counts, from_pandoc_stderr, scan_markdown


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
