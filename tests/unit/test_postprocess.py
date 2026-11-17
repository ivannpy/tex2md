from tex2md.postprocess.markdown import drop_counter_hacks, mathml_to_tex, restore_math_text, tidy_math


def test_restore_math_text_puts_back_text_in_math_and_removes_the_mark():
    m = "\u2063"
    body = r"A \textrm{x} and $a_{\textrm{" + m + r"in}}$ and $$b=\textrm{" + m + r"out}$$ and " + m + "UDLM"
    assert restore_math_text(body) == r"A \textrm{x} and $a_{\text{in}}$ and $$b=\text{out}$$ and UDLM"


def test_mathml_in_html_tables_goes_back_to_tex():
    html = ('<td>0.1<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>&#8804;</mo></mrow>'
            '<annotation encoding="application/x-tex">\\leq x_{&lt;t}</annotation></semantics></math></td>')
    assert mathml_to_tex(html) == r"<td>0.1$\leq x_{<t}$</td>"


def test_counter_hacks_are_removed_from_equations():
    assert drop_counter_hacks(r"x=1 \addtocounter{equation}{1}\tag{\theequation}\tag{15}") == r"x=1 \tag{15}"


def test_tidy_math_drops_tex_comments_and_merges_adjacent_formulas():
    assert tidy_math("with $a \\coloneqq %\n b$ and 90% of $x$") == "with $a \\coloneqq \n b$ and 90% of $x$"
    assert tidy_math("Dist-$1$$\\uparrow$ ok") == "Dist-$1 \\uparrow$ ok"
    assert tidy_math("$$\na = b % note\n$$") == "$$\na = b \n$$"
    assert tidy_math("cost is \\$5 and 50\\%") == "cost is \\$5 and 50\\%"
