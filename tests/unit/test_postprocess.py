from tex2md.convert import postprocess
from tex2md.postprocess.fixmath import fix_body
from tex2md.postprocess.markdown import (
    drop_counter_hacks,
    isolate_display_math,
    mathml_to_tex,
    normalize_math_macros,
    protect_code,
    resolve_ifthenelse,
    restore_code,
    restore_math_text,
    tidy_math,
)


def test_restore_math_text_puts_back_text_in_math_and_removes_the_mark():
    m = "⁣"
    body = r"A \textrm{x} and $a_{\textrm{" + m + r"in}}$ and $$b=\textrm{" + m + r"out}$$ and " + m + "UDLM"
    assert restore_math_text(body) == r"A \textrm{x} and $a_{\text{in}}$ and $$b=\text{out}$$ and UDLM"


def test_mathml_in_html_tables_goes_back_to_tex():
    html = ('<td>0.1<math display="inline" xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>&#8804;</mo></mrow>'
            '<annotation encoding="application/x-tex">\\leq x_{&lt;t}</annotation></semantics></math></td>')
    assert mathml_to_tex(html) == r"<td>0.1$\leq x_{<t}$</td>"


def test_counter_hacks_are_removed_from_equations():
    assert drop_counter_hacks(r"x=1 \addtocounter{equation}{1}\tag{\theequation}\tag{15}") == r"x=1 \tag{15}"


def test_tidy_math_drops_tex_comments_and_merges_adjacent_formulas():
    assert tidy_math("Dist-$1$$\\uparrow$ ok") == "Dist-$1 \\uparrow$ ok"
    assert tidy_math("$$\na = b % note\n$$") == "$$\na = b \n$$"
    assert tidy_math("cost is \\$5 and 50\\%") == "cost is \\$5 and 50\\%"


def test_tidy_math_puts_an_inline_formula_on_one_line():
    # a line break inside $...$ splits a table row; a continuation that starts with "-" or "+" reads as a list
    assert tidy_math("with $a \\coloneqq %\n b$ and 90% of $x$") == "with $a \\coloneqq b$ and 90% of $x$"
    assert tidy_math("so $a\n- b$ holds") == "so $a - b$ holds"
    assert tidy_math("$$\na\n- b\n$$") == "$$\na\n- b\n$$"          # display math keeps its lines


def test_a_comment_at_the_end_of_a_formula_does_not_eat_the_closing_dollar():
    # pandoc writes "...}%$" when a macro body ends in a comment; stripping "%..." used to remove the "$" too
    assert tidy_math("is $%\n a = b%$ for $x$.") == "is $ a = b$ for $x$."
    assert tidy_math("$$ a = b%$$ and $c$") == "$$ a = b$$ and $c$"


def test_a_bar_inside_math_in_a_table_row_does_not_end_the_cell():
    row = r"| $p(x|y)$ and $\|z\|$ | other |"
    assert tidy_math(row) == r"| $p(x\vert y)$ and $\Vert z\Vert $ | other |"
    assert tidy_math(r"text $p(x|y)$ outside a table") == r"text $p(x|y)$ outside a table"


def test_display_math_gets_its_own_paragraph_except_where_blank_lines_would_break_things():
    assert isolate_display_math("so $$x=1$$ then") == "so \n\n$$x=1$$\n\nthen"
    for kept in ("| a | $$x$$ | b |", "- item $$x$$ more", "> quote $$x$$", "<td>$$x$$</td>", "  indented $$x$$"):
        assert isolate_display_math(kept) == kept


def test_code_is_not_touched_by_the_math_clean_ups():
    body = "text $a$\n\n    echo $x 100% $y\n\n```\nprint('$$')  % c\n```\n\nand `$5 % $6` inline $b % c\nd$ end"
    out = postprocess(body)
    assert "    echo $x 100% $y" in out and "print('$$')  % c" in out and "`$5 % $6`" in out
    assert "$b d$" in out


def test_code_inside_a_list_item_is_found_four_columns_further_in():
    body = "- item\n\n      code $5 % $6\n\n  continued $x$ and $$y$$ end\n"
    protected, stash = protect_code(body)
    assert stash == ["      code $5 % $6"]
    assert restore_code(protected, stash) == body


def test_a_row_of_tildes_in_an_equation_is_not_a_code_fence():
    body = "so $$\\begin{align}\na\n~~~~~~\nb\\end{align}$$ then $x$ % c\n"
    protected, stash = protect_code(body)
    assert stash == [] and protected == body


def test_fixmath_is_idempotent_and_leaves_code_alone():
    body = "A $a$ and $$b$$ here\n\n    code $x % $y\n\n| t | $p(x|y)$ |\n|---|---|\n"
    once = fix_body(body)
    assert fix_body(once) == once and "    code $x % $y" in once


def test_unknown_commands_with_a_known_cousin_are_rewritten():
    body = r"$\textsc{Softmax}(x)+\nicefrac{1}{2}+\nicefrac{a_1}{\sqrt{2}}+\scalebox{0.8}{y}+\ensuremath{z}+\mathparagraph$"
    assert normalize_math_macros(body) == r"$\text{Softmax}(x)+{1}/{2}+{a_1}/{\sqrt{2}}+y+z+\P$"


def test_a_picture_inside_a_formula_goes_away_with_its_box():
    assert normalize_math_macros(r"a ${\includegraphics[width=0.2\linewidth]{x/a.jpg}}$ b") == "a  b"
    assert normalize_math_macros(r"$\mathrm{L}\left(\vcenter{\hbox{\includegraphics[height=6ex]{a.png}}}\right)$") == \
        r"$\mathrm{L}\left(\right)$"


def test_renames_do_not_touch_longer_names():
    # \Bar -> \bar must not turn \Barx into \barx, nor \mathbbm into \mathbb with a stray letter
    assert normalize_math_macros(r"$\Bar{y}+\Barx+\mathbbm{1}+\mathbbmx$") == r"$\bar{y}+\Barx+\mathbb{1}+\mathbbmx$"
    assert normalize_math_macros(r"$\begin{cases*}a\end{cases*}$") == r"$\begin{cases}a\end{cases}$"


def test_ifthenelse_left_in_an_equation_by_a_macro_with_an_optional_part_is_decided():
    body = r"$L = \ifthenelse{\equal{\log p(x)}{}}{A}{B} + \ifthenelse{\equal{}{}}{C}{D} + \ifthenelse{\boolean{x}}{E}{F}$"
    assert resolve_ifthenelse(body) == r"$L = B + C + \ifthenelse{\boolean{x}}{E}{F}$"
