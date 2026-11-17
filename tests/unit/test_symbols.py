from tex2md.preprocess.symbols import ding, symbols_to_unicode


def test_pifont_codes_become_their_unicode_character():
    assert ding(51) == "\u2713" and ding(55) == "\u2717" and ding(52) == "\u2714" and ding(56) == "\u2718"
    assert ding(172) == "\u2460" and ding(182) == "\u2776" and ding(192) == "\u2780" and ding(202) == "\u278a"
    assert ding(72) == "\u2605" and ding(170) == "\u2665"
    assert ding(1) == ""


def test_check_marks_and_text_symbols_are_written_out_in_text():
    src = r"A & \ding{51} & \checkmark{} & \Checkmark{} & \XSolidBrush{} & 12\textendash 15 \dag{} done\textdaggerdbl{}"
    assert symbols_to_unicode(src) == "A & \u2713 & \u2713 & \u2713 & \u2717 & 12\u201315 \u2020 done\u2021"


def test_a_control_word_eats_the_blanks_after_it_as_tex_does():
    assert symbols_to_unicode(r"yes \checkmark   and \checkmark{} and") == "yes \u2713and \u2713 and"


def test_math_is_left_alone_where_the_command_is_meaningful():
    src = r"$x \checkmark y$ and \[ a \dag b \] and \begin{equation} \checkmark \end{equation} but \checkmark here"
    assert symbols_to_unicode(src) == "$x \\checkmark y$ and \\[ a \\dag b \\] and \\begin{equation} \\checkmark \\end{equation} but \u2713here"


def test_ding_is_replaced_everywhere_because_it_is_text_only():
    assert symbols_to_unicode(r"$\ding{51}$ \ding{55}") == "$\u2713$ \u2717"


def test_cmark_xmark_follow_the_pifont_convention_unless_the_document_defines_them():
    assert symbols_to_unicode(r"a \cmark b \xmark") == "a \u2713b \u2717"
    own = r"\newcommand{\cmark}{yes} a \cmark b"
    assert symbols_to_unicode(own) == own


def test_a_user_definition_of_a_symbol_command_is_respected():
    own = r"\renewcommand{\dag}{X} \dag"
    assert symbols_to_unicode(own) == own
