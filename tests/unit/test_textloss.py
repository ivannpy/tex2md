from tex2md.preprocess.textloss import (
    detach_swallowed_groups,
    drop_drawn_rules,
    footnotetext_to_footnote,
    resolve_conditionals,
    starred_captions_to_text,
    subfloat_captions,
    table_rules_and_colors,
    text_outside_math,
    unwrap_text_boxes,
)


def test_a_layout_command_in_front_of_a_group_is_removed_so_pandoc_keeps_the_group():
    assert detach_swallowed_groups(r"\noindent{\bf Proof.} \vspace{1em}{T} \color{red}{c} \quad {q}") == "{\\bf Proof.} {T} {c} ~{q}"


def test_detaching_leaves_math_definitions_and_separate_paragraphs_alone():
    src = "$\\quad {m}$ \\renewcommand\\centering{\\raggedright} {\\color{red} d} \\medskip\n\n{x}"
    assert detach_swallowed_groups(src) == src


def test_boxes_pandoc_does_not_know_give_back_their_content():
    src = r"\fbox{A\fbox{B}} \makebox[2cm][c]{C} \raisebox{2pt}[0pt]{D} \rotatebox[origin=c]{90}{E} \fcolorbox{a}{b}{G} \uwave{F}"
    assert unwrap_text_boxes(src) == r"{A{B}} {C} {D} {E} {G} \underline{F}"
    assert unwrap_text_boxes(r"$\smash{m}$ \renewcommand{\fbox}{x}") == r"$\smash{m}$ \renewcommand{\fbox}{x}"


def test_cmidrule_and_colours_do_not_leak_into_the_table():
    src = r"\cmidrule(lr){2-3}\cline{1-2} \rowcolor[HTML]{EEE}\multicolumn{2}{c}{H} & \cellcolor{gray!20}{58.6} \\ \addlinespace[2pt] \specialrule{.1em}{0pt}{0pt} \rule{0pt}{2ex}x"
    out = table_rules_and_colors(src)
    assert "cmidrule" not in out and "cline" not in out and "rowcolor" not in out and "cellcolor" not in out
    assert r"\multicolumn{2}{c}{H} & {58.6} \\" in out and "x" in out and "rule{0pt}" not in out


def test_footnote_marks_and_texts_are_paired_by_number_and_the_title_block_is_skipped():
    src = r"\begin{document}\maketitle A\footnotemark[1] B\footnotemark{} C. \footnotetext[1]{one} \footnotetext{two} \tablefootnote{tf}"
    assert footnotetext_to_footnote(src) == r"\begin{document}\maketitle A\footnote{one} B\footnote{two} C.   \footnote{tf}"
    title = r"\author{X\footnotemark[1]}\maketitle \footnotetext[1]{Equal contribution}"
    assert footnotetext_to_footnote(title) == r"\author{X\footnotemark[1]}\maketitle \footnote{Equal contribution}"


def test_subfloat_commands_become_the_body_and_an_a_b_c_caption_line():
    src = (r"\begin{figure}\subfloat[Cap A\label{a}]{\includegraphics{a}}\subfigure[Cap B]{\includegraphics{b}}"
           r"\subcaptionbox{Cap C}[.3\linewidth]{img}\subcaption{Cap D}\caption{Main}\end{figure}")
    out = subfloat_captions(src)
    for expected in (r"\par\textit{(a) Cap A\label{a}}\par", r"\par\textit{(b) Cap B}\par", r"\par\textit{(c) Cap C}\par",
                     r"\par\textit{(d) Cap D}\par", r"\includegraphics{a}", r"\caption{Main}"):
        assert expected in out
    assert "subfloat" not in out and "subfigure" not in out and "subcaptionbox" not in out


def test_starred_captions_under_the_panels_become_plain_lines_and_the_numbered_one_stays():
    src = (r"\begin{figure}\includegraphics{a}\caption*{(a) one}\includegraphics{b}\caption*{(b) two}\caption{Main}\end{figure}"
           r"\begin{table}x\caption*{only}\end{table}")
    out = starred_captions_to_text(src)
    assert r"\par\textit{(a) one}\par" in out and r"\par\textit{(b) two}\par" in out and r"\caption{Main}" in out
    assert r"\caption*{only}" in out        # a float whose only caption is starred keeps it


def test_conditionals_the_document_decides_are_resolved_and_the_rest_is_kept():
    src = (r"\newcommand{\anonymous}{0}\newboolean{draft}\setboolean{draft}{true}\newtoggle{t} "
           r"\ifthenelse{\equal{\anonymous}{0}}{YES}{NO} \ifthenelse{\boolean{draft}}{D}{F} \iftoggle{t}{T1}{T2} "
           r"\ifthenelse{\isodd{3}}{a}{b} \newcommand{\m}[1]{\ifthenelse{\equal{#1}{}}{e}{n}}")
    out = resolve_conditionals(src)
    assert " YES D T2 " in out
    assert r"\ifthenelse{\isodd{3}}{a}{b}" in out and r"\ifthenelse{\equal{#1}{}}{e}{n}" in out


def test_drawn_rules_go_but_a_separator_rule_stays():
    assert drop_drawn_rules(r"a ~{\color{red}\rule[.5ex]{1em}{.2em}} b $\rule{1pt}{2pt}$ \rule{\linewidth}{1pt}") == \
        r"a ~{\color{red}} b $\rule{1pt}{2pt}$ \rule{\linewidth}{1pt}"


def test_text_in_a_text_command_outside_math_is_kept_in_the_body_only():
    src = ("\\newcommand{\\KL}{\\text{KL}}\n\\begin{document}\n$x \\text{ if } y$ \\text{\\scriptsize // note} & \\text{N/A}\n"
           "\\def\\z{\\text{z}}")
    out = text_outside_math(src)
    assert "{\\scriptsize // note} & {N/A}" in out and "$x \\text{ if } y$" in out
    assert "\\newcommand{\\KL}{\\text{KL}}" in out and "\\def\\z{\\text{z}}" in out


def test_neurips_checklist_answers_print_their_label_and_justification():
    from tex2md.preprocess.textloss import checklist_answers
    src = (r"\begin{document} \item Did you cite? \answerYes{We cite the {\bf creators}.} \answerNA{} \answerNo{No new assets.}"
           "\n" r"\newcommand{\answerYes}[1]{\textcolor{blue}{[Yes] #1}}")
    out = checklist_answers(src)
    assert "[Yes] We cite the {\\bf creators}. [N/A] [No] No new assets." in " ".join(out.split())
    assert r"\newcommand{\answerYes}[1]" in out          # the definition is not a use


def test_a_toggle_without_else_takes_the_next_token_as_its_third_argument_as_tex_does():
    src = r"\newtoggle{ws} \iftoggle{ws}{\section{Analysis}}\subsection{Analysis: IO} text"
    assert resolve_conditionals(src) == r"\newtoggle{ws} \subsection{Analysis: IO} text"
    on = r"\newtoggle{ws}\toggletrue{ws} \iftoggle{ws}{\section{A}}\subsection{B}"
    assert resolve_conditionals(on) == r"\newtoggle{ws}\toggletrue{ws} \section{A}{B}"


def test_if_zero_blocks_are_switched_off():
    # golovneva2024reverse: "\if 0 ... Figure~\ref{fig:rev_algo} ... \fi" came out with a "??"
    assert resolve_conditionals("a \\if 0\nhidden\n\\fi b") == "a \\iffalse\nhidden\n\\fi b"
    assert resolve_conditionals(r"\ifx\a\b x\fi") == r"\ifx\a\b x\fi"
