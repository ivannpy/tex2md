import json

from tex2md.build.latex import declared_toplevel, find_main, main_candidates

DOC = "\\documentclass{article}\n\\begin{document}\n%s\n\\end{document}\n"


def test_the_main_file_declared_in_00readme_json_wins(tmp_path):
    (tmp_path / "rebuttal.tex").write_text(DOC % "Thanks, reviewers.")
    (tmp_path / "sample-sigconf.tex").write_text(DOC % "The paper.")
    (tmp_path / "00README.json").write_text(json.dumps(
        {"sources": [{"usage": "ignore", "filename": "rebuttal.tex"},
                     {"usage": "toplevel", "filename": "sample-sigconf.tex"}]}))
    assert find_main(str(tmp_path)) == str(tmp_path / "sample-sigconf.tex")


def test_the_old_00readme_xxx_is_read_too(tmp_path):
    (tmp_path / "a.tex").write_text(DOC % "A")
    (tmp_path / "b.tex").write_text(DOC % "B")
    (tmp_path / "00README.XXX").write_text("b.tex toplevelfile\na.tex ignore\n")
    assert declared_toplevel(str(tmp_path)) == str(tmp_path / "b.tex")


def test_without_readme_the_file_with_its_bbl_and_inputs_beats_main_tex(tmp_path):
    (tmp_path / "main.tex").write_text(DOC % "Notes to myself.")
    (tmp_path / "tex").mkdir()
    (tmp_path / "tex" / "intro.tex").write_text("\\section{Intro}")
    (tmp_path / "iclr2023_conference.tex").write_text(DOC % "\\input{tex/intro.tex}")
    (tmp_path / "iclr2023_conference.bbl").write_text("")
    assert find_main(str(tmp_path)) == str(tmp_path / "iclr2023_conference.tex")


def test_subfiles_and_standalone_parts_are_not_candidates(tmp_path):
    (tmp_path / "paper.tex").write_text(DOC % "\\subfile{sec.tex}")
    (tmp_path / "sec.tex").write_text("\\documentclass[paper.tex]{subfiles}\n\\begin{document}\nx\n\\end{document}")
    (tmp_path / "fig.tex").write_text("\\documentclass[tikz]{standalone}\n\\begin{document}\nx\n\\end{document}")
    assert main_candidates(str(tmp_path)) == [str(tmp_path / "paper.tex")]
