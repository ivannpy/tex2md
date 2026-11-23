import shutil
import subprocess

import pytest

from tex2md.convert import latexmk_command


def test_latexmk_command_disables_shell_escape():
    cmd = latexmk_command("main")
    assert cmd[0] == "latexmk" and cmd[-1] == "main.tex"
    assert any(a.startswith("-pdflatex=") and "-no-shell-escape" in a for a in cmd)


@pytest.mark.skipif(shutil.which("latexmk") is None or shutil.which("pdflatex") is None, reason="TeX not installed")
def test_untrusted_source_cannot_run_commands(tmp_path):
    (tmp_path / "s.tex").write_text(
        "\\documentclass{article}\\begin{document}\\immediate\\write18{kpsewhich -var-value=TEXMFHOME}hi\\end{document}")
    subprocess.run(latexmk_command("s"), cwd=tmp_path, capture_output=True, timeout=300)
    log = (tmp_path / "s.log").read_text(errors="ignore")
    assert "disabled" in log and "executed safely" not in log


def test_drop_package_from_a_list():
    from tex2md.convert import drop_package
    assert drop_package(r"\usepackage[x]{a,tocstyle,b}", "tocstyle") == r"\usepackage[x]{a,b}"
    assert drop_package(r"\usepackage{tocstyle}", "tocstyle") == r"% tex2md: \usepackage{tocstyle}"
    assert drop_package(r"\usepackage{tocstyles}", "tocstyle") == r"\usepackage{tocstyles}"


def test_inputs_from_root(tmp_path):
    from tex2md.convert import inputs_from_root
    (tmp_path / "latex").mkdir()
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "intro.tex").write_text("x")
    (tmp_path / "acl.sty").write_text("")
    main = tmp_path / "latex" / "main.tex"
    main.write_text("\\usepackage{acl}\\input{sections/intro}")
    assert inputs_from_root(str(main), str(tmp_path))
    main.write_text("\\input{local}")
    (tmp_path / "latex" / "local.tex").write_text("")
    assert not inputs_from_root(str(main), str(tmp_path))
