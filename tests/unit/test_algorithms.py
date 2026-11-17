from tex2md.preprocess.algorithms import algorithm2e_to_lines, rewrite_algorithm2e


def test_algorithm2e_blocks_become_indented_lines():
    body = (r"\KwIn{Model $f$, latent $x$}" "\n" r"\For{$t=1$ \KwTo $T$}{" "\n"
            r"Compute $u_t = f(x_t)$\;" "\n" r"\If{$t>1$}{ $x \gets u_t$\; }\Else{ stop\; }" "\n}" "\n" r"\Return $x$\;")
    out = algorithm2e_to_lines(body)
    lines = [ln for ln in out.split("\n\n")]
    assert lines[0] == r"\textbf{Input:} Model $f$, latent $x$"
    assert lines[1] == r"\textbf{for} $t=1$ \textbf{to} $T$ \textbf{do}"
    assert lines[2] == r"\quad\ Compute $u_t = f(x_t)$"
    assert any("textbf{if}" in ln and ln.startswith(r"\quad") for ln in lines)
    assert lines[-1] == r"\textbf{return} $x$"


def test_rewrite_algorithm2e_keeps_caption_and_label():
    src = r"\begin{algorithm}[t]\caption{Sampling}\label{alg:s}\KwIn{$x$}\For{$i$}{step\;}\end{algorithm}"
    out = rewrite_algorithm2e(src)
    assert r"\caption{Sampling}\label{alg:s}" in out and r"\textbf{Input:} $x$" in out and "KwIn" not in out


def test_algorithmic_environments_are_not_touched_by_algorithm2e_rewrite():
    src = r"\begin{algorithm}\caption{A}\begin{algorithmic}\State $x$\end{algorithmic}\end{algorithm}"
    assert rewrite_algorithm2e(src) == src
