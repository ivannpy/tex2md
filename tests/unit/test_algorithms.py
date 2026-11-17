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


def test_algorithm_captions_take_the_number_of_the_compiled_document():
    from tex2md.preprocess.algorithms import rewrite_algorithm_captions
    labels = {"alg:a": {"num": "3.1", "type": "algorithm"}, "line:1": {"num": "1", "type": "ALC@unique"}}
    floats = [{"kind": "algorithm", "num": "A.1", "text": "samplingfromargmaxflows"},
              {"kind": "algorithm", "num": "A.2", "text": "optimizingargmaxflows"}]
    src = (r"\begin{algorithm}\caption{Sampling from Argmax Flows}\STATE x\label{line:1}\end{algorithm}"
           r"\begin{algorithm}\caption{Optimizing Argmax Flows}y\end{algorithm}"
           r"\begin{algorithm}\caption{Third\label{alg:a}} z\end{algorithm} \captionof{algorithm}{Loose one}"
           r"\begin{algorithm}\caption{Last}w\end{algorithm}")
    out = rewrite_algorithm_captions(src, labels, floats)
    for number, title in (("A.1", "Sampling from Argmax Flows"), ("A.2", "Optimizing Argmax Flows"), ("3.1", "Third"),
                          ("4", "Loose one"), ("5", "Last")):
        assert rf"\textbf{{Algoritmo {number}.}} {title}" in out


def test_algorithm2e_one_line_forms_and_eif():
    out = algorithm2e_to_lines(r"\eIf{$a>b$}{do A\;}{do B\;}\lIf{$c$}{quick\;}")
    assert out.split("\n\n") == [r"\textbf{if} $a>b$ \textbf{then}", r"\quad\ do A", r"\textbf{else}", r"\quad\ do B",
                                 r"\textbf{if} $c$ \textbf{then}", r"\quad\ quick"]


def test_a_thin_space_inside_a_formula_does_not_end_the_statement():
    body = (r"$C_0\gets \texttt{cat}(a);\;Y\gets\varnothing$\;" "\n"
            r"\For{$m \gets 1$ \KwTo $L$}{ $l \gets \arg\max\; H(x)$ \; \tcp{note $a\;b$} }")
    lines = algorithm2e_to_lines(body).split("\n\n")
    assert lines[0] == r"$C_0\gets \texttt{cat}(a);\;Y\gets\varnothing$"
    assert lines[2] == r"\quad\ $l \gets \arg\max\; H(x)$"
    assert r"// note $a\;b$" in lines[3]
