import gzip
import io
import os
import shutil
import tarfile
import urllib.error

import pytest

from tex2md.sources import arxiv
from tex2md.util import io as util_io


def tarball(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_the_source_is_unpacked_in_one_piece(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv, "http_get", lambda url: tarball({"main.tex": "\\documentclass{article}", "a/b.tex": "x"}))
    assert arxiv.fetch_source("2001.00001", str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["src"]                      # no ".partial" left behind
    assert (tmp_path / "src" / "a" / "b.tex").read_text() == "x"


def test_an_interrupted_unpacking_is_not_taken_for_the_whole_source(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv, "http_get", lambda url: tarball({"main.tex": "x"}))

    def boom(self, path, **kw):
        os.makedirs(os.path.join(path, "half"))
        raise OSError("disk full")
    monkeypatch.setattr(tarfile.TarFile, "extractall", boom)
    with pytest.raises(OSError):
        arxiv.fetch_source("2001.00001", str(tmp_path))
    assert not (tmp_path / "src").exists()        # the next run downloads again instead of using half a tree


def test_a_single_gzipped_tex_file_and_a_pdf_only_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv, "http_get", lambda url: gzip.compress(b"\\documentclass{article}"))
    assert arxiv.fetch_source("2001.00001", str(tmp_path / "one"))
    assert (tmp_path / "one" / "src" / "main.tex").read_bytes() == b"\\documentclass{article}"
    monkeypatch.setattr(arxiv, "http_get", lambda url: b"%PDF-1.5 ...")
    assert arxiv.fetch_source("2001.00002", str(tmp_path / "two")) is False
    assert not (tmp_path / "two" / "src").exists()


def test_a_not_found_is_not_retried(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(util_io.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(util_io.time, "sleep", lambda s: pytest.fail("slept before giving up on a 404"))
    with pytest.raises(urllib.error.HTTPError):
        util_io.http_get("https://example.org/x")
    assert len(calls) == 1


def test_a_server_error_is_retried(monkeypatch):
    calls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", {}, None)
        return Reply()
    monkeypatch.setattr(util_io.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(util_io.time, "sleep", lambda s: None)
    assert util_io.http_get("https://example.org/x") == b"ok" and len(calls) == 3


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_jats_text_with_cdata_keeps_its_less_than_and_ampersands(tmp_path):
    from tex2md.sources.jats import jats_body
    xml = tmp_path / "a.xml"
    xml.write_text("<article><body><sec><title>Intro</title><p><![CDATA[We need p < 0.05 & q > 2 here]]></p></sec></body></article>")
    body = jats_body(str(xml))
    assert "p < 0.05 & q > 2" in body.replace("\\<", "<").replace("\\>", ">").replace("\\&", "&")


def test_the_equation_labeller_keeps_the_bytes_of_a_latin1_source(tmp_path):
    from tex2md.convert import Workspace, label_equations
    build = tmp_path / "build"
    build.mkdir()
    raw = b"caf\xe9 \\begin{equation}x\\end{equation}"
    (build / "a.tex").write_bytes(raw)
    label_equations(Workspace(work=str(tmp_path), src=str(tmp_path / "src"), build=str(build)))
    assert (build / "a.tex").read_bytes() == b"caf\xe9 \\begin{equation}x\\label{tex2md-eq-1}\\end{equation}"


def test_a_rate_limited_answer_waits_as_long_as_the_server_asks(monkeypatch):
    waits = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"
    calls = []

    def urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "42"}, None)
        return Reply()
    monkeypatch.setattr(util_io.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(util_io.time, "sleep", waits.append)
    assert util_io.http_get("https://example.org/x") == b"ok" and waits == [42]


def test_the_command_line_says_what_went_wrong_instead_of_printing_a_traceback(monkeypatch):
    import argparse

    from tex2md.cli import main

    def rate_limited(a):
        raise urllib.error.HTTPError("https://export.arxiv.org/api/query", 429, "Too Many Requests", {}, None)

    class Parser:
        def parse_args(self, argv):
            return argparse.Namespace(fn=rate_limited)
    monkeypatch.setattr("tex2md.cli.build_parser", Parser)
    with pytest.raises(SystemExit) as e:
        main(["anything"])
    assert "HTTP 429" in str(e.value) and "rate limiting" in str(e.value)


def test_api_requests_keep_the_gap_arxiv_asks_for_even_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv, "CACHE", str(tmp_path))
    (tmp_path / "meta").mkdir()
    now = [1000.0]
    waits = []
    monkeypatch.setattr(arxiv.time, "time", lambda: now[0])
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: (waits.append(s), now.__setitem__(0, now[0] + s)))
    arxiv._pace_api()                       # the first request of all: no wait
    now[0] += 1.0
    arxiv._pace_api()                       # one second later: waits the other two
    now[0] += 10.0
    arxiv._pace_api()                       # long after: no wait
    assert waits == [2.0]


def test_dollars_around_plain_text_leave_the_title():
    from tex2md.sources.arxiv import plain_title
    assert plain_title("The Diffusion Duality, Chapter II: $Ψ$-Samplers") == "The Diffusion Duality, Chapter II: Ψ-Samplers"
    assert plain_title("Scaling $\\alpha$ and $x^2$ laws") == "Scaling $\\alpha$ and $x^2$ laws"


def test_tidy_marker_repairs_what_marker_gets_wrong():
    from tex2md.sources.pdf import tidy_marker
    body = ("<sup>&</sup>lt;sup>1</sup>Nyquist, H.\n\n# A B B A B A B A\n\n## I. INTRODUCTION\n\ntext\n\n"
            "#### II. METHOD\n\n# A. Analysis\n\n#### B. Results\n\n# • Setting 4: separate LRs.\n\n# REFERENCES\n")
    out = tidy_marker(body)
    assert out.startswith("<sup>1</sup>Nyquist, H.")
    assert [ln for ln in out.split("\n") if ln.startswith("#")] == \
        ["# I. INTRODUCTION", "# II. METHOD", "## A. Analysis", "## B. Results", "# REFERENCES"]
    assert "\nA B B A B A B A\n" in out and "\n• Setting 4: separate LRs.\n" in out
    assert tidy_marker(out) == out


def test_tidy_marker_md_drops_the_jstor_cover_and_keeps_the_header():
    from tex2md.sources.pdf import tidy_marker_md
    md = ('---\ntitle: "Maximum Likelihood"\n---\n\n# Maximum Likelihood\n\n**Autores:** A\n\n'
          "> Sin DOI.\n> Convertido desde el PDF con marker (OCR).\n\n"
          "A. P. Dempster\n\n### Stable URL:\n\nhttp://links.jstor.org/x\n\n"
          "JSTOR is an independent not-for-profit organization.\n\n# Maximum Likelihood\n\nBy A. P. DEMPSTER\n")
    out = tidy_marker_md(md)
    assert "JSTOR" not in out and out.count("# Maximum Likelihood") == 1 and "**Autores:** A" in out
    assert out.endswith("> Convertido desde el PDF con marker (OCR).\n\nBy A. P. DEMPSTER\n")
    assert tidy_marker_md(out) == out


def test_tidy_marker_writes_the_html_of_table_rows_as_markdown():
    from tex2md.sources.pdf import tidy_marker
    out = tidy_marker("| 10<sup>9</sup> | <b>Ours</b> | $a$<sup>2</sup> | <u>u</u><br/>z |\n\nText <b>kept</b>.\n")
    assert out == "| 10⁹ | **Ours** | $a^{2}$ | u<br>z |\n\nText <b>kept</b>.\n"
