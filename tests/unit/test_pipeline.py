from tex2md.preprocess.pipeline import RULES, Context, apply_rule, preprocess


def test_rule_names_are_unique_and_documented_order_is_stable():
    names = [n for n, _ in RULES]
    assert len(names) == len(set(names))
    # these relative orders are load-bearing (see pipeline.py)
    pos = {n: i for i, n in enumerate(names)}
    assert pos["tables-without-tabular"] < pos["retype-floats"] < pos["wrap-loose-captions"] < pos["split-captioned-blocks"]
    assert pos["split-captioned-blocks"] < pos["strip-float-layout"] < pos["flatten-subfloats"]
    assert pos["resolve-refs"] < pos["bibliography"]
    assert pos["captionof-to-caption"] < pos["add-preamble"] == len(names) - 1


def test_apply_rule_runs_one_rule():
    assert "TEX2MDAPPENDIXMARKER" in apply_rule("mark-appendix", "\\begin{document}x\\appendix y")


def test_preprocess_fills_stats_in_place():
    stats = {"unresolved_refs": 0, "unresolved_cites": 0}
    out = preprocess(r"See \ref{nope}.", {}, None, stats)
    assert stats["unresolved_refs"] == 1
    assert out.endswith("See ??.") or "??" in out


def test_context_defaults():
    ctx = Context(labels={}, bib=None)
    assert ctx.stats == {"unresolved_refs": 0, "unresolved_cites": 0}
