"""The ordered LaTeX preprocessing pipeline that feeds pandoc.

``RULES`` is the single place that fixes the order. Rules are plain ``text -> text`` functions
(wrapped with ``_pure``) or take the ``Context`` (labels, bibliography, counters) when they need it.
The order matters, so every rule says why it sits where it does; run one rule alone with
``apply_rule(name, text)`` when debugging.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from tex2md.preprocess.algorithms import rewrite_algorithm_captions, rewrite_algorithmic
from tex2md.preprocess.boxes import tcolorbox_to_quote
from tex2md.preprocess.citations import PREAMBLE, inject_bibliography
from tex2md.preprocess.floats import (
    captionof_to_caption,
    flatten_subfloats,
    retype_floats,
    split_captioned_blocks,
    starred_floats_to_plain,
    strip_float_layout,
    tables_without_tabular,
    unnest_tables,
    wrap_envs_to_plain,
    wrap_loose_captions,
)
from tex2md.preprocess.macros import (
    drop_digit_operators,
    drop_fragile_macro_defs,
    drop_primitive_spacing,
    drop_size_redefs,
    fix_stray_macro_braces,
)
from tex2md.preprocess.references import expand_wrappers, resolve_refs
from tex2md.preprocess.structure import (
    drop_title_hacks,
    expand_abbreviations,
    label_options,
    mark_appendix,
    strip_comment_lines,
)
from tex2md.preprocess.tables import (
    centered_column_types,
    drop_empty_math_cells,
    normalize_tabulars,
    plain_tabular_envs,
    simplify_multicolumn_specs,
    strip_layout_wrappers,
    unwrap_cell_macros,
)


@dataclass
class Context:
    """What the rules that resolve references need besides the text."""
    labels: dict
    bib: list | None
    stats: dict = field(default_factory=lambda: {"unresolved_refs": 0, "unresolved_cites": 0})


Rule = Callable[[str, Context], str]


def _pure(fn: Callable[[str], str]) -> Rule:
    return lambda text, ctx: fn(text)


def _bold_math(text: str) -> str:
    return re.sub(r"\\bm\b", r"\\boldsymbol", text)


def _bibliography(text: str, ctx: Context) -> str:
    return inject_bibliography(text, ctx.bib, ctx.stats) if ctx.bib else text


RULES: list[tuple[str, Rule]] = [
    # --- document structure
    ("mark-appendix", _pure(mark_appendix)),
    ("expand-abbreviations", _pure(expand_abbreviations)),
    ("fix-stray-macro-braces", _pure(fix_stray_macro_braces)),
    # --- floats: normalise so pandoc keeps every caption (order: wrap -> retype -> split -> flatten)
    ("wrap-envs-to-plain", _pure(wrap_envs_to_plain)),
    ("tables-without-tabular", _pure(tables_without_tabular)),
    ("retype-floats", _pure(retype_floats)),
    ("wrap-loose-captions", _pure(wrap_loose_captions)),
    ("split-captioned-blocks", _pure(split_captioned_blocks)),
    ("strip-float-layout", _pure(strip_float_layout)),
    ("flatten-subfloats", _pure(flatten_subfloats)),
    ("unnest-tables", _pure(unnest_tables)),
    # --- macros and preamble pandoc cannot expand
    ("strip-comment-lines", _pure(strip_comment_lines)),
    ("drop-title-hacks", _pure(drop_title_hacks)),
    ("drop-fragile-macro-defs", _pure(drop_fragile_macro_defs)),
    ("drop-primitive-spacing", _pure(drop_primitive_spacing)),
    ("drop-digit-operators", _pure(drop_digit_operators)),
    ("drop-size-redefs", _pure(drop_size_redefs)),
    ("label-options", _pure(label_options)),
    ("simplify-multicolumn-specs", _pure(simplify_multicolumn_specs)),
    ("tcolorbox-to-quote", _pure(tcolorbox_to_quote)),
    # --- references and citations (need the .aux / .bbl numbers)
    ("expand-ref-wrappers", _pure(expand_wrappers)),
    ("resolve-refs", lambda text, ctx: resolve_refs(text, ctx.labels, ctx.stats)),
    ("bibliography", _bibliography),
    # --- environments pandoc does not know -> plain counterparts
    ("wrap-envs-to-plain-again", _pure(wrap_envs_to_plain)),
    ("starred-floats-to-plain", _pure(starred_floats_to_plain)),
    ("strip-layout-wrappers", _pure(strip_layout_wrappers)),
    ("plain-tabular-envs", _pure(plain_tabular_envs)),
    ("normalize-tabulars", _pure(normalize_tabulars)),
    ("rewrite-algorithmic", _pure(rewrite_algorithmic)),
    ("rewrite-algorithm-captions", lambda text, ctx: rewrite_algorithm_captions(text, ctx.labels)),
    ("drop-empty-math-cells", _pure(drop_empty_math_cells)),
    ("centered-column-types", _pure(centered_column_types)),
    ("unwrap-cell-macros", _pure(unwrap_cell_macros)),
    ("captionof-to-caption", _pure(captionof_to_caption)),
    ("bold-math", _pure(_bold_math)),
    ("add-preamble", lambda text, ctx: PREAMBLE + text),
]

_BY_NAME = dict(RULES)


def apply_rule(name: str, text: str, ctx: Context | None = None) -> str:
    """Run a single rule by name (tests and debugging)."""
    return _BY_NAME[name](text, ctx or Context(labels={}, bib=None))


def preprocess(text, labels, bib, stats):
    """Run every rule in order. ``stats`` is filled in place (unresolved refs and cites)."""
    ctx = Context(labels=labels, bib=bib, stats=stats)
    for _name, rule in RULES:
        text = rule(text, ctx)
    return text
