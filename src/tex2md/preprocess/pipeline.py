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

from tex2md.preprocess.algorithms import rewrite_algorithm2e, rewrite_algorithm_captions, rewrite_algorithmic
from tex2md.preprocess.boxes import tcolorbox_to_quote
from tex2md.preprocess.citations import PREAMBLE, inject_bibliography, simplify_citeproc
from tex2md.preprocess.comments import plain_verbatim_environments
from tex2md.preprocess.environments import (
    acks_to_section,
    ctable_to_table,
    drop_picture_code,
    expand_glossary,
    restatable_to_theorem,
    tabular_in_math,
    tblr_to_tabular,
    unwrap_multicols,
    unwrap_picture_puts,
)
from tex2md.preprocess.floats import (
    captionof_to_caption,
    dissolve_multi_caption_floats,
    flatten_subfloats,
    floatrow_to_floats,
    retype_floats,
    single_paragraph_captions,
    split_captioned_blocks,
    split_multi_caption_floats,
    starred_floats_to_plain,
    strip_float_layout,
    tables_without_tabular,
    unnest_floats,
    unwrap_algorithm_floats,
    wrap_envs_to_plain,
    wrap_loose_captions,
)
from tex2md.preprocess.macros import (
    drop_digit_operators,
    drop_fragile_macro_defs,
    drop_newcolumntype,
    drop_primitive_spacing,
    drop_size_redefs,
    expand_conditional_macros,
    expand_delimited_macros,
    expand_paired_delimiters,
    fix_stray_macro_braces,
    text_in_macro_bodies,
)
from tex2md.preprocess.references import expand_wrappers, hyperref_to_text, resolve_refs
from tex2md.preprocess.structure import (
    drop_title_hacks,
    expand_abbreviations,
    label_options,
    mark_appendix,
    quad_before_digits,
    strip_comment_lines,
    texorpdfstring_tex,
    unwrap_twocolumn,
)
from tex2md.preprocess.symbols import symbols_to_unicode
from tex2md.preprocess.tables import (
    centered_column_types,
    cr_to_row_end,
    drop_empty_math_cells,
    normalize_tabulars,
    plain_tabular_envs,
    simplify_longtable_specs,
    simplify_multicolumn_specs,
    strip_layout_wrappers,
    tighten_span_arguments,
    unwrap_cell_macros,
)
from tex2md.preprocess.textloss import (
    checklist_answers,
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


@dataclass
class Context:
    """What the rules that resolve references need besides the text."""
    labels: dict
    bib: list | None
    stats: dict = field(default_factory=lambda: {"unresolved_refs": 0, "unresolved_cites": 0})
    floats: list = field(default_factory=list)     # the .aux caption lines (build.aux.parse_floats)


Rule = Callable[[str, Context], str]


def _pure(fn: Callable[[str], str]) -> Rule:
    return lambda text, ctx: fn(text)


def _bold_math(text: str) -> str:
    return re.sub(r"\\bm\b", r"\\boldsymbol", text)


def _bibliography(text: str, ctx: Context) -> str:
    return inject_bibliography(text, ctx.bib, ctx.stats) if ctx.bib else text


RULES: list[tuple[str, Rule]] = [
    # --- verbatim first: every rule below reads the text as LaTeX
    ("plain-verbatim-environments", _pure(plain_verbatim_environments)),
    # --- document structure
    ("resolve-conditionals", _pure(resolve_conditionals)),  # before the branches are read by any other rule
    ("expand-conditional-macros", _pure(expand_conditional_macros)),
    ("delimited-macros", _pure(expand_delimited_macros)),
    ("paired-delimiters", _pure(expand_paired_delimiters)),
    ("symbols-to-unicode", _pure(symbols_to_unicode)),  # before text-in-macro-bodies (\cmark = \text{\ding{51}})
    ("text-in-macro-bodies", _pure(text_in_macro_bodies)),  # before anything expands or drops macros
    ("texorpdfstring", _pure(texorpdfstring_tex)),  # a one-argument call makes pandoc fail on the whole paper
    ("mark-appendix", _pure(mark_appendix)),
    ("expand-abbreviations", _pure(expand_abbreviations)),
    ("expand-glossary", _pure(expand_glossary)),
    ("quad-before-digits", _pure(quad_before_digits)),
    ("fix-stray-macro-braces", _pure(fix_stray_macro_braces)),
    ("unwrap-twocolumn", _pure(unwrap_twocolumn)),  # before the float rules: it holds the teaser figure
    ("tabular-in-math", _pure(tabular_in_math)),  # before any rule that looks for tabulars as table content
    ("tblr-to-tabular", _pure(tblr_to_tabular)),  # before the float rules look for a tabular in each table
    ("ctable-to-table", _pure(ctable_to_table)),  # (same)
    ("unwrap-picture-puts", _pure(unwrap_picture_puts)),  # an image and a table laid out with \put: a table to the float rules
    # --- floats: normalise so pandoc keeps every caption (order: wrap -> retype -> split -> flatten)
    ("floatrow-to-floats", _pure(floatrow_to_floats)),  # first: its boxes hold the captions the rules below look for
    ("single-paragraph-captions", _pure(single_paragraph_captions)),  # pandoc cannot read a caption with a paragraph break
    ("wrap-envs-to-plain", _pure(wrap_envs_to_plain)),
    ("unwrap-algorithm-floats", _pure(unwrap_algorithm_floats)),  # before a rule takes their captions for the float's
    ("tables-without-tabular", _pure(tables_without_tabular)),
    ("retype-floats", _pure(retype_floats)),
    ("dissolve-multi-caption-floats", _pure(dissolve_multi_caption_floats)),
    ("wrap-loose-captions", _pure(wrap_loose_captions)),
    ("split-captioned-blocks", _pure(split_captioned_blocks)),
    ("tables-without-tabular-after-split", _pure(tables_without_tabular)),     # (a block of a figure* can be a table with no tabular)
    ("strip-float-layout", _pure(strip_float_layout)),
    ("flatten-subfloats", lambda text, ctx: flatten_subfloats(text, ctx.floats)),
    ("subfloat-captions", _pure(subfloat_captions)),
    ("split-multi-caption-floats", _pure(split_multi_caption_floats)),  # what the flattened sub-floats leave
    ("unnest-floats", _pure(unnest_floats)),
    ("starred-captions-to-text", _pure(starred_captions_to_text)),  # after the floats have their final shape
    # --- macros and preamble pandoc cannot expand
    ("strip-comment-lines", _pure(strip_comment_lines)),
    ("drop-title-hacks", _pure(drop_title_hacks)),
    ("drop-fragile-macro-defs", _pure(drop_fragile_macro_defs)),
    ("drop-primitive-spacing", _pure(drop_primitive_spacing)),
    ("drop-digit-operators", _pure(drop_digit_operators)),
    ("drop-newcolumntype", _pure(drop_newcolumntype)),
    ("drop-size-redefs", _pure(drop_size_redefs)),
    ("label-options", _pure(label_options)),
    ("tighten-span-arguments", _pure(tighten_span_arguments)),
    ("simplify-multicolumn-specs", _pure(simplify_multicolumn_specs)),
    ("tcolorbox-to-quote", _pure(tcolorbox_to_quote)),
    ("restatable-to-theorem", _pure(restatable_to_theorem)),
    ("unwrap-multicols", _pure(unwrap_multicols)),
    ("acks-to-section", _pure(acks_to_section)),
    ("drop-picture-code", _pure(drop_picture_code)),
    # --- references and citations (need the .aux / .bbl numbers)
    ("simplify-citeproc", _pure(simplify_citeproc)),
    ("expand-ref-wrappers", _pure(expand_wrappers)),
    ("hyperref-to-text", _pure(hyperref_to_text)),  # a link to an anchor the Markdown does not have
    ("resolve-refs", lambda text, ctx: resolve_refs(text, ctx.labels, ctx.stats)),
    ("bibliography", _bibliography),
    # --- environments pandoc does not know -> plain counterparts
    ("wrap-envs-to-plain-again", _pure(wrap_envs_to_plain)),
    ("starred-floats-to-plain", _pure(starred_floats_to_plain)),
    ("strip-layout-wrappers", _pure(strip_layout_wrappers)),
    ("cr-to-row-end", _pure(cr_to_row_end)),
    ("plain-tabular-envs", _pure(plain_tabular_envs)),
    ("normalize-tabulars", _pure(normalize_tabulars)),
    ("simplify-longtable-specs", _pure(simplify_longtable_specs)),
    ("table-rules-and-colors", _pure(table_rules_and_colors)),
    ("rewrite-algorithmic", _pure(rewrite_algorithmic)),
    ("rewrite-algorithm2e", _pure(rewrite_algorithm2e)),
    ("rewrite-algorithm-captions", lambda text, ctx: rewrite_algorithm_captions(text, ctx.labels, ctx.floats)),
    ("drop-empty-math-cells", _pure(drop_empty_math_cells)),
    ("centered-column-types", _pure(centered_column_types)),
    # --- text pandoc would drop with the command in front of it (last: the rules above still write such commands)
    ("footnotetext-to-footnote", _pure(footnotetext_to_footnote)),
    ("drop-drawn-rules", _pure(drop_drawn_rules)),
    ("text-outside-math", _pure(text_outside_math)),
    ("unwrap-text-boxes", _pure(unwrap_text_boxes)),
    ("unwrap-cell-macros", _pure(unwrap_cell_macros)),
    ("checklist-answers", _pure(checklist_answers)),
    ("detach-swallowed-groups", _pure(detach_swallowed_groups)),
    ("captionof-to-caption", _pure(captionof_to_caption)),
    ("bold-math", _pure(_bold_math)),
    ("add-preamble", lambda text, ctx: PREAMBLE + text),
]

_BY_NAME = dict(RULES)


def apply_rule(name: str, text: str, ctx: Context | None = None) -> str:
    """Run a single rule by name (tests and debugging)."""
    return _BY_NAME[name](text, ctx or Context(labels={}, bib=None))


def preprocess(text, labels, bib, stats, floats=None):
    """Run every rule in order. ``stats`` is filled in place (unresolved refs and cites)."""
    ctx = Context(labels=labels, bib=bib, stats=stats, floats=floats or [])
    for _name, rule in RULES:
        text = rule(text, ctx)
    return text
