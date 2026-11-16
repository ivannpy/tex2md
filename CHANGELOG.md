# Changelog

## 0.1.0 (refactor)
- Package `tex2md` under `src/`; one CLI (`tex2md arxiv|jats|pdf|frontmatter|add-mn|check|index|fixmath|doctor`).
  `tex2md.py`, `papers.py` and `doc2md.py` remain as thin wrappers with the old arguments.
- `convert` split into stages (fetch, compile, flatten, preprocess, pandoc, postprocess, write).
- Preprocessing is an ordered list of 36 named rules (`preprocess/pipeline.py`).
- Lua filter split into modules (`filters/lib/`) with a single entry point (`filters/tex2md.lua`).
- Cache and model weights moved out of the repository (`TEX2MD_CACHE`, `TEX2MD_MODELS`).
- `tex2md doctor`, pytest suite (unit, Lua filter, CLI, golden), ruff, mypy, Makefile.
- Output identical to the pre-refactor version on the 124 LaTeX papers of the library.

## After the refactor: the last 8 missing captions
- `tex2md audit DIR` compares every converted .md with the numbers of its compiled PDF.
- New rules: `unwrap-twocolumn` (teaser figures), `dissolve-multi-caption-floats` (table + figure in one
  `wrapfigure`), `unnest-floats` (figures inside figures), `simplify-longtable-specs` (pandoc-written longtables:
  plain columns, repeated headers and `\endlastfoot` dropped), `simplify-citeproc` (inside tables), `quad-before-digits`
  (`\quad 16 steps` kept its number).
- Lua: tables turned into figures are numbered in the Table pass (document order); a valid `\label` outranks the
  loose substring match of captions that start with a macro.
- `retype-floats` no longer retypes a figure that has its own `\caption`.
- Library: 8 papers gained captions (baek2026, cheng2025, hoogeboom2021, lambert2025, schiff2025, wu2023, zeng2025,
  wang2025 recovered `1)`-style numbers); the rest of the 124 outputs are byte-identical.

## Before the refactor (single `tex2md.py`)
- Float numbering from the `.aux` caption lines; recovery of captions lost in `subtable`, `minipage`,
  `parbox`, `captionof`, `SCfigure`, image-only tables and text-only tables.
- `\newcite` author citations, `table*`, `adjustbox`/`resizebox` around loose captions.
