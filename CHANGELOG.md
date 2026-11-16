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

## Before the refactor (single `tex2md.py`)
- Float numbering from the `.aux` caption lines; recovery of captions lost in `subtable`, `minipage`,
  `parbox`, `captionof`, `SCfigure`, image-only tables and text-only tables.
- `\newcite` author citations, `table*`, `adjustbox`/`resizebox` around loose captions.
