# Changelog

## HTML in tables (2026-11-18)
`audit` counted 32 `html-residue` in the library; none is left.

- **Cells with block content** (lists, a rule, a quote, several paragraphs) are flattened to one line: list items as
  `• item` / `1. item` separated by `<br>`, the line break of a pipe-table cell. pandoc wrote the whole table as HTML.
- **One-column tables whose cells hold more than a line** (a dataset card, a sample of generated text) become their
  blocks, with consecutive code-only cells joined into one code block (`ouyang2022instructgpt`: 16 tables,
  `hoffmann2022`, `wen2025dija`, generated samples in `sahoo2025duo`, `sahoo2025esolm`, `havasi2025editflows`).
- **A rule macro made of primitives** (`\newcommand{\thickhline}{\noalign{...}\hrule height 1pt ...}`) is removed
  and its uses become `\hline`: pandoc printed "height 1pt" in a cell (`hendrycks2021mmlu`).
- **marker tables**: `<b>`, `<i>`, `<u>`, `<sup>`, `<sub>` in table rows become Markdown, Unicode scripts or a
  formula (`tidy_marker`, also through `fixmarker`); `<br>` stays.
- **`html-residue`** no longer counts an escaped tag (`\<table\>`, text of the paper) or one inside code.

## Odd `$`, false `??`, titles and marker output (2026-11-18)
The rest of `HALLAZGOS_2026-11-18.md` §3 and §4.

- **`$x=\ $Forward$(a)$`**: a control space before the closing `$` (pandoc trimmed the math and left `\$`:
  `chen2024dforcing`) becomes a plain space; before an opening `$` (`vs.\ $x$`) it is kept. `$\ $` alone (an empty
  table cell, `li2022universal`) is dropped: pandoc printed `$$`, and that table had rows run together.
- **References**: a `\ref` key with commas is one label when the `.aux` has it (`brown2020gpt3`);
  `\if 0 ... \fi` is `\iffalse` (`golovneva2024reverse` showed the `??` of a switched-off block; `sun2023hollow` showed a paragraph the authors had
  switched off). The other `??`
  of the library are also in the PDF (ROADMAP F14, checked one by one).
- **Titles**: dollars around plain text leave arXiv's title (`$Ψ$-Samplers`); `arxiv --title` sets the title,
  and `tests/regress.py` passes the one of the header, so hand-corrected titles survive a regeneration.
- **`\part`**: a header above the level of `\section` (once a header matched the `.aux` toc) gets no number
  (`\part{Appendix}` came out as `# A Appendix`).
- **marker**: new `tidy_marker`, run on every PDF conversion and, through the new `fixmarker` command, on converted
  `.md`: footnote marks escaped twice (`<sup>&</sup>lt;sup>1</sup>`, 11 library files), the JSTOR cover page
  (`dempster1977em`), heading levels where sections are numbered I., II. with subsections A., B.
  (`bengio1994longterm`), and headings that are a list item, a row of single letters or a sentence.

## Missing captions (2026-11-18)
The 14 library papers whose `.md` lacked captions the `.aux` has (`HALLAZGOS_2026-11-18.md` §3): each had its own
cause. `audit --level error` now shows no float differences in the library. Five new rules (77).

- **`.aux` captions**: the text is the caption group only (a short caption had the page and anchor glued to it), with
  `\hyperref[..]` dropped and `\ref{x}` replaced by its number, as in the Markdown (`allenzhu2023`). A
  `\caption[short]{long}` takes the long form when the short one is shared (`\caption[Caption]{...}` twice:
  `besiroglu2024chinchilla`).
- **References**: wrappers of `\ref`/`\cite` with up to three arguments and through other wrappers are expanded
  (`\figref{x}` -> `\myref{Figure}{x}`); new rule `hyperref-to-text` keeps the text of `\hyperref[x]{...}` (a link to an
  anchor the Markdown does not have; with a `\ref` inside it came out as HTML: 70 links in `allenzhu2023`).
- **Floats pandoc lost**: `wrapstuff` (`bansal2026ttcd`), `teaserfigure` (`chen2026bard`), a caption-less
  `wrapfigure` inside another float, `\twocolumn[\begin{@twocolumnfalse}...]` with a `]` inside (`chang2022maskgit`),
  `floatrow` boxes (`kingma2021vdm`, new rule `floatrow-to-floats`), `ctable` and its `\FL`/`\ML`/`\NN`/`\LL` and a
  `picture` laid out with `\put` (`chen2024dforcing`, new rules `ctable-to-table` and `unwrap-picture-puts`).
- **Several captions in one float**: a captioned block between or under the minipages (`wang2026computegap`), a
  minipage holding two captioned tables (`lu2025adablock`), and two captions left once the sub-figures are flattened
  (`diffusiongemma2026`, new rule `split-multi-caption-floats`).
- **Tables pandoc dropped with their caption**: a column spec shorter than the usual row (`deschenaux2026duo2`:
  `llllc` for 12 cells) grows to fit it (the most frequent row width, not the widest: rows written by a macro, an
  `aligned` in a cell, author-year `&` and `\tabularnewline` were checked against the library); `\strut` is dropped (inside a caption it made pandoc fail on the table:
  `yamabe2025priming`; the CSL bibliography of `lambert2025reinforcemen` now has each number on its entry's line).
- **Sub-figures**: every caption of a panel becomes a sub-caption line (`xie2025dreamcoder` lost its panel (c)).
- **tcolorbox**: boxes defined again inside each figure were cut in the wrong place (positions found once, text
  changed before them: `tae2025tess2`).
- **`audit`**: floats kept out of the `.aux` by an empty short caption (`\caption[]{...}`) are no longer reported
  (`pan2025demasking`, `vonrutte2026scaling`: ROADMAP F7).

## Library reconversion fixes (2026-11-18)
From `HALLAZGOS_2026-11-18.md` (158 library papers reconverted: 4 pandoc failures, 4 conversions that lost content
with the report in `ok`) and from a new diagnostic that found 8 more.

- **Main file**: the `00README.json` (`"usage": "toplevel"`) or `00README.XXX` (`toplevelfile`) of the e-print names
  it; without one, a `.bbl` of the same name and the number of files pulled in beat the name `main.tex`. `subfiles`
  parts and `standalone` figures are not candidates. `chen2026bard` had converted the rebuttal and `sun2023hollow` a
  draft of notes; `2607.14106` also changes (its README names `neurips_2026.tex`).
- **`\subfile{x}`** is expanded (without the preamble of `x`): `he2021exposure` had lost half its sections.
- **New diagnostics**: `missing-sections` (error: numbered sections of the `.aux` table of contents with no heading of
  that number in the Markdown) and `ambiguous-main-file` (warning). The report also records `main`.
- **Section numbers from the table of contents**: the filter gives each header the number of the `.aux` toc entry
  with the same title (in order, keeping the level offset of the first match), before the label and the level
  counters. Headings were unnumbered or misnumbered under a `\part` (`feng2025theory`), with lost labels, custom
  appendix commands (`\beginappendix`: `cheng2025sdar` now has A instead of 10) and IEEE numbers.
- **ACM-Reference-Format `.bbl`**: `\bibitem[label]%` with the key on the next line, `\urldef`/`\showURL`
  (`kuleshov2026modern`: 91 unresolved citations; `kwon2023vllm` and `muennighoff2023` now use their `.bbl`).
- **pandoc failures**: `\texorpdfstring` with one argument (new rule `texorpdfstring`); a hand-made verbatim
  environment built with `\catcode` (`\def\zzz{...}` that makes `\` an ordinary character); a `{` of the preamble never closed
  (`\usepackage{listings}{`, reported as `unbalanced-braces`); environments that write their body to
  `\jobname.vrb` and read it back (`\tcbverbatimwrite`) become `verbatim`.
- **Figures** keep a code block inside a quote (sample texts in a `tcolorbox`), and the math clean-ups leave code
  inside block quotes alone (`$` in those samples gave `unbalanced-math`).
- **References**: a key written with a text macro (`\ref{alg:\framework{}-train}`) is looked up as LaTeX wrote it to
  the `.aux` (`alg:OeMDM{}-train`); the key may hold braces, also in `parse_aux`, which cut it at the first `}`
  (`hong2026oemdm`: 6 `??`).
- **Equation labels** are no longer put inside a macro definition (`\newcommand{\ee}{\end{equation}}`): every
  equation got a second label and amsmath dropped the author's (`kaplan2020`: 32 `??`). Cached builds made with
  the old labeller must be deleted to be recompiled (`~/.cache/tex2md/<id>/build`).
- **A figure captioned with `\captionof{table}`** (a plot beside a table in one `figure` environment) is printed as
  `Tabla N` when its caption is, word for word, that of a table in the `.aux` and of no figure (`wang2026computegap`
  printed its Table 2 as "Figura 2").

## Audit pass (2026-11-17): the library and 65 arXiv papers that were not in it
Method: reconvert the 124 library papers into a scratch folder (the library's `.md` files were not touched), convert
65 more arXiv sources (2012–2025, classics and recent), and look for what pandoc does without saying: text that
disappears, formulas whose `$` do not pair up, numbers of the PDF that are not in the Markdown, captions the `.aux`
numbers and the `.md` does not. Result: 0 float differences against the compiled PDFs in the 65 new papers and one
(known, false) in the library; test suite from 138 to 271 tests.

- **Packaging bug**: `.gitignore` had `build/` and `dist/`, which also hid the package `src/tex2md/build/` (`latex.py`,
  `aux.py`): it was never committed, so a fresh clone failed with `ModuleNotFoundError: tex2md.build`. The patterns are
  anchored (`/build/`, `/dist/`); the new package files still have to be added to git.
- **New diagnostics**: `unbalanced-math` (error: a paragraph whose `$` do not pair up), `lost-text` (warning: runs of
  12+ words of the LaTeX that are nowhere in the Markdown; hidden branches, drawing code, configuration commands and
  author blocks are left out), `unbalanced-braces` (a repaired brace), `dropped-definition`; the `.md` scan also
  finds captions inside list items.
- **Recovery**: besides environments, pandoc's recovery now drops the definition that holds the error
  (`\newcolumntype{..}[1]{#1}`), every definition of a macro that expands into itself ("Loop encountered in expanding
  macro"), and a preamble line it cannot read (`\global\setlength`). An empty pandoc output and PDF wrappers
  (`\includepdf`) give a clear error; plain TeX and LaTeX 2.09 sources are reported as unsupported.
- **Text pandoc used to lose without a warning** (new modules `preprocess/textloss.py`, `comments.py`,
  `environments.py`, `symbols.py`, `braces.py`; rules went from 45 to 71): unknown commands (from a `.sty`)
  swallow all their brace arguments; boxes (`\fbox`, `\framebox`, `\centerline`, `\raisebox`, `\rotatebox`…);
  `\cellcolor{c}{x}` and `\rowcolor`; `\cmidrule` ranges; `\footnotetext`; `\ifthenelse`/`\iftoggle`; NeurIPS
  checklist answers (`\answerYes`); `\acks{}` and `ack`; `\ding`, `\checkmark`, `\cmark`/`\xmark`; listings (`\lstinputlisting`
  and files `\input` that latexpand leaves, `%` inside listings, `DefineVerbatimEnvironment`); `forest`/inline `\tikz`
  code printed as text; `multicols`; `restatable`; `tblr`; glossaries (`\gls`); a `tabular` used as a piecewise
  definition inside a formula; a stray `{` that swallowed the rest of a paper.
- **Floats and numbering**: captions with paragraph breaks, lists or tabulars; several `\caption` in one float;
  sub-captions that the document numbers as figures; `\captionsetup{type=table}` in a figure (also inside the
  blocks of a `figure*`); alternating "table, caption, table, caption" loose captions; `\multicolumn{1}{l}` followed
  by a line break; `NiceTabular`; algorithm numbers from the `loa` list (`A.1`); `\small{…}` font glue and
  `\abovedisplayskip` noise in the `.aux` caption keys; captions of tables wrapped in `\resizebox`.
- **Citations and references**: Unicode accents and letters in a `.bbl` (`{\L}ukasz`, `\'{e}`), `\penalty0`, author-year
  labels, natbib/apacite variants (`\citeyearpar`, `\citeNP`, `\footcite`, `\shortcite`…), `\subref`, `\crefrange`,
  `\labelcref`, `\namecref`.
- **Math**: `$` lost after a comment at the end of a formula (`…}%$`), `|` and `\|` in table cells, code blocks and
  list items left alone by the math clean-ups, display math inside list items and tables, `\nicefrac`/`\sfrac`,
  `\mathbbm`, `\sl`, pictures inside formulas, paired delimiters that glued a command to the next letter,
  algorithm2e `\;` inside `$…$`, `\eIf`, `\lIf`.
- **Lua**: tables that are mostly text count as content; sub-captions kept inside figures; single-cell tables
  unwrapped; header promotion skips blank spacing rows; code listings keep their language and a "Listado N" caption;
  line breaks in cells as `<br>` (14 tables stayed HTML before, 1 does now).
- **Sources and CLI**: e-print unpacked into a temporary folder and moved when complete (an interrupted download no
  longer leaves a half-extracted cache); at least 3 s between requests to the arXiv API across runs, `Retry-After`
  honoured, no retries on 4xx; network errors as one-line messages; `--mn` is validated; `arxiv` exits 1 when
  the conversion failed; JATS CDATA escaped.
- **Library regenerated**: the 124 arXiv-LaTeX `.md` files of the library were rewritten with this version (same
  front matter, MarginNote links included; the previous files were copied outside the library first). Against the
  previous ones: paragraphs with an odd `$` 40 -> 0, decimal numbers of the PDF missing from the text 248 -> 53,
  files with `error` diagnostics 15 -> 4 (the genuine `??`), unnumbered captions 42 -> 21, citations in the PDF's
  own format (author-year where the paper uses it), table values and marks that `\cellcolor` and ✓/✗ used to
  drop, listings as code instead of `#` headings, checklist answers and "Acknowledgments" headings back.
- `tests/regress.py --out DIR` keeps the reconverted files and skips `INDEX.md` and the PDF/JATS conversions.

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

## Robustness pass (diagnostics, recovery, no shell)
- `diagnostics.py`: typed diagnostics (code, severity, count, examples) in the conversion report and in
  `audit [--level]`; `audit` exits 1 on float differences or `error` diagnostics.
- `pandoc_recovery.py`: when pandoc cannot parse an environment it is replaced by a visible note (diagnostic
  `dropped-environment`) and the conversion is retried, up to 5 times, instead of aborting.
- `latexmk` runs with `-no-shell-escape` (untrusted source); all 124 papers recompiled with identical output.
- `ROADMAP.md`: what the project does, how, known failures and future work.

## Verification pass over random samples (18 papers read against their PDFs)
- Output is now plain Markdown: `Div`/`Span`/small caps are unwrapped, tables with colspan/rowspan or several header
  rows become pipe tables (math as `$...$`), and no MathML/`<span>`/`<div>` remains (14 tables that hold code
  blocks stay HTML, with their formulas turned back into TeX).
- Lost text recovered: macros whose body used `\text` (method names such as UDLM, Nemotron-Labs-Diffusion), prompts
  and algorithms inside `figure` environments, `algorithm2e` algorithm bodies, proofs written with delimited macros.
- Every numbered equation now carries its real `\tag{N}` (73 papers had equations without number); unresolved
  `(??)` references dropped from 90 to 12.
- Local `.sty` macros are expanded transitively; `\DeclarePairedDelimiter`, `\citeproc` citations, counter hacks
  (`\addtocounter`), `%` comments and back-to-back formulas inside math are handled.
- Float numbering tolerates a `.aux` that lists floats in placement order; longest-prefix caption matching.
- New tools: `tex2md audit DIR`; the golden set has 23 papers.

## Before the refactor (single `tex2md.py`)
- Float numbering from the `.aux` caption lines; recovery of captions lost in `subtable`, `minipage`,
  `parbox`, `captionof`, `SCfigure`, image-only tables and text-only tables.
- `\newcite` author citations, `table*`, `adjustbox`/`resizebox` around loose captions.
