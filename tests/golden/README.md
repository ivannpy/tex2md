# Golden conversions

Each `.md` is the accepted output of `tex2md arxiv` for one paper. They cover the structures that
have broken conversions before:

| File | Why it is here |
|---|---|
| kim2016sequence | `\newcite` author citations in a numbered bibliography |
| sanh2019distilbert | two captioned `parbox` tables side by side |
| arriola2025e2d2 | tables with `\captionof{table}` inside a figure's minipages |
| zhou2026dllm | `subtable`s and multi-panel floats sharing one caption |
| zheng2023reparameteri | two captioned minipages in one table, `\captionsetup{type=table}` |
| uno2026 | colour-box macros, `table*`, many appendix tables |
| song2019generative | two floats with identical captions |
| fu2026nemotron | `\captionof{figure}` outside any float |
| baek2026generative | `wrapfigure`, `\captionof` followed by `adjustbox` |
| gong2024diffullama | text-only tables with `tcolorbox` |
| zhang2024lingualift | table holding only an image |
| sahoo2024mdlm | general reference (equations, appendix, tables) |
| zeng2025diffusionvl | `\captionof{figure}` inside `\twocolumn[{...}]` |
| wu2023ardiffusion | figure with one `\caption` and one `\captionof{table}` in sibling minipages |
| hoogeboom2021 | figure nested inside a minipage of another figure, with `subfigure`s |
| cheng2025sdar | text-only tables numbered in document order next to normal ones |
| li2025situ | `algorithm2e` algorithm (`\KwIn`, `\For{..}{..}`, `\;`) whose body used to be lost |
| liu2023prefix | every numbered equation gets its `\tag{N}` (labels added before compiling) |
| liu2022flow | delimited-argument macros (`\def\bb#1\ee{..}`) around equations |
| zhou2024difflm | figures that hold a prompt in a code box: text kept, not just the caption |
| schiff2025guidance | macro names set with `\text` (`\method` -> UDLM) and tables with spans |
| lambert2025reinforcemen | pandoc-written `longtable` with `\citeproc`, repeated headers and `\endlastfoot` |

The test replays the conversion from the local cache (`~/.cache/tex2md`), so it needs no network once
the sources were downloaded. Run `make regress`. After an intended change, regenerate the files with
`tex2md arxiv ID -o tests/golden/NAME.md` (keeping the `--mn` links of the front matter) and review the diff.
