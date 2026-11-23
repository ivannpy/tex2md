# Hallazgos de la reconversión de la biblioteca (2026-11-18)

> **Estado (2026-11-18, mismo día):** corregidos §1 (los cuatro fallos de pandoc), §2 (los cuatro casos), el
> diagnóstico de §2 (`missing-sections`, más `ambiguous-main-file`) y, de §3, las `??` de `kaplan2020` (32) y
> `hong2026oemdm` (6). Detalle en `CHANGELOG.md`. Causas que resultaron distintas de las supuestas aquí:
> `zheng2024time` no era una `\newenvironment` partida sino un verbatim hecho a mano con `\catcode` (`\zzz`);
> `sennrich2016bpe` tenía `\usepackage{listings}{` en el preámbulo; `davis2026scfm` además perdía los ejemplos
> dentro de las figuras; `kaplan2020` no era F14: nuestro etiquetador metía `\label` en `\newcommand{\ee}` y
> amsmath descartaba las etiquetas del autor; en `hong2026oemdm` `parse_aux` cortaba la clave en la primera `}`.
> **Nuevo hallazgo:** con `missing-sections` aparecieron 8 `.md` ya copiados con encabezados sin número o con
> otro número (`feng2025theory`, `gao2020pile`, `gat2024dfm`, `lanzillotta2025datasize`, `lee2024arithmetic`,
> `qin2026a2d`, `zhou2023distillspec`, `zhu2023diffusion`): ahora la numeración sale de la tabla de contenidos
> del `.aux`. La biblioteca se regeneró ese día.
>
> **Leyendas ausentes de §3, corregidas (2026-11-18):** cada una tenía su propia causa: `\caption[Caption]{…}`
> repetido (`besiroglu2024`), `\hyperref`/`\ref` en la leyenda del `.aux` y envoltorios de `\ref` con dos
> argumentos (`allenzhu2023`, `kaplan2020`), un `wrapfigure` sin leyenda dentro de un `figure*`, `wrapstuff`
> (`bansal2026`), `teaserfigure` (`chen2026bard`), `\twocolumn[\begin{@twocolumnfalse}…]` (`chang2022`), `ctable` y
> `picture` con `\put` (`chen2024dforcing`), `tabular` con más celdas que columnas (`deschenaux2026duo2`),
> `floatrow` (`kingma2021vdm`), `minipage` con dos tablas (`lu2025adablock`), una tabla bajo dos `minipage`
> (`wang2026computegap`), dos `\caption` en un `figure` con `subfigure` (`diffusiongemma2026`), `\strut` en una
> leyenda (`yamabe2025priming`) y cajas `tcolorbox` definidas dentro de cada figura (`tae2025tess2`).
> `vonrutte2026scaling` y `pan2025demasking` eran falsos positivos de `audit` (`\caption[]`).
>
> **Resto de §3 y §4, corregidos (2026-11-18):** el `unbalanced-math` de `chen2024dforcing` era
> `$\xtk=\ $ForwardDiffuse$…$` (un espacio de control antes del `$` que cierra; los de `lee2024arithmetic` y
> `schiff2026clock` ya no aparecían). De las `??` sueltas, dos no eran reales: `brown2020gpt3` (una clave con
> comas, `\ref{section:Language Modeling, Cloze, and Completion Tasks}`) y `golovneva2024reverse` (texto dentro de
> `\if 0 … \fi`); las de `arriola2025`, `hendrycks2021mmlu`, `lin2021limitations`, `mirzadeh2019takd`,
> `sahoo2025duo` y `su2026opdlm` también están en el PDF (ROADMAP F14). §4: los `$` alrededor de texto se quitan
> del título de la API, `arxiv --title` fija el título del PDF y `regress.py` repite el del encabezado (los
> títulos corregidos a mano sobreviven a una regeneración); `fixmarker` repara el `<sup>` escapado, la portada de
> JSTOR, los niveles con secciones I./A. y los encabezados falsos evidentes; después, los que solo etiquetan algo
> en artículos numerados (autor y afiliación de `efron2011tweedie` y `shannon1948`, «Example:», los pasos de
> pseudocódigo de `bengio2003nplm`, las 31 etiquetas de muestras de `radford2019gpt2`). Queda `nvidia2023h100`, un
> folleto sin secciones numeradas, cuyos encabezados se dejan como están. El HTML de las tablas (de marker y de LaTeX) se corrigió ese mismo
> día; los `<br>` de las celdas se conservan (es el salto de línea de una celda de tabla Markdown).

Contexto: en `~/Documents/MindBooksBreakdown/md/` había 180 `.md` del conversor anterior
(`papers_taller/articulos`), traídos de `pdf/md/`. Se reconvirtieron con tex2md:

- 156 de LaTeX de arXiv con `tests/regress.py md --out DIR --only ...` (repite id y versión del encabezado),
  más 2 que venían de PDF pero tienen id de arXiv (GAN `1406.2661v1`, BPE `1508.07909v5`) con `tex2md.py`.
- 13 artículos que solo existen en PDF con `doc2md.py pdf` (marker, `balanced`): los 13 copiados (ver §4).

Resultado de LaTeX: **150 copiados a la biblioteca, 8 descartados** (4 fallos de pandoc y 4 conversiones que
pierden contenido). Los 8 conservan en la biblioteca el `.md` del conversor anterior. Las fuentes están en el
caché (`~/.cache/tex2md/<id>v<N>/`), así que todo se reproduce sin red:

```
python3 tex2md.py <id>v<N> -o /tmp/x.md
```

Copias de la biblioteca anteriores a la reconversión: `~/Documents/MindBooksBreakdown_md_backup_2026-11-18b/`
(sirven de comparación: el `.md` viejo de cada caso de §2 sí tiene el cuerpo completo).

---

## 1. Fallos de pandoc (no se produce `.md`)

| Archivo | id | Error | Causa probable | Corrección |
|---|---|---|---|---|
| `diffusiongemma2026_2608.00146.md` | `2608.00146v1` | `unexpected ~` en `\section{Samples Before and After \texorpdfstring{\sdrl}~Training}` (línea 990 del aplanado) | `\texorpdfstring` con **un solo argumento** (el autor olvidó el segundo); LaTeX lo tolera porque se come el `~` como segundo argumento | regla de preprocesado: `\texorpdfstring{A}{B}` → `A`, y si el segundo argumento no es un grupo, tomar el siguiente token. Además, `pandoc_recovery.py` no recupera errores dentro de `\section{…}`: valorar que pueda quitar el comando problemático del título en vez de abortar |
| `zheng2024time_2409.02908.md` | `2409.02908v6` | `unexpected end of input` (línea 1985, final del archivo) | el aplanado tiene **una llave sin cerrar** y un `\end{zzz}` sin `\begin` (probablemente de una `\newenvironment` que el preprocesado parte mal) | ver por qué `braces.py` no repara la llave; buscar el origen de `\end{zzz}` |
| `sennrich2016bpe_1508.07909.md` | `1508.07909v5` | `unexpected end of input` (línea 772, final) | **una llave sin cerrar**; el archivo tiene un `lstlisting` | igual que el anterior: `braces.py` no la detecta; comprobar si la llave está dentro o cerca del listado |
| `davis2026scfm_2605.07820.md` | `2605.07820v2` | `unexpected end of input` (línea 1542, final) y avisos `Could not load include file \jobname.vrb` (líneas 1049, 1052, 1081, 1300) | llaves y entornos **balanceados**; los `\jobname.vrb` son archivos temporales de `fancyvrb`/`beamer` (verbatim escrito a disco) que latexpand deja como `\input` | reemplazar o quitar los `\input{\jobname.vrb}` antes de pandoc; luego ver si el fin de entrada desaparece |

En los cuatro, la recuperación de `pandoc_recovery.py` no consigue evitar el fallo (F1 del `ROADMAP.md`). No
comprobé si llegó a reintentar o si el error quedó fuera de lo que sabe quitar.

## 2. Conversiones que pierden contenido sin error (lo más grave: el informe dice `ok`)

| Archivo | id | Síntoma | Causa | Corrección |
|---|---|---|---|---|
| `chen2026bard_2604.16514.md` | `2604.16514v5` | 3 encabezados en vez de 17; el `.md` es la respuesta a revisores | hay 4 archivos con `\documentclass` y se compiló **`rebuttal_v1.tex`**. El `00README.json` de arXiv dice `"usage": "toplevel", "filename": "sample-sigconf.tex"` | **leer `00README.json`** (y el antiguo `00README.XXX`) del e-print y usar el `toplevel` que declara; es la fuente de verdad de arXiv |
| `sun2023hollow_2211.16750.md` | `2211.16750v2` | 9 encabezados en vez de 35; el texto termina en notas del autor («I previously guess…») | se eligió **`main.tex`** (un borrador de notas) en vez de **`iclr2023_conference.tex`** (el artículo, que hace `\input{tex/*.tex}`). No hay `00README` | heurística de archivo principal más robusta: preferir el que (a) tiene un `.bbl` del mismo nombre enviado por los autores (`iclr2023_conference.bbl` está en el e-print), (b) alcanza más `.tex` por `\input`/`\include`, (c) produce el PDF con más páginas; el nombre `main.tex` no debe ganar por sí solo |
| `he2021exposure_1905.10617.md` | `1905.10617v10` | 12 encabezados en vez de 24, la mitad del texto | el principal (`emnlp2021.tex`) carga el cuerpo con **`\subfile{sec_intro.tex}`** etc. (paquete `subfiles`), que latexpand no expande | regla de aplanado: `\subfile{X}` → contenido de `X` sin su preámbulo (`\documentclass[...]{subfiles}` … `\begin{document}` … `\end{document}`) |
| `kuleshov2026modern_2610.33391.md` | `2610.33391v1` | 91 citas sin resolver (`[@arriola2025block]`), `bib_entries: 0` | el `.bbl` es de **`ACM-Reference-Format`**: `\bibitem[Arriola et~al\mbox{.}(2025a)]%` con la clave en la **línea siguiente** (`{arriola2025block}`) y campos `\bibfield{author}{\bibinfo{person}{…}}`, `\showarticletitle{…}` | ampliar el lector de `.bbl`: permitir `%` + salto de línea entre `[…]` y `{clave}`, y entender las macros `\bibinfo`/`\bibfield`/`\showarticletitle` de ACM |

Propuesta de red de seguridad: un diagnóstico `error` cuando el `.md` tiene muchos menos encabezados que el
número de `\section` del `.aux` (o cuando el PDF compilado tiene muchas menos páginas que el PDF de arXiv). Los
cuatro casos de esta sección habrían saltado.

## 3. Avisos de `audit --level error` sobre los 150 copiados

Se copiaron porque, comparados con el `.md` anterior, no pierden secciones ni texto y ganan leyendas numeradas;
pero quedan estos problemas:

**Leyendas ausentes** (están en el `.aux` y no en el `.md`):

| Archivo | Faltan |
|---|---|
| `allenzhu2023_2309.14402` | Figura 6, Figura 14 |
| `besiroglu2024chinchilla_2404.10102` | Tabla 1 |
| `chang2022maskgit_2202.04200` | Figura 9 |
| `chen2024dforcing_2407.01392` | Tablas 1, 2, 3 |
| `deschenaux2026duo2_2602.21185` | Tablas 7, 11 |
| `kaplan2020_2001.08361` | Figura 7 |
| `kingma2021vdm_2107.00630` | Tabla 2, Figura 5 |
| `lu2025adablock_2509.26432` | Tablas 5, 6 |
| `tae2025tess2_2502.13917` | Figuras 5, 6 |
| `yamabe2025priming_2510.00565` | Tablas 6, 7 |
| `vonrutte2026scaling_2512.10858` | al revés: Figura 1 en el `.md` y no en el `.aux` |

**Errores**:

| Archivo | Diagnóstico | Detalle |
|---|---|---|
| `kaplan2020_2001.08361` | `unresolved-reference` ×32 | «Equation (??)». El conversor anterior mostraba la etiqueta: `[eq:FundamentalLikelihioodvsModelandSte…]` (con errata «Likelihiood»): probablemente `\ref` y `\label` no coinciden en la fuente; **confirmar contra el PDF** si también muestra `??` (sería F14) |
| `hong2026oemdm_2602.02112` | `unresolved-reference` ×6 | «Eq. ??\_loss», «Algorithm ??-train»; el viejo mostraba `[alg:OeMDM{}-train]`. Hipótesis (sin comprobar en la fuente): el `\ref` contiene una macro del nombre del método que no se expande antes de buscar la etiqueta |
| `brown2020gpt3`, `hendrycks2021mmlu`, `golovneva2024reverse`, `lin2021limitations` | `unresolved-reference` ×1–2 | revisar contra el PDF |
| `chen2024dforcing_2407.01392` | `unbalanced-math` | `$\mathbf{x}_t^{k_t}=\$ForwardDiffuse$(…)$`: un `\textrm`/`\text` con `$` dentro de un algoritmo |
| `lee2024arithmetic_2307.03381` | `unbalanced-math` ×3 | listados con `escapeinside={(*@}{@*)}` y `\highlighttext{…}`: las secuencias de escape no se procesan |
| `schiff2026clock_2610.00894` | `unbalanced-math` | dentro de un bloque de ejemplo (GSM8K) en cita |

Totales del `audit` sobre los 154 reconvertidos: `unknown-math-macro` 2354, `unresolved-citation` 93 (91 son de
`kuleshov2026modern`), `html-residue` 30, `unresolved-reference` 43, `unbalanced-math` 5. Las
`unknown-math-macro` no se han mirado: con 2354 usos conviene un `audit --level warning` agrupado por archivo
antes de decidir (F2).

## 4. Otros

- **Títulos de la API de arXiv con matemáticas**: `2602.21185` llega como `The Diffusion Duality, Chapter II:
  $Ψ$-Samplers`. En la biblioteca se conservó el título anterior (`Ψ-Samplers`). Propuesta: quitar los `$` del
  título cuando lo de dentro es texto Unicode.
- **Versión del PDF frente a arXiv**: el PDF de GAN en la biblioteca es la versión de NeurIPS («Generative
  Adversarial Nets»); arXiv `1406.2661v1` se titula «Generative Adversarial Networks». No es un error del
  conversor; en la biblioteca se conservó el título del PDF. Quizá `frontmatter` debería permitir fijar el título.
- **Tanda de PDF (marker)**: 13 artículos (`bengio1994longterm`, `bengio2003nplm`, `blei2003lda`,
  `dempster1977em`, `efron2011tweedie`, `hochreiter1997lstm`, `mikolov2010rnnlm`, `nvidia2023h100`,
  `owen2013mc_varianza`, `radford2019gpt2`, `rnd12025_reporte`, `shannon1948`, `shannon1951`), 308 páginas en
  3 h 10 min (~0.6 min/página). Los 13 terminaron con estado 0 y **se copiaron a la biblioteca**: frente al pypdf
  anterior (sin encabezados ni ecuaciones; el de `bengio1994longterm` tenía 101 palabras y el de `hochreiter1997lstm`
  era texto ilegible, «Ho c hreiter», «/1/9/9/7») tienen todas las secciones hasta los apéndices, ecuaciones en
  LaTeX y tablas. Fallas que quedan, por orden de importancia:
  - **`<sup>` escapado dos veces** (error, se ve en el texto): las notas al pie salen como
    `<sup>&</sup>lt;sup>1</sup>Nyquist, H., …` en lugar de `<sup>1</sup>`. Afecta a **11 `.md`** de la biblioteca,
    los 7 de esta tanda (`shannon1948` ×18, `hochreiter1997lstm` ×5, `shannon1951` ×5, `mikolov2010rnnlm` ×4,
    `nvidia2023h100` ×3, `radford2019gpt2` ×2, `bengio1994longterm` ×1) y 4 de marker anteriores (`amin2020humor`,
    `mi2026relics`, `ma2026diff4tst`, `sahoo2026foundations`). Parece que un paso escapa `<` a `&lt;` y otro vuelve a
    envolver el `&` en `<sup>`. Reproducir con `md/shannon1948.md` (línea ~41); `audit` debería detectarlo
    (patrón `&</sup>lt;`). Tras corregir, `fixmath`/un post-proceso puede arreglar los 11 sin volver a pasar marker.
  - **Niveles de encabezado incoherentes**: marker asigna `#`/`##`/`####` según el tamaño de letra, no según la
    jerarquía (`bengio1994longterm`: `## A. Analysis` y luego `# B. Multi-Grid Random Search` dentro de la misma
    sección V; `#### II.` al mismo nivel que `## I.`). Propuesta: renivelar por la numeración (`I.`/`1.` → `##`,
    `A.`/`1.1` → `###`) cuando la hay.
  - **Encabezados falsos**: texto destacado convertido en `#`: muestras del apéndice de `radford2019gpt2`
    («# Outfit 2: High End», «# **GPT-2** completion»), la cadena «# A B B A B A …» de `shannon1948`, «# 1. FORWARD
    PHASE» (pseudocódigo de `bengio2003nplm`), «# • Setting 4: …» y «# Conditional generation sample» de
    `rnd12025_reporte`, la leyenda de una figura entera como `##` en `nvidia2023h100`, autor y afiliación como `#`
    en `efron2011tweedie`.
  - **Portadas de repositorio**: `dempster1977em` empieza con la página de JSTOR («### Stable URL:», condiciones
    de uso) y repite el título como segundo `#`. Propuesta: descartar la primera página si es una portada de JSTOR.
  - HTML residual (`<br>`, `<sup>`, `<sub>`) además del caso anterior: 0–31 por archivo, sobre todo en tablas.

## 5. Orden sugerido

1. `00README.json` y heurística del archivo principal (§2: `chen2026bard`, `sun2023hollow`). Riesgo bajo, efecto
   grande: hoy un e-print con varios `\documentclass` puede producir otro documento sin ningún aviso.
2. `\subfile` (§2: `he2021exposure`).
3. Diagnóstico de cuerpo incompleto (encabezados del `.md` frente a `\section` del `.aux`).
4. `.bbl` de ACM (§2: `kuleshov2026modern`).
5. Los cuatro fallos de pandoc (§1).
6. Leyendas ausentes y `??` de §3, empezando por `kaplan2020` y `hong2026oemdm`.
7. `<sup>` escapado dos veces en la salida de marker (§4): afecta 11 `.md` y es mecánico de reparar.

Para cada corrección: caso en `tests/golden/` con el artículo real, `make check` y `make regress`; después
volver a convertir los 8 descartados y los de §3 en la biblioteca (flujo «Regenerar los `.md`» de
`MindBooksBreakdown/CLAUDE.md`).
