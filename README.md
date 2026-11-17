# tex2md

Convierte artículos científicos a Markdown para la biblioteca de la tesis (`MindBooksBreakdown/md`).
Tres fuentes, un solo comando:

| Fuente | Subcomando | Cómo lo hace |
|---|---|---|
| Fuente LaTeX de arXiv | `tex2md arxiv` | descarga el e-print, lo compila una vez (`latexmk`) para obtener la numeración real (`.aux`) y la bibliografía (`.bbl`), lo preprocesa y lo convierte con pandoc |
| XML JATS (revistas abiertas, PubMed Central) | `tex2md jats` | pandoc; metadatos de Crossref |
| Solo PDF | `tex2md pdf` | [marker](https://github.com/VikParuchuri/marker) (OCR + modelo de diseño) en un entorno `uv` aislado |

Además mantiene la biblioteca: `frontmatter`, `add-mn`, `check`, `index`, `audit`, `fixmath` y `doctor`.

## Requisitos

- Python ≥ 3.11 (el paquete usa solo la biblioteca estándar).
- Herramientas externas: **pandoc** (≥ 3), **TeX** con `latexmk`, `pdflatex` y `latexpand`, y **uv** (solo para `pdf`).
- `tex2md doctor` comprueba todo y dice cómo instalar lo que falte.

## Uso sin instalar

No hace falta instalar nada: los scripts `tex2md.py`, `papers.py` y `doc2md.py` de la raíz ejecutan el CLI directamente. `make install` (opcional) deja un comando global `tex2md` con `uv tool install --editable .`; se quita con `uv tool uninstall tex2md`.

```sh
make dev            # entorno de desarrollo (.venv con pytest, ruff, mypy)
tex2md doctor
```

Los scripts antiguos siguen funcionando como envoltorios del mismo CLI:
`python3 tex2md.py ID -o out.md` ≡ `tex2md arxiv ID -o out.md`; `papers.py` y `doc2md.py` igual.

## Uso

```sh
tex2md arxiv 2603.23507v1 -o md/ding2026did_2603.23507.md --mn BOOKMD5::PCIC/Tesis/core/x.pdf
tex2md jats  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC123" -o md/x.md --doi 10.1000/x
tex2md pdf   paper.pdf -o md/x.md --doi 10.1000/x --mode balanced
tex2md index md && tex2md check md
tex2md audit md --level warning     # números de figuras y tablas frente al PDF compilado + diagnósticos
```

`arxiv` imprime un informe JSON de calidad (tablas en el tex vs. en el md, referencias y citas sin resolver) y
una lista de **diagnósticos tipados** (código, gravedad `error`/`warning`/`info`, cantidad y ejemplos): fórmulas
con `$` sin cerrar (`unbalanced-math`), texto del LaTeX que no llegó al Markdown (`lost-text`), llaves
reparadas (`unbalanced-braces`), entornos o definiciones que pandoc no pudo leer (`dropped-environment`,
`dropped-definition`), comandos que KaTeX no conoce, referencias y citas sin resolver… El comando termina con
estado 1 si la conversión no se completó. `audit` repite la comprobación de los `.md` ya escritos.

## Datos y configuración

El caché y los modelos viven **fuera** del código y se pueden mover con variables de entorno:

| Variable | Contenido | Por defecto |
|---|---|---|
| `TEX2MD_CACHE` | descargas, compilaciones y metadatos de arXiv | `~/.cache/tex2md` |
| `TEX2MD_MODELS` | pesos de marker | `~/.local/share/tex2md/models` |
| `TEX2MD_MARKER` | proyecto `uv` aislado que ejecuta marker | `<repo>/marker` |
| `TEX2MD_MN_ROOT` | carpeta Documents de MarginNote (iCloud) | ruta estándar de iCloud |

`marker` está fijado en 1.10.2 (la 2.x necesita un servidor `llama.cpp`) y no se mezcla con el paquete principal.

## Arquitectura

```
src/tex2md/
├── cli.py              subcomandos (argparse)
├── convert.py          arxiv → md como etapas: fetch → compile → flatten → preprocess → pandoc → postprocess → write
├── config.py           rutas y variables de entorno
├── sources/            arxiv.py (API y e-print) · jats.py · pdf.py (marker) · external.py (Crossref)
├── build/              latex.py (documento principal) · aux.py (etiquetas y flotantes del .aux)
├── preprocess/         LaTeX → LaTeX que pandoc entiende
│   ├── pipeline.py     RULES: lista ordenada de reglas con nombre (único lugar donde se fija el orden)
│   └── structure · floats · macros · tables · boxes · algorithms · references · citations
│       comments (`%`, `\input`, listados) · environments (forest, multicols, tblr, glosarios…)
│       symbols (`\ding`, `\cmark`…) · textloss (cajas y comandos que pandoc vacía) · braces (llaves sin cerrar)
├── postprocess/        limpieza del Markdown (encabezados, matemáticas)
├── filters/            filtros Lua de pandoc
│   ├── tex2md.lua      punto de entrada: orden de pasadas
│   └── lib/            state · headers · captions · floats · equations · document · labels · strutil
├── library/            front matter, índice, comprobación de enlaces con MarginNote
├── util/               tex.py (grupos balanceados, macros) · io.py (procesos, HTTP, YAML)
├── diagnostics.py      diagnósticos tipados (informe de conversión y `audit`)
├── pandoc_recovery.py  reintento sin el entorno, la definición o la línea del preámbulo que pandoc no puede leer
└── doctor.py
tests/                  unit/ · golden/ · regress.py
marker/                 proyecto uv aislado (pyproject + uv.lock)
```

### Cómo se numeran figuras y tablas

pandoc no conoce la numeración del PDF. `tex2md` compila una vez y lee del `.aux` (a) `\newlabel` → secciones,
ecuaciones y etiquetas, y (b) los `\caption` en orden (`floats.json`) → el número real de cada figura y tabla.
El filtro Lua empareja cada leyenda con su entrada del `.aux` por el comienzo del texto y el orden;
si el `.aux` no trae datos de ese tipo usa la etiqueta, y como último recurso un contador.

### Añadir una regla de preprocesado

1. Escribe una función `texto -> texto` en el módulo temático (`preprocess/floats.py`, `tables.py`, …).
2. Regístrala en `RULES` de `preprocess/pipeline.py` **en su sitio** (el orden importa) con un nombre.
3. Añade una prueba mínima en `tests/unit/` (entrada LaTeX → salida esperada) y, si arregla un artículo real, un caso golden.
4. Un módulo nuevo del paquete hay que añadirlo a git (`git add src/tex2md/<paquete>/`): `.gitignore` solo ignora
   `/build/` y `/dist/` de la raíz, no los paquetes de `src/` que se llamen así.

Qué hace, cómo, fallas conocidas y trabajo futuro: ver `ROADMAP.md`.

## Desarrollo

```sh
make check      # ruff + mypy + pytest (rápido)
make regress    # reconvierte los artículos golden desde el caché local y compara
python3 tests/regress.py DIR_CON_MD_BASE   # regresión completa contra una línea base
```

La regresión usa el front matter de cada `.md` base para repetir la conversión exacta y compara el cuerpo
byte a byte (ignora la fecha `converted`).
