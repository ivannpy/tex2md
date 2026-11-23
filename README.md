# tex2md

Convierte artículos científicos a Markdown conservando la numeración del PDF. A partir de la fuente LaTeX de arXiv genera un `.md` con secciones, ecuaciones, tablas, leyendas, algoritmos y bibliografía numerados como en el artículo compilado, y un informe de calidad que señala lo que no se pudo convertir bien. También convierte XML JATS (PubMed Central y editoriales abiertas) y, cuando no hay otra fuente, PDF.

Sirve para leer, buscar y citar un artículo en texto plano sin fiarse de la capa de texto del PDF: las cifras de las tablas, las ecuaciones con su número y las referencias cruzadas salen como en el original.

Verificado contra el `.aux` de 189 artículos reales de arXiv (124 de una biblioteca de tesis sobre modelos de lenguaje de difusión y 65 de temas y años variados, de 2012 a 2025): los números de figuras y tablas coinciden con los del PDF compilado. La excepción es un artículo cuyo `.aux` no registra las tablas, que no se pudo comprobar.

## Instalación

Requisitos:

- Python 3.11 o superior. El paquete no tiene dependencias de Python.
- [pandoc](https://pandoc.org) 3 o superior (probado con 3.14). Algunas distribuciones de Linux traen versiones anteriores.
- TeX Live o MacTeX con `latexmk`, `pdflatex` y `latexpand` (`tlmgr install latexpand`). Conviene una distribución completa: las fuentes de arXiv usan paquetes de todo tipo y, si falta uno, la compilación sigue pero la numeración queda incompleta.
- [uv](https://docs.astral.sh/uv/), solo para convertir PDF.
- Acceso a arxiv.org. Cada artículo se guarda en un caché y las conversiones posteriores no usan la red.

Descargar e instalar el comando `tex2md`:

```sh
git clone https://github.com/ivannpy/tex2md.git ~/Developer/tex2md
cd ~/Developer/tex2md
pip install .             # mejor dentro de un entorno virtual
tex2md doctor             # comprueba las herramientas externas y los directorios de datos
```

`doctor` indica qué falta y cómo instalarlo (en macOS: `brew install pandoc uv` y MacTeX).

Clona el repositorio en una carpeta local, no en iCloud ni en otra carpeta sincronizada: los entornos de uv (`.venv`, `marker/.venv`) guardan rutas absolutas, y la sincronización puede quitar archivos del disco o duplicarlos.

**WSL.** La copia de macOS ya no está en iCloud, así que desde WSL no se ve. Hay que clonar el repositorio también en el sistema de archivos de WSL (por ejemplo en `~/Developer/tex2md`, no bajo `/mnt/c/`) y ejecutar ahí `tex2md doctor`. Cada copia tiene sus propios entornos y su propio caché (`~/.cache/tex2md`), y los cambios pasan de una a otra con git (`git push` / `git pull`).

Alternativas:

```sh
uv tool install .                                           # comando aislado, sin gestionar un entorno
pip install git+https://github.com/ivannpy/tex2md.git      # sin clonar
```

Sin instalar nada, desde la raíz de la copia clonada: `python3 tex2md.py ID -o salida.md` equivale a `tex2md arxiv ID -o salida.md`, y `python3 papers.py SUBCOMANDO ...` ejecuta cualquier otro subcomando (`doctor`, `audit`, `jats`...). `doc2md.py` hace lo mismo que `papers.py`.

## Uso

### arXiv

```sh
tex2md arxiv 1910.01108 -o distilbert.md          # última versión
tex2md arxiv 1910.01108v4 -o distilbert.md        # versión concreta
```

Un id con versión fija la fuente que se convierte; úsalo cuando la numeración deba coincidir con un PDF concreto, porque las versiones de un artículo pueden diferir. `--title` sustituye el título de arXiv (por ejemplo, el de la versión publicada); los `$` que la API pone alrededor de texto sin fórmula (`$Ψ$-Samplers`) se quitan solos. El comando imprime el informe JSON (ver más abajo) y termina con estado 1 si la conversión no se completó.

### JATS

```sh
tex2md jats "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC7759461" \
    -o numpy.md --doi 10.1038/s41586-020-2649-2
```

La fuente puede ser una URL o un archivo XML. Los metadatos salen de Crossref a partir de `--doi`; sin DOI se pasan `--title`, `--authors "Nombre Apellido; Nombre Apellido"`, `--year` y `--venue`.

### PDF

```sh
tex2md pdf articulo.pdf -o articulo.md --doi 10.1000/xyz
tex2md pdf articulo.pdf -o articulo.md --title "Título" --authors "Nombre Apellido; Nombre Apellido" --year 2024 --mode fast
```

Usa [marker](https://github.com/VikParuchuri/marker) (OCR y modelo de diseño) en un entorno que uv crea la primera vez a partir de `marker/`, y descarga varios GB de pesos en `TEX2MD_MODELS`. Está fijado en la versión 1.10.2 porque la 2.x necesita un servidor externo de llama.cpp. `--mode balanced` (predeterminado) hace OCR de todas las páginas, es más lento y mejora las ecuaciones; `fast` solo lo hace donde la capa de texto es mala. El resultado tiene errores de reconocimiento: úsalo solo si no existe la fuente LaTeX ni JATS. Con una instalación normal de pip hay que apuntar `TEX2MD_MARKER` a la carpeta `marker/` de una copia clonada.

### Comprobar una carpeta de conversiones

```sh
tex2md audit md/ --level warning
```

Para cada `.md` convertido desde LaTeX compara los números de figuras y tablas con los del PDF compilado (guardado en el caché) y muestra los diagnósticos de su cuerpo. Termina con estado 1 si falta una leyenda o hay un diagnóstico de gravedad `error`. `--level` fija la gravedad mínima que se imprime.

### Biblioteca de `.md` (opcional)

`tex2md index DIR` regenera `INDEX.md`, una tabla con todos los `.md` de la carpeta a partir de su encabezado. `tex2md fixmath ARCHIVOS` repite la limpieza de matemáticas sobre `.md` ya convertidos. `tex2md fixmarker ARCHIVOS` repara en `.md` hechos con marker lo que marker hace mal (notas al pie `<sup>&</sup>lt;sup>1</sup>`, la portada de JSTOR, niveles de encabezado con secciones I., II. y subsecciones A., B., y encabezados que son una viñeta, una fila de letras, una frase larga o, en un artículo con secciones numeradas, una etiqueta: el autor, «Example:», los pasos de un algoritmo, las muestras de un apéndice); las conversiones nuevas ya salen así. `tex2md frontmatter ARCHIVO --arxiv ID` añade el encabezado a un `.md` que no lo tiene.

Integración con MarginNote: `--mn BOOKMD5::RUTA` guarda en el encabezado el vínculo con un documento de MarginNote, `tex2md add-mn ARCHIVO BOOKMD5::RUTA` añade otro y `tex2md check DIR` lista los vínculos cuyo PDF ya no existe en la carpeta de MarginNote (`TEX2MD_MN_ROOT`).

## Salida

Cada conversión es un solo `.md` sin imágenes. Primeras líneas de `tex2md arxiv 1910.01108`:

```markdown
---
title: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
authors: ["Victor Sanh", "Lysandre Debut", "Julien Chaumond", "Thomas Wolf"]
year: 2019
arxiv: "1910.01108"
arxiv_version: "v4"
source: "arXiv LaTeX, tex2md 0.1"
converted: 2026-11-17
---

# DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
```

Una tabla y una ecuación numerada (de `1910.01108` y de `2406.07524`):

```markdown
**Tabla 1.** **DistilBERT retains 97% of BERT performance.** Comparison on the dev sets of the GLUE benchmark. [...]

| Model      | **Score** | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STS-B | WNLI |
|:-----------|:---------:|:----:|:----:|:----:|:----:|:----:|:----:|:-----:|:-----:|:----:|
| ELMo       |   68.7    | 44.1 | 68.6 | 76.6 | 71.1 | 86.2 | 53.4 | 91.5  | 70.4  | 56.3 |
| DistilBERT |   77.0    | 51.3 | 82.2 | 87.5 | 89.2 | 88.5 | 59.9 | 91.3  | 86.9  | 56.3 |

$$\begin{equation}
\tag{1}
    {\mathbf z}_t = \sqrt{\alpha_t}  {\mathbf x}+ \sqrt{1 - \alpha_t}  \boldsymbol\epsilon
\end{equation}$$
```

Convenciones:

- Encabezado YAML con título, autores, año, id y versión de arXiv, fuente y fecha; más `doi` y `venue` en JATS y PDF, y `marginnote` si se usó `--mn`.
- Secciones y apéndices con el número que les da el PDF (`# 1 Introduction`, `## A.1 Generic case`).
- Cada figura se reduce a su leyenda: `**Figura N.** ...`. Las tablas llevan `**Tabla N.**` y son tablas de tuberías; solo caen a HTML las que no caben en ese formato (por ejemplo, con bloques de código en las celdas). Los algoritmos salen como `**Algoritmo N.**` seguido de sus pasos, y los listados de código como `**Listado N.**` y un bloque de código.
- Ecuaciones como `$...$` y `$$...$$` en LaTeX; las numeradas llevan `\tag{N}` con el número del PDF. Las macros del artículo (`\newcommand`, `\DeclarePairedDelimiter`, paquetes `.sty` locales) se expanden para que la fórmula se pueda dibujar con KaTeX o MathJax.
- Citas resueltas con la bibliografía compilada (autor-año o número, según el estilo del artículo); la bibliografía se conserva en su sección final.
- Markdown plano: sin `<div>`, `<span>` ni MathML.
- Las etiquetas que genera el conversor (`Figura`, `Tabla`, `Algoritmo`, `Listado`, `Autores` y el aviso bajo el título) están en español; el texto del artículo conserva su idioma.

## Cómo funciona

```
id de arXiv
  -> descarga del e-print y de los metadatos (caché)
  -> compilación con latexmk, sin shell: .aux y .bbl
  -> aplanado con latexpand
  -> preprocesado: reglas ordenadas, de LaTeX a LaTeX
  -> pandoc con filtros Lua
  -> posprocesado del Markdown
  -> .md e informe JSON
```

- **Compilación.** pandoc no sabe qué número tendrá cada elemento: lo decide la compilación. tex2md compila la fuente una vez (tolerante a errores y con `-no-shell-escape`, porque la fuente es de terceros) y lee del `.aux` los números de secciones, figuras, tablas, ecuaciones y algoritmos, y del `.bbl` la bibliografía. Antes de compilar da una etiqueta propia a cada ecuación numerada que no tenga una, para que el `.aux` registre su número.
- **Preprocesado.** Reglas con nombre y orden fijo (`preprocess/pipeline.py`), cada una `texto -> texto` y probada por separado. Resuelven `\ref` y `\cite` a lo que imprime el PDF, expanden macros que pandoc no entiende, normalizan los flotantes (subfiguras, `minipage`, `\captionof`, varias leyendas en un flotante) y reescriben los algoritmos. Otro grupo rescata texto que pandoc descartaría sin avisar: un comando definido en un `.sty` ajeno se traga todos sus argumentos entre llaves, y lo mismo pasa con cajas, colores de celda o listados leídos de archivos.
- **pandoc y filtros Lua.** Numeran y reescriben figuras, tablas y ecuaciones con los datos del `.aux`. Si pandoc no puede leer una construcción, tex2md localiza el entorno, la definición o la línea del preámbulo que la contiene, la quita y reintenta (hasta 5 veces) en lugar de abortar. Si lo quitado es un entorno, queda una nota visible en el texto.
- **Posprocesado.** Aísla las ecuaciones de display, restaura el `\text` dentro de las fórmulas, quita comentarios `%` y restos de contadores, y protege los bloques de código de esas limpiezas.

## Informe y diagnósticos

`tex2md arxiv` imprime al terminar un informe JSON (también queda en el caché como `report.json`) con `ok`, las tablas y figuras de entrada y de salida, las referencias y citas sin resolver y la lista `diagnostics`. Cada diagnóstico tiene código, gravedad, cantidad y ejemplos. `error` significa contenido erróneo o ausente. `audit` calcula los mismos sobre el cuerpo de los `.md` ya escritos.

| Código | Gravedad | Qué indica |
|---|---|---|
| `unbalanced-math` | error | párrafo con un número impar de `$`: una fórmula perdió o ganó un delimitador y el texto y las matemáticas quedan intercambiados |
| `unresolved-reference` | error | un `\ref` sin resolver quedó como `??` en el Markdown |
| `unresolved-latex-reference` | error | `\ref` a una etiqueta que no está en el `.aux` (el mismo problema, visto en el LaTeX) |
| `dropped-environment` | error | pandoc no pudo leer un entorno y se sustituyó por una nota: su contenido no está en el Markdown |
| `missing-sections` | error | secciones numeradas del `.aux` (tabla de contenidos) sin un encabezado con ese número en el Markdown: se perdió parte del cuerpo o los encabezados perdieron su número |
| `compile-failed` | warning | LaTeX no produjo el PDF (se muestra el primer error del registro): faltan los números del `.aux` y `missing-sections` no puede comprobar si se perdió parte del cuerpo; comparar con el PDF. Un paquete que no está instalado se quita y se vuelve a compilar (`dropped_packages` en el informe) |
| `ambiguous-main-file` | warning | varios archivos del e-print son documentos completos y no hay `00README` que diga cuál es el principal: se eligió uno por heurística (comprobar que no sea una respuesta a revisores o un borrador) |
| `dropped-definition` | warning | se quitó una definición que pandoc no podía leer, o una macro que se expandía a sí misma |
| `unbalanced-braces` | warning | llaves sin pareja en el LaTeX: se cerró el grupo al final del párrafo o se descartó la sobrante |
| `lost-text` | warning | tramos de 12 o más palabras de la fuente que no están en el Markdown; es una red de seguridad y da falsos positivos |
| `unknown-math-macro` | warning | comandos en ecuaciones que KaTeX y MathJax no conocen, normalmente macros del autor sin expandir |
| `unresolved-citation` | warning | cita que quedó como clave de pandoc (`[@clave]`): no se encontró en la bibliografía |
| `unresolved-latex-citation` | warning | `\cite` de una clave que no está en la bibliografía |
| `html-residue` | warning | HTML en el Markdown, normalmente una tabla con bloques de código en las celdas |
| `pandoc-warning` | warning | aviso de pandoc |
| `equation-without-tag` | info | ecuación sin `\tag{N}`: el artículo no le dio número (`autonum`, `\nonumber`) o no se pudo recuperar |
| `unnumbered-float` | info | figura o tabla sin número: paneles o flotantes sin `\caption` propio |

## Configuración

| Variable | Contenido | Por defecto |
|---|---|---|
| `TEX2MD_CACHE` | fuentes descargadas, compilaciones, metadatos de arXiv e informes | `~/.cache/tex2md` |
| `TEX2MD_MODELS` | pesos de marker | `~/.local/share/tex2md/models` |
| `TEX2MD_MARKER` | proyecto uv que ejecuta marker | `marker/`, junto a `src/` |
| `TEX2MD_MN_ROOT` | carpeta Documents de MarginNote | la de iCloud en macOS |

El caché ocupa del orden de 10 MB por artículo. Se puede borrar, pero `audit` y las conversiones sin red dependen de él. Al consultar la API de arXiv se espera al menos 3 s entre peticiones, también entre ejecuciones distintas.

## Límites

- Necesita la fuente LaTeX. Los envíos de arXiv que solo contienen el PDF (o un envoltorio con `\includepdf`) y las fuentes en TeX plano o LaTeX 2.09 no se convierten, y el error lo dice; para ellos queda `pdf`.
- Las figuras se reducen a su leyenda; no se extraen imágenes.
- Las macros generadas por programa (`\foreach` con `\csname`, `\NewDocumentCommand` con argumentos `e_`) y los `\if...\fi` con argumentos opcionales pueden quedar sin expandir dentro de las ecuaciones; `unknown-math-macro` los cuenta.
- La fuente de terceros se compila con `-no-shell-escape`, pero sigue siendo TeX procesando entrada no confiable. Para fuentes dudosas conviene un contenedor.
- Las etiquetas del conversor están en español y todavía no se pueden cambiar.

Las fallas conocidas, con ejemplos, y el trabajo previsto están en [ROADMAP.md](ROADMAP.md) (en español); los cambios, en [CHANGELOG.md](CHANGELOG.md).

## Desarrollo

```sh
pip install -e ".[dev]"        # o: make dev (usa uv)
pytest -m "not slow"           # o: make test
ruff check src tests && mypy   # o: make check, que ejecuta ruff, mypy y las pruebas rápidas
```

`tests/golden/` guarda la salida aceptada de artículos reales, cada uno por una estructura que dio problemas (ver `tests/golden/README.md`). `pytest -m slow` (o `make regress`) los reconvierte desde el caché y compara byte a byte; los que no están en el caché se omiten, y `python3 tests/regress.py tests/golden` lo llena descargando las fuentes. Se generaron con pandoc 3.14: otra versión puede cambiar detalles de formato. `tests/regress.py DIR --out NUEVO` reconvierte una carpeta de `.md` y deja el resultado en `NUEVO`.

Estructura de `src/tex2md/`:

| Ruta | Contenido |
|---|---|
| `cli.py` | subcomandos |
| `convert.py` | etapas de `arxiv`: descarga, compilación, aplanado, preprocesado, pandoc, posprocesado, escritura |
| `sources/` | arXiv (API y e-print), JATS, PDF con marker, Crossref |
| `build/` | documento principal y lectura del `.aux` |
| `preprocess/` | LaTeX a LaTeX; `pipeline.py` fija el orden de las reglas |
| `filters/` | filtros Lua de pandoc; `tex2md.lua` fija el orden de las pasadas |
| `postprocess/` | limpieza del Markdown |
| `diagnostics.py`, `pandoc_recovery.py` | diagnósticos tipados y reintento sin lo que pandoc no puede leer |
| `library/` | encabezado YAML, índice, `audit` y vínculos con MarginNote |

Para añadir una regla de preprocesado:

1. Escribe una función `texto -> texto` en el módulo temático (`preprocess/floats.py`, `tables.py`, `macros.py`...).
2. Regístrala en `RULES` de `preprocess/pipeline.py`, en su sitio (el orden importa), con un nombre.
3. Añade una prueba mínima en `tests/unit/` (LaTeX de entrada y salida esperada) y, si corrige un artículo real, un caso en `tests/golden/`.

Desde la raíz del repositorio `import tex2md` falla, porque el envoltorio `tex2md.py` tapa al paquete: ejecuta Python desde otra carpeta o usa pytest.

## Reportar una falla

Si una conversión falla o sale mal, abre un issue con el id de arXiv (con versión), el informe JSON y el fragmento del PDF y del `.md` que difieren.
