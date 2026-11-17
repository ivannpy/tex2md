# tex2md: qué hace, cómo, dónde falla y qué sigue

Documento vivo. Se actualiza cuando cambia el comportamiento del conversor. Para instrucciones de uso y
arquitectura de código ver `README.md`; para el historial de cambios, `CHANGELOG.md`.

Estado a 2026-11-16 · 45 reglas de preprocesado · 9 módulos Lua · 118 pruebas (22 de ellas reconvierten
artículos reales y comparan byte a byte) · 124 artículos de LaTeX, 7 de PDF y 1 de JATS en la biblioteca.

---

## 1. Qué hace

Convierte artículos científicos a Markdown para la biblioteca de la tesis (`MindBooksBreakdown/md`), de modo
que se puedan leer, buscar y citar con exactitud (cifras, ecuaciones, tablas y numeración como en el PDF).

| Fuente | Subcomando | Resultado |
|---|---|---|
| Fuente LaTeX de arXiv | `arxiv` | `.md` con secciones, ecuaciones (`$…$`, `\tag{N}`), tablas, leyendas de figuras y tablas, algoritmos y bibliografía, numerados como el PDF compilado |
| XML JATS (revistas abiertas, PubMed Central) | `jats` | `.md` vía pandoc, metadatos de Crossref |
| Solo PDF | `pdf` | `.md` por OCR con marker (más propenso a errores) |
| Biblioteca | `frontmatter`, `add-mn`, `check`, `index`, `audit`, `fixmath`, `doctor` | encabezados YAML con el vínculo a MarginNote, índice, comprobaciones |

Cada `.md` lleva un encabezado YAML (título, autores, año, arXiv, versión, vínculo `bookmd5`/ruta de
MarginNote, fuente de la conversión, fecha). Las figuras se reducen a su leyenda.

**Qué garantiza.** Los números de secciones, figuras, tablas y ecuaciones salen de la compilación real del
artículo, no de un conteo propio. Todo texto que viene del LaTeX se conserva, salvo lo listado en la sección 3.

**Qué no pretende.** No reproduce el diseño de página ni las imágenes. No es un motor de TeX: delega en pandoc.
No convierte documentos Word, HTML genérico ni PDF escaneados con calidad de LaTeX.

---

## 2. Cómo lo hace

### 2.1 Tubería de `arxiv`

```
id de arXiv ─► descargar e-print ─► (etiquetar ecuaciones) ─► compilar con latexmk ─► .aux / .bbl
          ─► aplanar con latexpand ─► preprocesar (45 reglas) ─► pandoc + filtros Lua ─► postprocesar ─► .md
```

1. **Descarga** (`sources/arxiv.py`): metadatos de la API de arXiv y e-print con reintentos; extracción de `tar`
   con `filter="data"`. Todo se guarda en `~/.cache/tex2md/<id>/`.
2. **Etiquetado de ecuaciones** (`preprocess/equations.py`): antes de compilar, cada ecuación numerada sin
   `\label` recibe uno propio, para que el `.aux` registre su número. Se hace una vez por compilación guardada.
3. **Compilación** (`latexmk`, una pasada tolerante a errores): produce el `.aux` (números reales de secciones,
   figuras, tablas y ecuaciones) y el `.bbl` (bibliografía).
4. **Aplanado** con `latexpand`: un solo `.tex` con `\input`/`\include` resueltos.
5. **Preprocesado** (`preprocess/pipeline.py`, lista ordenada de 45 reglas con nombre, cada una una función
   `texto → texto` probada por separado). Resuelve `\ref`/`\cite` a los números del PDF, normaliza entornos que
   pandoc pierde (subfloats, minipages, `\captionof`, tablas sin `tabular`…), expande macros que pandoc no
   entiende (`.sty` locales, `\DeclarePairedDelimiter`, macros con argumento delimitado), reescribe algoritmos a
   líneas y limpia el preámbulo.
6. **pandoc** (`-f latex -t markdown`) con la tubería Lua `filters/tex2md.lua`: numera encabezados, figuras,
   tablas y ecuaciones con los datos del `.aux`; desenvuelve `Div`/`Span`; convierte tablas con colspan/rowspan
   en tablas de tuberías; conserva el contenido de figuras que son texto (prompts, algoritmos).
7. **Postprocesado** (`postprocess/markdown.py`): matemáticas aisladas, `\text` restaurado, MathML → TeX,
   comentarios `%` y fórmulas pegadas, restos de contadores LaTeX.
8. **Informe de calidad** (JSON en la salida y en el caché): tablas en el tex frente al md, referencias y
   citas sin resolver, avisos.

### 2.2 Numeración de figuras, tablas y ecuaciones
pandoc no conoce la numeración del PDF. tex2md lee del `.aux`: (a) `\newlabel` → secciones, ecuaciones y
etiquetas; (b) cada `\caption` en orden → número real de cada figura y tabla. El filtro Lua empareja cada
leyenda con su entrada por el comienzo del texto (prefijo más largo primero; subcadena si la leyenda empieza
con una macro; la etiqueta propia tiene prioridad sobre el emparejamiento difuso). El `.aux` lista los
flotantes en orden de **colocación**, no de documento, así que cada entrada se usa una sola vez y se busca
sin depender del orden.

### 2.3 Verificación
- `tests/unit/`: reglas, filtro Lua (con pandoc real), CLI, posproceso y 22 conversiones reales («golden»)
  que se repiten desde el caché y se comparan byte a byte.
- `tests/regress.py`: reconvierte toda la biblioteca y compara contra una línea base.
- `audit`: compara cada `.md` con los números de su PDF compilado (leyendas ausentes o inexistentes).
- Comprobaciones usadas en las revisiones por muestreo (no son parte del paquete): texto del PDF frente al
  `.md`, números de tablas, títulos, `\tag` frente a «(N)» del PDF.

### 2.4 Qué NO pasa por esta tubería
Los `.md` de **PDF** (marker) y de **JATS** se generan con otros caminos y **no reciben** las mejoras de
preprocesado LaTeX. Los 8 existentes no se han reconvertido con la versión actual.

---

## 3. Fallas conocidas

Ordenadas por impacto. «Frecuencia» es sobre los 124 artículos de LaTeX.

| # | Falla | Frecuencia / ejemplo | Efecto |
|---|---|---|---|
| F1 | **Un error de pandoc aborta la conversión entera** | raro (no ocurre hoy en la biblioteca, pero ocurrió varias veces durante el desarrollo) | no se produce `.md` |
| F2 | **Macros del autor sin expandir en ecuaciones** cuando se generan por programa (`\foreach` + `\csname`), se definen en paquetes sin fuente o con `\NewDocumentCommand` de especificación `e_` | `\vz`, `\vx`, `\gB` (busbridge2025), `\grad` (hubotter2026), `\nm` (zhang2026), `\filledstar` (sahoo2025) | la ecuación contiene un comando que ningún motor de Markdown conoce |
| F3 | **Ecuaciones sin `\tag`** o con `(??)` por colisión de `\label` | 4 compilaciones con una colisión (`2406.04329`, `2503.09573`, `2506.10892`, `1503.03585`): código comentado y `\\ \nonumber` de estilo antiguo; hasta 3 números sin etiqueta en `alcaraz2022`, `gong2022`, `bonnaire2025` | falta el número de alguna ecuación; 12 referencias `(??)` en 5 archivos |
| F4 | **Tablas con bloques de código dentro de las celdas** siguen como HTML | 14 tablas en 8 archivos (`liu2022flow`, `han2022ssdlm`, `iyer2024quality`…) | HTML en el Markdown (con las fórmulas ya en TeX) |
| F5 | **Líneas `**Figura.**`/`**Tabla.**` sin número** | 40 en 19 archivos: paneles de una figura o tablas sin `\caption` propio en el LaTeX | no es un error de conversión, pero no tienen número que citar |
| F6 | **Fórmulas con trucos de TeX sin equivalente** (`\ifthenelse`, `\if…\fi`, `\textsc` en matemáticas) | `zheng2023`, `sahoo2022` | fórmula truncada o con un comando inválido |
| F7 | **`audit` da un falso positivo** | `pan2025demasking`: su `.aux` no registra 9 de sus tablas (sus números salen de las etiquetas y no están comprobados contra el PDF) | un aviso permanente en `audit` |
| F8 | **Dependen de herramientas externas con versiones concretas** | pandoc ≥ 3 (probado con 3.14), `latexmk`, `latexpand`; `uv` solo para `pdf` | la conversión falla si faltan; `doctor` lo avisa |
| F9 | **marker (PDF) comete errores de OCR** en ecuaciones y tablas | los 7 `.md` de PDF | no son fiables en cifras; el PDF manda |
| F10 | **Compilar LaTeX de terceros sin `-no-shell-escape`** | TeX Live aplica `shell_escape=p` (lista blanca de comandos), pero no se desactiva | superficie de ataque pequeña pero evitable |
| F11 | **Sin caché de versiones del conversor en el `.md`** | `VERSION` sigue en `0.1`; el encabezado no indica con qué versión se generó | no se distingue un `.md` viejo de uno nuevo |
| F12 | **Verificaciones de calidad solo manuales** | los scripts de muestreo (cobertura frente al PDF, `\tag` frente a «(N)») viven fuera del repositorio | la calidad depende de que alguien los vuelva a escribir |

**Límites de diseño** (no son fallas a corregir): las imágenes no se extraen; la fidelidad tipográfica
(posición de figuras, columnas) no se conserva; un `.md` de un artículo con macros muy exóticas puede requerir
una regla nueva.

---

## 4. Trabajo futuro

### Inmediato (en curso en esta iteración)
Estado: **pendiente**; al terminarse se marcará aquí y en `CHANGELOG.md`.

1. **Diagnósticos tipados en el informe.** En lugar de un solo contador, una lista con código, gravedad,
   cantidad y ejemplos: macros sin expandir en ecuaciones, avisos de pandoc (hoy se descartan), restos de
   HTML, `??`, flotantes sin número y ecuaciones sin `\tag`. `audit` los mostrará y fallará si hay errores.
   *Cierra parcialmente F2, F3, F6, F12.*
2. **Reintento automático cuando pandoc falla.** pandoc indica línea y columna del error: se localiza el
   entorno más interno que lo contiene, se sustituye por un marcador visible, se registra como diagnóstico y se
   reintenta (hasta 5 veces). Un artículo con una construcción imposible dejará de bloquear la conversión.
   *Cierra F1.*
3. **Compilar con `-no-shell-escape`.** Una línea y una prueba; se comprobará sobre toda la biblioteca que no
   cambia ningún resultado. *Cierra F10.*

### Siguiente
4. **Contraste con el HTML de arXiv** (`arxiv.org/html/<id>` o ar5iv) como oráculo: por HTTP, sin instalar
   LaTeXML. Comparar secciones, ecuaciones y leyendas con nuestro `.md`. Útil donde `audit` no puede apoyarse
   en el `.aux` y como referencia de cómo debería verse una macro que pandoc no expande (F2, F3).
5. **Procedencia en el encabezado YAML**: versión del conversor y commit, para saber qué `.md` están
   desactualizados (F11). Subir `VERSION` y registrar un «formato de salida».
6. **Verificación dentro del repositorio**: llevar al paquete (como herramienta de desarrollo, no
   dependencia) las comprobaciones de cobertura de texto frente al PDF y de `\tag` frente a «(N)» (F12).
7. **Pruebas de invariantes sobre las reglas**: aplicar una regla dos veces debe dar lo mismo, y ninguna debe
   perder texto. Habría detectado la mayoría de las regresiones del desarrollo.
8. **Expansión de macros generadas por programa** (`\foreach`, `\csname`), quizá obteniendo la expansión de la
   propia compilación de TeX o del HTML de arXiv (F2). Requiere un experimento previo.

### Más adelante
9. **Archivo auxiliar estructurado** junto a cada `.md` (secciones, ecuaciones con su `\tag`, flotantes con su
   leyenda y página) para alimentar los desgloses de MarginNote sin buscar a mano en el texto.
10. **Reconvertir los `.md` de JATS y de PDF** con la versión actual (necesita el XML o DOI originales; los PDF
    cuestan ~1 min por página).
11. **Cobertura de pruebas** (`pytest-cov` en el entorno de desarrollo) y activar `check_untyped_defs` en mypy
    módulo a módulo.
12. **Tablas con código** (F4): una representación Markdown alternativa para celdas con bloques de código.

### Descartado (con motivo)
- **Reescribir sobre una representación intermedia propia**: el AST de pandoc más los filtros Lua ya cumple
  esa función; no compensa el costo.
- **Instalar LaTeXML, Tralics o GROBID**: pesados para esta biblioteca, y LaTeXML tampoco convierte todo.
- **Instalar un comando global**: el proyecto se usa con los envoltorios `tex2md.py`, `papers.py` y
  `doc2md.py` para mantener el sistema limpio.
- **Cambiar marker por MinerU o Docling** para los PDF: sin evidencia sólida de que compense.
