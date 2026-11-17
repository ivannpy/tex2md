# tex2md: qué hace, cómo, dónde falla y qué sigue

Documento vivo. Se actualiza cuando cambia el comportamiento del conversor. Para instrucciones de uso y
arquitectura de código ver `README.md`; para el historial de cambios, `CHANGELOG.md`.

Estado a 2026-11-17 · 71 reglas de preprocesado · 9 módulos Lua · 271 pruebas (22 de ellas reconvierten
artículos reales y comparan byte a byte) · 124 artículos de LaTeX, 7 de PDF y 1 de JATS en la biblioteca · además,
65 artículos de arXiv que no están en la biblioteca (2012–2025) convierten sin error y con los mismos números de
figuras y tablas que su PDF.

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
          ─► aplanar con latexpand ─► preprocesar (71 reglas) ─► pandoc + filtros Lua ─► postprocesar ─► .md
```

1. **Descarga** (`sources/arxiv.py`): metadatos de la API de arXiv (con 3 s como mínimo entre dos peticiones,
   también entre ejecuciones distintas, y respetando `Retry-After`) y e-print con reintentos; extracción de `tar`
   con `filter="data"` en una carpeta temporal que se renombra al terminar. Todo se guarda en
   `~/.cache/tex2md/<id>/`.
2. **Etiquetado de ecuaciones** (`preprocess/equations.py`): antes de compilar, cada ecuación numerada sin
   `\label` recibe uno propio, para que el `.aux` registre su número. Se hace una vez por compilación guardada.
3. **Compilación** (`latexmk`, una pasada tolerante a errores y con `-no-shell-escape`, porque el LaTeX es de
   terceros): produce el `.aux` (números reales de secciones,
   figuras, tablas y ecuaciones) y el `.bbl` (bibliografía).
4. **Aplanado** con `latexpand`: un solo `.tex` con `\input`/`\include` resueltos.
5. **Preprocesado** (`preprocess/pipeline.py`, lista ordenada de 71 reglas con nombre, cada una una función
   `texto → texto` probada por separado). Resuelve `\ref`/`\cite` a los números del PDF, normaliza entornos que
   pandoc pierde (subfloats, minipages, `\captionof`, `\captionsetup{type=…}`, tablas sin `tabular`…), expande
   macros que pandoc no entiende (`.sty` locales, `\DeclarePairedDelimiter`, macros con argumento delimitado,
   glosarios), reescribe algoritmos a líneas y limpia el preámbulo. Un grupo de reglas existe para un riesgo
   concreto de pandoc: **un comando que no conoce (porque viene de un `.sty`) se traga todos sus argumentos entre
   llaves sin avisar**. Por eso `textloss.py` desenvuelve cajas y comandos de texto (`\fbox`, `\centerline`,
   `\cellcolor`…), `symbols.py` convierte `\ding`/`\cmark` a Unicode, `comments.py` incorpora los `\input` y
   listados que latexpand deja, `environments.py` quita código de dibujo y adapta `tblr`/`multicols`/`restatable`,
   y `braces.py` cierra una llave sin cerrar que se tragaría el resto del artículo.
6. **pandoc** (`-f latex -t markdown`) con la tubería Lua `filters/tex2md.lua`: numera encabezados, figuras,
   tablas y ecuaciones con los datos del `.aux`; desenvuelve `Div`/`Span`; convierte tablas con colspan/rowspan
   en tablas de tuberías; conserva el contenido de figuras que son texto (prompts, algoritmos). Si pandoc no
   puede leer una construcción (`pandoc_recovery.py`), localiza qué la contiene y lo quita: un **entorno**
   (se sustituye por una nota visible, porque su contenido falta), una **definición** (`\newcolumntype`, una macro
   que se expande a sí misma) o una línea del **preámbulo**; lo registra como diagnóstico y reintenta (hasta 5
   veces) en lugar de abortar.
7. **Postprocesado** (`postprocess/markdown.py`): matemáticas aisladas, `\text` restaurado, MathML → TeX,
   comentarios `%` y fórmulas pegadas, restos de contadores LaTeX.
8. **Informe de calidad** (JSON en la salida y en el caché): tablas en el tex frente al md, referencias y
   citas sin resolver, y una lista de **diagnósticos tipados** (`diagnostics.py`): código estable, gravedad
   (`error` = contenido erróneo o ausente, `warning`, `info`), cantidad y ejemplos. Cubren comandos que
   KaTeX/MathJax no conocen en las ecuaciones, avisos de pandoc, `??`, citas sin resolver, restos de HTML,
   flotantes sin número, ecuaciones sin `\tag`, entornos y definiciones omitidos por la recuperación, y tres que
   buscan pérdidas silenciosas: fórmulas con `$` que no se emparejan (`unbalanced-math`, error), texto del
   LaTeX que no está en el Markdown (`lost-text`, aviso) y llaves reparadas (`unbalanced-braces`).

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
- `tests/regress.py`: reconvierte toda la biblioteca y compara contra una línea base (`--out DIR` conserva los
  archivos reconvertidos, que es como se regenera la biblioteca sin tocarla hasta revisar el resultado).
- `audit [--level error|warning|info]`: compara cada `.md` con los números de su PDF compilado (leyendas
  ausentes o inexistentes, también las que están dentro de una lista o una cita) y muestra los diagnósticos
  tipados de su cuerpo; sale con estado 1 si hay diferencias de flotantes o diagnósticos de gravedad `error`.
- **Método de la revisión del 2026-11-17** para encontrar lo que pandoc hace sin avisar: convertir artículos que
  nadie ha mirado (65, de 2012 a 2025), pasar `audit`, y buscar tres señales: `$` sin pareja por párrafo,
  números decimales del PDF que no están en el `.md`, y palabras de la fuente que no llegan al Markdown (el
  diagnóstico `lost-text`). Cada hallazgo se convierte en una prueba mínima antes de corregirlo.
- Comprobaciones usadas en las revisiones por muestreo que siguen fuera del paquete: texto del PDF frente al
  `.md`, títulos, `\tag` frente a «(N)» del PDF.

### 2.4 Qué NO pasa por esta tubería
Los `.md` de **PDF** (marker) y de **JATS** se generan con otros caminos y **no reciben** las mejoras de
preprocesado LaTeX. Los 8 existentes no se han reconvertido con la versión actual.

---

## 3. Fallas conocidas

Ordenadas por impacto. «Frecuencia» es sobre los 124 artículos de LaTeX.

| # | Falla | Frecuencia / ejemplo | Efecto |
|---|---|---|---|
| F1 | **Si pandoc no puede leer una construcción, se omite**: un entorno (con nota visible y diagnóstico `dropped-environment`), una definición o una línea del preámbulo (`dropped-definition`); una llave sin cerrar se repara (`unbalanced-braces`). La conversión aún aborta si el error persiste tras 5 reintentos o está fuera de todo lo anterior | no ocurre en la biblioteca ni en los 65 artículos nuevos tras las correcciones (antes abortaban varios: llave sin cerrar, `\centerline{…\par…}`, `\newcolumntype` con `#1`, macros que se expanden a sí mismas, `\global\setlength`) | hueco marcado en el `.md`, o ningún `.md` |
| F2 | **Comandos sin expandir o inválidos en ecuaciones** (diagnóstico `unknown-math-macro`): macros generadas por programa (`\foreach` + `\csname`), definidas en paquetes sin fuente o con `\NewDocumentCommand` de especificación `e_`; y comandos de texto que KaTeX no admite en matemáticas | 454 usos en 14 archivos (antes 596 en ~35): `\if`/`\fi` (216, sahoo2022), `\vz`/`\vx`/`\gB`… (97, busbridge2025), `\d` (31, liu2022flow), `\filledstar` (30, sahoo2025duo), `\grad` (24, hubotter2026), `\nm` (23, zhang2026). `\textsc`, `\nicefrac` y `\includegraphics` dentro de fórmulas ya se reescriben | la ecuación no se renderiza en un visor de Markdown |
| F3 | **Ecuaciones sin `\tag`** (diagnóstico `equation-without-tag`, informativo) o con `(??)` por colisión de `\label` | 161 en 15 archivos: 119 son de `campbell2022`, que usa `autonum` (solo numera lo que se cita: no es una falla); el resto, repartido (`uno2026` 16, `shi2024md4` 8…), no se ha revisado una a una. Colisiones de `\label` en `2406.04329` y `2503.09573` | falta el número de alguna ecuación |
| F4 | **Tablas con bloques de código dentro de las celdas** pueden quedar como HTML | 1 tabla en 1 archivo (`nie2025llada`, celdas con color por token); antes 14 en 8 | HTML en el Markdown (con las fórmulas ya en TeX) |
| F5 | **Líneas `**Figura.**`/`**Tabla.**` sin número** | 21 en 7 archivos (`pan2025demasking` 10, `hu2026elf` 5…): paneles de una figura o tablas sin `\caption` propio en el LaTeX | no es un error de conversión, pero no tienen número que citar |
| F13 | **Citas sin resolver** (`[@clave]`, diagnóstico `unresolved-citation`) | 87 en `sahoo2022backpropagat` (bibliografía en un formato que no se empareja con las claves) | la cita aparece como clave de pandoc, no como autor o número |
| F14 | **Referencias `??` reales** (diagnóstico `unresolved-reference`) | 9 en 4 archivos: `arriola2025`, `sahoo2025duo`, `su2026opdlm`, `mirzadeh2019takd`. La etiqueta no está en el `.aux`: en `su2026opdlm` está en un `subfigure` cuyo `\caption` se comentó, y el PDF también muestra `??` | «(??)» en el texto en vez del número, igual que en el PDF |
| F6 | **Fórmulas con trucos de TeX sin equivalente** (`\if…\fi` con argumento opcional, `\foreach`) | `sahoo2022` (`\ifthenelse` ya se resuelve) | fórmula truncada o con un comando inválido |
| F7 | **`audit` da un falso positivo** | `pan2025demasking`: su `.aux` no registra 9 de sus tablas (sus números salen de las etiquetas y no están comprobados contra el PDF) | un aviso permanente en `audit` |
| F15 | **Texto de la fuente que no llega al `.md`** (diagnóstico `lost-text`): red de seguridad, no prueba | 996 palabras en 19 archivos; casi todo es ruido (opciones de listados y de macros, cabeceras de tablas reordenadas, notas del autor dentro de macros); no se ha mirado uno por uno | posible texto perdido que habría que revisar |
| F16 | **Fuentes que no se pueden convertir**: TeX plano y LaTeX 2.09 (sin `\documentclass`), envoltorios de PDF (`\includepdf`), envíos que solo tienen PDF | 2 envoltorios de PDF entre los artículos probados | error claro; para esos hay que usar `pdf` (OCR) |
| F8 | **Dependen de herramientas externas con versiones concretas** | pandoc ≥ 3 (probado con 3.14), `latexmk`, `latexpand`; `uv` solo para `pdf` | la conversión falla si faltan; `doctor` lo avisa |
| F9 | **marker (PDF) comete errores de OCR** en ecuaciones y tablas | los 7 `.md` de PDF | no son fiables en cifras; el PDF manda |
| F10 | ~~Compilar LaTeX de terceros sin `-no-shell-escape`~~ **resuelta**: `latexmk` se llama con `-no-shell-escape` (probado: `\write18` queda `disabled`) y las 124 conversiones siguen idénticas | — | — |
| F11 | **Sin caché de versiones del conversor en el `.md`** | `VERSION` sigue en `0.1`; el encabezado no indica con qué versión se generó | no se distingue un `.md` viejo de uno nuevo |
| F12 | **Parte de las verificaciones sigue siendo manual** | los diagnósticos ya viven en el paquete (`audit`, y desde esta revisión `unbalanced-math` y `lost-text`), pero los scripts de muestreo (cobertura de texto y de números frente al PDF, `\tag` frente a «(N)») siguen fuera del repositorio | esas comprobaciones dependen de que alguien las vuelva a escribir |
| F17 | ~~Los `.md` de la biblioteca no se habían regenerado con esta versión~~ **resuelta** el 2026-11-17: los 124 de LaTeX se reconvirtieron (la copia anterior se guardó fuera de la biblioteca) y `tests/regress.py` los reproduce idénticos | — | — |

**Límites de diseño** (no son fallas a corregir): las imágenes no se extraen; la fidelidad tipográfica
(posición de figuras, columnas) no se conserva; un `.md` de un artículo con macros muy exóticas puede requerir
una regla nueva.

---

## 4. Trabajo futuro

### Hecho en la revisión del 2026-11-17
1. **Revisión de fallas y errores lógicos** con la biblioteca y 65 artículos nuevos de arXiv. Lo encontrado se
   resume en `CHANGELOG.md`; en lo esencial: texto que pandoc perdía sin avisar (comandos desconocidos que se
   tragan sus argumentos, cajas, colores de celda, listados, comentarios `%`), fórmulas con `$` sin cerrar,
   leyendas y números de figuras y tablas que no coincidían con el PDF, y fallas de conversión completas.
2. **Tres diagnósticos nuevos que buscan pérdidas silenciosas** (`unbalanced-math`, `lost-text`,
   `unbalanced-braces`) y recuperación ampliada (definiciones, macros en bucle, líneas del preámbulo).
3. **Un error de empaquetado**: `.gitignore` ocultaba el paquete `src/tex2md/build/` y nunca se había
   versionado, de modo que un clon limpio no arrancaba. Los patrones están anclados a la raíz; **falta añadir
   esos archivos a git** (`git add src/tex2md/build/` y los módulos nuevos de `preprocess/`).
4. **Ritmo de la API de arXiv** (3 s entre peticiones, `Retry-After`, sin reintentos en errores 4xx) y errores de
   red como un mensaje de una línea.
5. 271 pruebas (antes 138); `ruff` y `mypy` sin avisos.
6. **Biblioteca regenerada** (F17): los 124 `.md` de LaTeX, con el mismo encabezado YAML (vínculos de MarginNote
   incluidos). Frente a los anteriores: `$` sin pareja 40 → 0 párrafos, números decimales del PDF ausentes
   248 → 53, archivos con diagnósticos `error` 15 → 4 (los `??` reales). `tests/regress.py --out DIR` conserva los
   archivos reconvertidos y omite `INDEX.md` y los de PDF/JATS; es el flujo descrito en el `CLAUDE.md` de la
   biblioteca.

### Hecho el 2026-11-16
1. **Diagnósticos tipados** en el informe de conversión y en `audit` (`diagnostics.py`). Hicieron visibles F2,
   F13 y F14.
2. **Recuperación cuando pandoc falla** (`pandoc_recovery.py`): entorno culpable → nota visible → reintento
   (máximo 5), con diagnóstico `dropped-environment`. Probado con pandoc real.
3. **Compilación sin shell** (`-no-shell-escape`). Reconstruidas y reconvertidas las 124 compilaciones: salida
   idéntica byte a byte.

### Siguiente, derivado de los diagnósticos
- **Revisar a mano los `lost-text`** de mayor tamaño (`hu2026elf`, `madeira2026trajectory`, `han2022ssdlm`) para
  afinar el diagnóstico (F15), y ver las ecuaciones sin `\tag` que no son de `campbell2022` (F3).
- **Investigar las 87 citas de `sahoo2022`** (F13) y añadir una lista de diagnósticos aceptados a `audit`
  (F7).

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
