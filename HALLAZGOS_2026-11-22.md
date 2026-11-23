# Hallazgos de la reconversión de la biblioteca (2026-11-22)

Contexto: entre el 2026-11-19 y el 2026-11-22 se añadieron ~56 PDF a `~/Documents/MindBooksBreakdown/pdf/`
(todos con fuente LaTeX en arXiv; sus `.md` se habían hecho con `tex2md arxiv` el 21 y el 22). El usuario
reportó `.md` que no se renderizan bien (sospecha: falta de saltos de línea alrededor de `$`/`$$`), que pierden
información o que no se convierten. Se revisaron los 59 `.md` más recientes con `audit` y con un escáner de
delimitadores de matemáticas, se corrigió el conversor y se reconvirtió la biblioteca entera desde el caché.

Copia previa de la biblioteca: `~/Documents/MindBooksBreakdown_md_backup_2026-11-22/`.

## 1. Corregido (detalle en `CHANGELOG.md`, entrada del 2026-11-22)

| Síntoma | Artículos | Causa | Corrección |
|---|---|---|---|
| `$$…$$` en medio de una línea: `> *Let … $$\begin{equation}…\end{equation}$$*`, `\end{align}$$ They can…` (no se renderiza en MarginNote/Obsidian) | 403 casos en la biblioteca; 313 en `jiang2026semanticdlm` | pandoc deja la ecuación dentro del párrafo; `isolate_display_math` no tocaba las de citas (cajas de teorema), listas ni cursivas | pasada Lua nueva `displaymath.lua`: parte el párrafo, cierra y reabre la cursiva |
| `> ` dentro de una fórmula en línea (`$\bar p > := …$`) | teoremas en cita | fórmula escrita en dos líneas dentro de una cita | la fórmula en línea pasa a una línea en el filtro |
| `$ x$`, `$x $` | 31 + 18 casos | espacios dentro de los delimitadores | se quitan (filtro y `tidy_math`) |
| `$$$\displaystyle …` y `$` impares (6 `unbalanced-math`) | `zekri2026gdds` | `\resizebox{\linewidth}{!}{$\displaystyle …$}` dentro de `\[ \]` | regla `boxes-around-math` |
| Tabla impresa como LaTeX crudo (`\@llllll@ & …`) | `liu2024ddpd` | `tabu`, tabla guardada en una macro, línea en blanco, `\vspace` y `\\` dentro de una celda, `\pbox` | reglas `tabu-to-tabular`, `expand-float-macros`, `cell-line-breaks`, `\pbox` en `unwrap_cell_macros` |
| Tabla 3 y ecuaciones escaladas perdidas | `wang2026ludi` | `\fitbox` definido en un `.cls` | se lee como `\resizebox` |
| Tablas 1 y 7 perdidas | `dhariwal2021beat` | `\multirow{2}*{X}` | se normaliza a `{*}` |
| Tabla 9 vacía, Tabla 8 dos veces | `utpala2026loopedudlm` | fila de tablas y luego fila de leyendas en minipages | regla `pair-content-with-captions` |
| Figura 1 perdida | `zhu2026wamdiff2`, `yu2026groundanything` | figura dibujada por un `\maketitle` redefinido | regla `teaser-in-maketitle` |
| 47 `??`, sin numeración | `heusel2017fid` | LaTeX no compilaba: falta `tocstyle.sty` (retirado de TeX Live) | se quita el paquete que falta y se recompila |
| **`.md` con solo el resumen** (28 líneas; el informe decía `ok`) | `cheng2026vocalnetmdm` | archivo principal en `latex/` con `\input{sections/…}` relativos a la raíz | se compila desde la raíz; diagnóstico nuevo `compile-failed` |
| Figuras 5 y 6 con el mismo número | `heusel2017fid` | leyendas iguales en sus 40 primeros caracteres y `.aux` en orden de colocación | `use_long_captions` busca también hacia atrás; el `\label` de la leyenda decide |
| 866 comandos sin expandir en ecuaciones | `gourevitch2026udm`, `yang2025anyprocess` | `\newcommandx` (xargs, argumentos opcionales al final) | regla `expand-newcommandx` (quedan 8) |
| `\tag{$$-leaping}`: se desemparejan todos los `$$` siguientes | `gourevitch2026udm` | `\tag` del autor más el `\tag` que añade la etiqueta | no se añade un segundo `\tag` |
| `Pixart-$$`, `$$-gpts` en las referencias | `yang2025mmada`, `prabhudesai2025` | `\(\alpha\)` en el `.bbl` perdía su contenido | `plain_bib` lee `\(…\)` |
| `$^\$$` (dólar como marca de nota) parece abrir `$$` | `schiff2026clock`, `schiff2025guidance` | | `\$` final → `\textdollar` |
| Falsos `unknown-math-macro` | varios | lista propia incompleta | se añaden `\frown`, `\mathrlap`, `\blacktriangleleft`… (KaTeX los conoce) |

Totales de la biblioteca (345 `.md` de LaTeX), antes → después: `$$` a mitad de línea o con texto detrás
631 → ~0 (salvo el caso de §2.4), `$` con espacio junto al delimitador 41 → 0 en fórmulas, `unbalanced-math`
6 → 0, `unresolved-reference` 61 → 13 (todas reales), `unknown-math-macro` 2875 → ~2019, leyendas ausentes
según `audit` 7 archivos → 1. Ningún `.md` pierde encabezados, leyendas ni texto (comparación automática con
la copia previa); el encabezado YAML de todos queda igual salvo `converted:`.

## 2. Fallas que persisten (para trabajar después)

### 2.1 Figura 2 de `heusel2017fid` (entorno `figwindow`)
`\begin{figwindow}[3,r,{\includegraphics…},[Heavy Ball with Friction]{Heavy Ball with Friction, … \label{fig:HBF}}]`
(paquete `picins`/`floatflt`): la leyenda va dentro del argumento opcional del entorno y pandoc no la ve.
Reproducir: `tex2md.py 1706.08500v6 -o /tmp/x.md`; `audit` dice `missing: Figura 2`. Corrección sugerida: regla
que convierta `figwindow` (y `figwindow`/`tabwindow` de picins) en `figure` con su `\caption`.

### 2.2 Las `??` restantes son reales (13 en 8 archivos)
Las 12 de ROADMAP F14 y una nueva en `qian2026ctopd` (`\Cref{tab:ct_signal_coverage}`: la tabla está en
`generated/CT_OPD_SIGNAL_COVERAGE_TABLE.tex`, que solo se incluye `\IfFileExists{…_COMPLETE.tex}`, y ese archivo
no viene en el e-print; el PDF de arXiv también muestra `??`). No hay que corregir nada.

### 2.3 Los `.md` de PDF (marker) no se tocaron
Los 20 `.md` hechos con marker no pasan por pandoc. Tienen detalles menores de matemáticas (`$ \begin{array}…$`
con espacios en tablas de `shannon1948` y `efron2011tweedie`, `$% …$` en una celda de `efron2011tweedie`).
`fixmath` los arreglaría, pero también reescribe muchas ecuaciones (las parte en varias líneas) en archivos ya
anotados en MarginNote, así que no se aplicó. Si se quiere, hacer un modo de `fixmarker` que solo quite los
espacios dentro de `$…$` (ya lo hace `tidy_math`) sin el resto de `fix_body`.

### 2.4 Restos menores
- `gourevitch2026udm`: 3 `\ifthenelse{\equal…}` sin resolver tras expandir `\newcommandx` (anidados con un
  argumento que lleva llaves); `\updelta` (paquete `upgreek`, KaTeX no lo tiene).
- `unknown-math-macro` sigue en ~2000 usos (F2 del ROADMAP; sobre todo `sahoo2022` y `busbridge2025`).
- `\NewDocumentCommand`/`\newcommandx` con argumentos opcionales **en medio** se dejan como están.

### 2.5 Sin comprobar: texto pegado a `$` (`P$\times$UniD`, `Time steps $t_i$`)
Hay ~6000 fórmulas en línea pegadas a una palabra (`PPL$\downarrow$`), tal como las escribe el autor. Pandoc,
Obsidian y markdown-it-katex las leen bien; no pude comprobar el visor de MarginNote. Si ahí no se renderizan,
la corrección sería separar con un espacio en `tidy_math` (cambia el texto de casi todos los `.md`).
