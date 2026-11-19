# tex2md

Conversor de artículos científicos a Markdown: fuente LaTeX de arXiv (`arxiv`), XML JATS (`jats`) y PDF (`pdf`, con marker). Qué hace y cómo: `README.md`. Tubería, fallas conocidas y trabajo pendiente: `ROADMAP.md`. Historial: `CHANGELOG.md`.

## Antes de tocar el código

- `make check` (ruff, mypy y pruebas rápidas) y `make regress` (reconvierte los artículos golden desde el caché y compara byte a byte). Los dos deben pasar antes de un commit.
- Ejecutar con `uv run` (o `make`). Desde la raíz del repositorio `import tex2md` falla porque el envoltorio `tex2md.py` tapa al paquete: usar pytest o ejecutar Python desde otra carpeta.
- No hay comando global instalado: `tex2md.py`, `papers.py` y `doc2md.py` son envoltorios del mismo CLI (`src/tex2md/cli.py`), con subcomandos `arxiv|jats|pdf|frontmatter|add-mn|check|index|audit|fixmath|doctor`. `doctor` comprueba pandoc, TeX y uv.

## Arquitectura

Estructura de `src/tex2md/` y cómo añadir una regla de preprocesado: sección «Desarrollo» de `README.md`. En resumen: una regla es una función `texto -> texto` en su módulo de `preprocess/`, registrada en `RULES` de `preprocess/pipeline.py` (el orden importa), con una prueba en `tests/unit/` y, si corrige un artículo real, un caso en `tests/golden/` (ver `tests/golden/README.md`). Los filtros Lua de pandoc están en `filters/` (`tex2md.lua` fija el orden de las pasadas).

## Entorno y datos

- Caché de fuentes de arXiv en `~/.cache/tex2md/` y modelos de marker en `~/.local/share/tex2md/models/`, fuera del código (`TEX2MD_CACHE`, `TEX2MD_MODELS`). `make regress` necesita el caché; `python3 tests/regress.py tests/golden` lo llena descargando las fuentes.
- Los golden se generaron con pandoc 3.14: otra versión puede cambiar detalles de formato.
- **marker está fijado en 1.10.2**, en su propio entorno uv (`marker/`). La 2.x necesita un servidor externo de llama.cpp (`llama-server`), que no se instaló para no meter binarios fuera de PyPI.

## Diagnósticos

El informe JSON de `arxiv` trae una lista de diagnósticos tipados (`diagnostics.py`) y `audit` los muestra por nivel: `error` (`unbalanced-math`, `unresolved-reference`, `dropped-environment`), `warning` (`unknown-math-macro`, citas sin resolver, HTML residual, definiciones omitidas, `lost-text`) e `info`. Cuando pandoc no puede leer un entorno, `pandoc_recovery.py` lo quita, reintenta y deja una nota visible. Un diagnóstico nuevo se añade ahí y a la tabla de `README.md` («Informe y diagnósticos»).

**Trabajo abierto:** `HALLAZGOS_2026-11-18.md` (fallas encontradas al reconvertir 158 artículos de la biblioteca, con causa, reproducción y orden sugerido de corrección).

Fallas conocidas y su estado (macros que pandoc no puede expandir: `\foreach`/`\csname`, `\if…\fi`, `\NewDocumentCommand` con `e_`; falsos positivos de `audit`): `ROADMAP.md` §3. Al corregir una, actualizar `ROADMAP.md` y `CHANGELOG.md`.
