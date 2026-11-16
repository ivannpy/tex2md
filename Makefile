.DEFAULT_GOAL := help
PY ?= uv run

help:            ## list the targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

install:         ## install the `tex2md` command (editable) with uv
	uv tool install --editable .

dev:             ## create .venv with the development tools
	uv sync --extra dev

doctor:          ## check pandoc, TeX, uv and the data directories
	$(PY) tex2md doctor

lint:            ## ruff
	$(PY) ruff check src tests

format:          ## ruff format (does not touch files unless you run it)
	$(PY) ruff format src tests

typecheck:       ## mypy
	$(PY) mypy

test:            ## unit tests (fast)
	$(PY) pytest -m "not slow"

regress:         ## replay the golden conversions from the local cache (needs the cache)
	$(PY) pytest -m slow

check: lint typecheck test   ## everything that must pass before a commit

.PHONY: help install dev doctor lint format typecheck test regress check
