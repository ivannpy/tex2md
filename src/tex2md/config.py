"""Version, external-tool locations and the data directories used by every module.

The cache and the model weights live outside the source tree and can be moved with
environment variables:

    TEX2MD_CACHE    downloads, builds and arXiv metadata   (default ~/.cache/tex2md)
    TEX2MD_MODELS   marker model weights                   (default ~/.local/share/tex2md/models)
    TEX2MD_MARKER   isolated uv project that runs marker   (default <repo>/marker)
    TEX2MD_MN_ROOT  MarginNote Documents folder (iCloud)   (default the standard iCloud path)
"""
from __future__ import annotations

import os

VERSION = "0.1"
UA = {"User-Agent": "tex2md/" + VERSION}
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERS_DIR = os.path.join(PACKAGE_DIR, "filters")
FILTER = os.path.join(FILTERS_DIR, "filter.lua")
UNWRAP_FILTER = os.path.join(FILTERS_DIR, "unwrap.lua")


def _path(var: str, default: str) -> str:
    return os.path.abspath(os.path.expanduser(os.environ.get(var, default)))


CACHE = _path("TEX2MD_CACHE", "~/.cache/tex2md")
MODELS = _path("TEX2MD_MODELS", "~/.local/share/tex2md/models")
# <repo>/src/tex2md/config.py -> <repo>/marker
MARKER = _path("TEX2MD_MARKER", os.path.join(os.path.dirname(os.path.dirname(PACKAGE_DIR)), "marker"))
MN_ROOT = _path("TEX2MD_MN_ROOT", "~/Library/Mobile Documents/iCloud~QReader~MarginStudy~easy/Documents")
