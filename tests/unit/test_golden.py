"""Replay the golden conversions from the local cache and compare byte for byte (slow)."""
import importlib.util
import os
import re
from pathlib import Path

import pytest

from tex2md.config import CACHE

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "golden"

spec = importlib.util.spec_from_file_location("regress", HERE.parent / "regress.py")
regress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regress)


def cached(md: Path) -> bool:
    fm = md.read_text().split("---\n", 2)[1]
    aid = re.search(r'^arxiv: "([^"]+)"', fm, re.M).group(1)
    ver = re.search(r'^arxiv_version: "([^"]+)"', fm, re.M).group(1)
    return os.path.isdir(os.path.join(CACHE, (aid + ver).replace("/", "_")))


@pytest.mark.slow
@pytest.mark.parametrize("md", sorted(p for p in GOLDEN.glob("*.md") if p.name != "README.md"), ids=lambda p: p.stem)
def test_golden(md, tmp_path):
    if not cached(md):
        pytest.skip("source not in the local cache (TEX2MD_CACHE)")
    name, status, detail = regress.check(str(md), str(tmp_path))
    assert status == "same", detail
