#!/usr/bin/env python3
"""Compatibility wrapper: ``python3 doc2md.py jats|pdf ...`` == ``tex2md jats|pdf ...``."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from tex2md.cli import legacy  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(legacy(None))
