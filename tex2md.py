#!/usr/bin/env python3
"""Compatibility wrapper: ``python3 tex2md.py <arxiv-id> -o out.md`` == ``tex2md arxiv <arxiv-id> -o out.md``."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from tex2md.cli import legacy  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(legacy("arxiv"))
