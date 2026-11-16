"""PDF -> Markdown body with marker, run in its own isolated uv project (see ``tex2md.config``)."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from tex2md.config import MARKER, MODELS


def marker_body(pdf, mode):
    env = dict(os.environ, MODEL_CACHE_DIR=MODELS,
               HF_HOME=os.path.join(MODELS, "hf"))
    with tempfile.TemporaryDirectory() as tmp:
        # marker 1.x (pure PyTorch; 2.x needs an external llama.cpp server).
        # balanced = OCR every page (better math); fast = OCR only where the text layer is bad.
        cmd = ["uv", "run", "--project", MARKER, "marker_single", pdf, "--output_dir", tmp,
               "--disable_image_extraction"] + (["--force_ocr"] if mode == "balanced" else [])
        p = subprocess.run(cmd, cwd=MARKER, env=env, capture_output=True, text=True)
        mds = [os.path.join(dp, f) for dp, _, fns in os.walk(tmp) for f in fns if f.endswith(".md")]
        if p.returncode != 0 or not mds:
            raise SystemExit("marker failed: " + p.stderr[-800:])
        body = Path(mds[0]).read_text(encoding="utf-8")
    # marker leaves empty image links when image extraction is off
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", body)
    body = re.sub(r"<span id=\"page-\d+-\d+\"></span>", "", body)
    body = re.sub(r"(?m)^(#+) \*\*(.+?)\*\*\s*$", r"\1 \2", body)

    def level(m):
        # marker guesses heading levels; numbered headings get theirs from the number
        text = re.sub(r"\*\*([0-9A-Z][0-9.]*)\*\*", r"\1", m.group(2)).strip()
        n = re.match(r"^([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)+)\.?\s+\S", text)
        depth = n.group(1).count(".") + 1 if n else None
        if depth:
            return "#" * min(depth, 4) + " " + text
        if re.match(r"(?i)^(abstract|references|bibliography|acknowledge?ments?|appendix)\b", text):
            return "# " + text
        return m.group(1) + " " + text
    body = re.sub(r"(?m)^(#+) (.+)$", level, body)
    return re.sub(r"\n{3,}", "\n\n", body)
