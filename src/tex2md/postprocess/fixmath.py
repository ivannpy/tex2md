"""``tex2md fixmath FILE...``: re-run the math clean-up on already converted .md (body only, idempotent)."""
from __future__ import annotations

import re

from tex2md.postprocess.markdown import (
    cap_heading_levels,
    fix_text_math,
    isolate_display_math,
    normalize_math_macros,
    protect_code,
    restore_code,
    tidy_math,
)


def fix_body(body: str) -> str:
    body, code = protect_code(cap_heading_levels(body))
    body = isolate_display_math(tidy_math(fix_text_math(normalize_math_macros(body))))
    return re.sub(r"\n{3,}", "\n\n", restore_code(body, code))


def cmd_fixmath(a):
    for f in a.files:
        with open(f) as fh:
            t = fh.read()
        head, sep, body = t.partition("\n---\n\n")
        if not sep:
            continue
        new = fix_body(body)
        if new != body:
            with open(f, "w") as fh:
                fh.write(head + sep + new)
            print("fixed", f)
    return 0
