r"""\ref/\cref/\eqref resolution to the numbers of the compiled PDF."""
from __future__ import annotations

import re

from tex2md.build.aux import fmt_num, ref_name
from tex2md.util.tex import group


def resolve_refs(text, labels, stats):
    def one(cmd, keys):
        out = []
        for k in [k.strip() for k in keys.split(",") if k.strip()]:
            lab = labels.get(k)
            if not lab:
                stats["unresolved_refs"] += 1
                out.append(("", "??"))
                continue
            out.append((ref_name(lab["type"]), fmt_num(lab) if cmd in ("eqref", "cref", "Cref", "autoref", "Autoref", "cpageref") else lab["num"]))
        if cmd in ("ref", "eqref", "pageref", "cpageref", "nameref"):
            if cmd == "eqref":
                return ", ".join(n if n.startswith("(") else f"({n})" for _, n in out)
            if cmd in ("pageref", "cpageref"):
                return ""
            return ", ".join(n for _, n in out)
        # cref / Cref / autoref: prefix with the kind of object
        parts, i = [], 0
        while i < len(out):
            name = out[i][0]
            nums = [out[i][1]]
            while i + 1 < len(out) and out[i + 1][0] == name:
                i += 1
                nums.append(out[i][1])
            i += 1
            plural = name if len(nums) == 1 or not name else (name[:-1] + "s." if name.endswith(".") else name + "s")
            joined = nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]
            parts.append((plural + "~" + joined) if name else joined)
        return ", ".join(parts)

    pat = re.compile(r"\\(ref|eqref|cref|Cref|autoref|Autoref|pageref|cpageref|nameref|vref|Vref)\*?\s*\{([^}]*)\}")

    def sub(m):
        if "#" in m.group(2):  # inside a macro definition
            return m.group(0)
        return one({"vref": "cref", "Vref": "Cref"}.get(m.group(1), m.group(1)), m.group(2))
    return pat.sub(sub, text)


WRAPPED = re.compile(r"\\(ref|eqref|cref|Cref|autoref|Autoref|cite[a-z]*|Cite[a-z]*|newcite)\b")


def expand_wrappers(text):
    """Inline one-argument user macros that wrap \\ref or \\cite (\\eqnref{x} -> Eq.~\\eqref{x})
    so references and citations can be resolved before pandoc sees them."""
    defs = {}
    for m in re.finditer(r"\\(?:re)?newcommand\*?\s*\{?\\([a-zA-Z]+)\}?\s*\[1\]\s*", text):
        i = m.end()
        if i < len(text) and text[i] == "{":
            body, _ = group(text, i)
            if WRAPPED.search(body):
                defs[m.group(1)] = body
    for m in re.finditer(r"\\def\s*\\([a-zA-Z]+)\s*#1\s*", text):
        i = m.end()
        if i < len(text) and text[i] == "{":
            body, _ = group(text, i)
            if WRAPPED.search(body):
                defs[m.group(1)] = body
    if not defs:
        return text
    pat = re.compile(r"\\(" + "|".join(sorted(map(re.escape, defs), key=len, reverse=True)) + r")(?![a-zA-Z])\s*\{")
    out, i = [], 0
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        arg, j = group(text, m.end() - 1)
        if "#" in arg:
            out.append(text[m.start():j])
        else:
            out.append(defs[m.group(1)].replace("#1", arg))
        i = j
    return "".join(out)
