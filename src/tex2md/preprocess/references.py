r"""\ref/\cref/\eqref resolution to the numbers of the compiled PDF."""
from __future__ import annotations

import re

from tex2md.build.aux import fmt_num, ref_name
from tex2md.util.tex import group


def _join(nums):
    return nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]


def _plural(name):
    return name[:-1] + "s." if name.endswith(".") else name + "s"


_TEXT_MACRO = re.compile(r"\\(?:def\s*\\([a-zA-Z]+)|(?:re)?newcommand\*?\s*\{?\\([a-zA-Z]+)\}?)\s*\{([\w .-]*)\}")
# a key with one level of braces: \ref{alg:\method{}-train}
_KEY = r"\{((?:[^{}]|\{[^{}]*\})*)\}"


def _expand_key(key, macros):
    r"""``eq:\method{}_loss`` -> ``eq:LoMDM{}_loss``, the way LaTeX writes the key to the .aux."""
    return re.sub(r"\\([a-zA-Z]+)(?![a-zA-Z])\s*", lambda m: macros.get(m.group(1), m.group(0)), key)


def resolve_refs(text, labels, stats):
    macros = {}
    if re.search(r"\\(?:ref|eqref|cref|Cref|autoref)\*?\s*\{[^{}]*\\", text):
        macros = {m.group(1) or m.group(2): m.group(3) for m in _TEXT_MACRO.finditer(text)}

    def lookup(key):
        lab = labels.get(key)
        if not lab and "\\" in key and macros:
            lab = labels.get(_expand_key(key, macros))
        if not lab:
            stats["unresolved_refs"] += 1
        return lab

    def one(cmd, keys):
        out = []
        for k in [k.strip() for k in keys.split(",") if k.strip()]:
            lab = lookup(k)
            if not lab:
                out.append(("", "??"))
                continue
            formatted = cmd in ("eqref", "cref", "Cref", "autoref", "Autoref", "cpageref", "labelcref")
            out.append((ref_name(lab["type"]), fmt_num(lab) if formatted else lab["num"]))
        if cmd in ("ref", "eqref", "pageref", "cpageref", "nameref"):
            if cmd == "eqref":
                return ", ".join(n if n.startswith("(") else f"({n})" for _, n in out)
            if cmd in ("pageref", "cpageref"):
                return ""
            return ", ".join(n for _, n in out)
        if cmd == "labelcref":      # the numbers alone, joined as cleveref does
            return _join([n for _, n in out])
        if cmd in ("namecref", "nameCref"):     # the kind of object alone ("section")
            name = out[0][0] if out else ""
            return name if cmd == "nameCref" else name.lower()
        # cref / Cref / autoref: prefix with the kind of object
        parts, i = [], 0
        while i < len(out):
            name = out[i][0]
            nums = [out[i][1]]
            while i + 1 < len(out) and out[i + 1][0] == name:
                i += 1
                nums.append(out[i][1])
            i += 1
            plural = name if len(nums) == 1 or not name else _plural(name)
            parts.append((plural + "~" + _join(nums)) if name else _join(nums))
        return ", ".join(parts)

    pat = re.compile(r"\\(ref|eqref|cref|Cref|autoref|Autoref|pageref|cpageref|nameref|vref|Vref|labelcref|namecref|nameCref)"
                     r"\*?\s*" + _KEY)

    def sub(m):
        if "#" in m.group(2):  # inside a macro definition
            return m.group(0)
        return one({"vref": "cref", "Vref": "Cref"}.get(m.group(1), m.group(1)), m.group(2))

    def subref(m):
        r"""``\subref{fig:a}`` -> "(a)": the letter of the sub-figure (subcaption writes it as ``sub@fig:a``)."""
        key = m.group(2).strip()
        if "#" in key:
            return m.group(0)
        lab = labels.get("sub@" + key)
        letter = lab["num"] if lab else re.sub(r"^[0-9.]+", "", (lookup(key) or {}).get("num", "??")) or "??"
        letter = letter.strip("()")
        return letter if m.group(1) and m.group(3) else f"({letter})"     # "(\subref{x})" has its own parentheses

    def crefrange(m):
        a, b = lookup(m.group(2).strip()), lookup(m.group(3).strip())
        if not a or not b:
            return "??"
        name = ref_name(a["type"])
        return (_plural(name) + "~" if name else "") + fmt_num(a) + " to " + fmt_num(b)

    text = re.sub(r"(\()?\\subref\*?\s*\{([^}]*)\}(\))?",
                  lambda m: (m.group(1) or "") + subref(m) + (m.group(3) or ""), text)
    text = re.sub(r"\\([cC])refrange\*?\s*\{([^}#]*)\}\s*\{([^}#]*)\}", crefrange, text)
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
