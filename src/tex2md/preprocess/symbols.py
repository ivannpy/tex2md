"""Symbols pandoc drops in text mode: pifont dingbats, check marks, ``\\text...`` symbols."""
from __future__ import annotations

import re

from tex2md.util.tex import rewrite_outside_math

# Zapf Dingbats codes that do not sit at U+2700 + (code - 32)
_DING_EXCEPTIONS = {37: 0x260E, 42: 0x261B, 43: 0x261E, 72: 0x2605, 108: 0x25CF, 110: 0x25A0, 115: 0x25B2,
                    116: 0x25BC, 117: 0x25C6, 119: 0x25D7, 168: 0x2663, 169: 0x2666, 170: 0x2665, 171: 0x2660,
                    213: 0x2192, 214: 0x2194, 215: 0x2195}


def ding(code: int) -> str:
    """The Unicode character of pifont's ``\\ding{code}`` ("" when the code has none)."""
    if code in _DING_EXCEPTIONS:
        return chr(_DING_EXCEPTIONS[code])
    if 33 <= code <= 126:
        return chr(0x2700 + code - 32)
    if 161 <= code <= 167:
        return chr(0x2761 + code - 161)
    if 172 <= code <= 181:      # circled digits 1-10
        return chr(0x2460 + code - 172)
    if 182 <= code <= 212:      # negative circled digits, sans-serif circled digits, heavy arrow
        return chr(0x2776 + code - 182)
    if 216 <= code <= 239 or 241 <= code <= 254:
        return chr(0x2798 + code - 216)
    return ""


# command -> replacement (LaTeX-safe text). Only applied in text mode: several are valid math commands.
TEXT_SYMBOLS = {
    "checkmark": "\u2713", "Checkmark": "\u2713", "CheckmarkBold": "\u2714", "XSolid": "\u2717",
    "XSolidBrush": "\u2717", "XSolidBold": "\u2718", "FiveStar": "\u2605", "FiveStarOpen": "\u2606",
    "faTimes": "\u2717", "faXmark": "\u2717", "faStar": "\u2605",
    "dag": "\u2020", "ddag": "\u2021", "textdaggerdbl": "\u2021", "textasteriskcentered": "\u2217",
    "textperiodcentered": "\u00b7", "texttimes": "\u00d7", "textpm": "\u00b1", "textdiv": "\u00f7",
    "textminus": "\u2212", "textrightarrow": "\u2192", "textleftarrow": "\u2190", "textuparrow": "\u2191",
    "textdownarrow": "\u2193", "textbar": "|", "textunderscore": "\\_", "texttrademark": "\u2122",
    "texteuro": "\u20ac", "textellipsis": "\u2026", "textendash": "\u2013", "textemdash": "\u2014",
    "textvisiblespace": "\u2423", "textohm": "\u2126", "textordfeminine": "\u00aa", "textordmasculine": "\u00ba",
    "textcent": "\u00a2", "textexclamdown": "\u00a1", "textquestiondown": "\u00bf", "textbraceleft": "\\{",
    "textbraceright": "\\}", "textperthousand": "\u2030", "textnumero": "\u2116", "textopenbullet": "\u25e6",
    "textlangle": "\u27e8", "textrangle": "\u27e9", "textsurd": "\u221a", "textcelsius": "\u2103",
    "textreferencemark": "\u203b", "textbardbl": "\u2016", "textlnot": "\u00ac", "textcurrency": "\u00a4",
    "textonesuperior": "\u00b9", "textinterrobang": "\u203d",
}
# pifont convention (\newcommand{\cmark}{\ding{51}}): only when the document does not define them itself
CONVENTIONAL = {"cmark": "\u2713", "xmark": "\u2717"}

_DEFINES = r"\\(?:(?:re)?newcommand|providecommand|DeclareRobustCommand|def|let|gdef)\*?\s*\{?\\%s(?![a-zA-Z@])"


def symbols_to_unicode(text: str) -> str:
    r"""``\ding{51}`` -> U+2713 everywhere; ``\checkmark``, ``\textendash``, ``\dag``... -> their character in
    text mode. pandoc knows none of them outside math and drops them (empty table cells where the PDF shows a
    check mark)."""
    text = re.sub(r"\\ding\s*\{\s*(\d+)\s*\}", lambda m: ding(int(m.group(1))) or m.group(0), text)
    table = dict(TEXT_SYMBOLS)
    for name, char in CONVENTIONAL.items():
        if re.search(r"\\" + name + r"(?![a-zA-Z@])", text):
            table[name] = char
    names = [n for n in table if not re.search(_DEFINES % n, text)]
    if not names:
        return text
    pattern = re.compile(r"\\(" + "|".join(sorted(names, key=len, reverse=True)) + r")(?![a-zA-Z@])(\{\}|[ \t]*)")

    def handler(m, _text):
        # a control word eats the blanks after it; "{}" ends it explicitly
        return table[m.group(1)], m.end()
    return rewrite_outside_math(text, pattern, handler, passes=1)
