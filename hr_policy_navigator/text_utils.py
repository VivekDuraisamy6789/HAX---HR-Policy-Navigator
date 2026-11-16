"""Shared tokenisation helpers (no heavy dependencies)."""
from __future__ import annotations

import re
from typing import List

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be been but by can do does for from had has have how i if in into is it its
    me my of on or our should so than that the their them then there these they this to up us was we
    were what when where which who will with would you your about any much many""".split()
)


def stem(tok: str) -> str:
    """Very light suffix stripper: good enough to match plural/-ing/-ed forms."""
    if len(tok) <= 3:
        return tok
    for suf, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""), ("ly", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + repl
    return tok


def tokenize(text: str, remove_stop: bool = True) -> List[str]:
    toks = TOKEN_RE.findall(text.lower())
    if remove_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]
