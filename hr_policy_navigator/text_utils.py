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


def lcs_length(x: List[str], y: List[str]) -> int:
    """Length of Longest Common Subsequence of token lists (dynamic programming)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev + 1 if x[i - 1] == y[j - 1] else max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def rouge_lcs(candidate: str, reference: str, beta: float = 1.0) -> tuple[float, float, float]:
    """Compute ROUGE-LCS (recall, precision, harmonic F1) on token sequences."""
    c_tokens = tokenize(candidate)
    r_tokens = tokenize(reference)
    if not c_tokens or not r_tokens:
        return 0.0, 0.0, 0.0
    lcs = lcs_length(c_tokens, r_tokens)
    rec = lcs / len(r_tokens)
    prec = lcs / len(c_tokens)
    if rec + prec == 0:
        return rec, prec, 0.0
    f1 = ((1 + beta**2) * rec * prec) / (rec + beta**2 * prec)
    return rec, prec, f1

