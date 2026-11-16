"""Chunking strategies for the markdown policy corpus.

Strategies (compared in eval/run_retrieval_eval.py):
  fixed_<N>     N-word windows with 15% overlap, ignores structure (baseline)
  heading       one chunk per markdown section (structure-aware)
  heading_cap   heading sections, but oversize sections are split on paragraph
                boundaries to <= max_words, tiny sections merged forward
Every chunk's *indexed* text is prefixed with "Doc title > heading breadcrumb"
(contextual chunk headers), which helps lexical and dense retrieval alike.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)


@dataclass
class Chunk:
    chunk_id: str
    path: str          # relative to corpus root, posix style (matches eval expected_sources)
    doc_title: str
    section: str       # heading breadcrumb
    text: str          # raw text shown to the model
    index: int

    @property
    def indexed_text(self) -> str:
        return f"{self.doc_title} > {self.section}\n{self.text}"


def doc_title_from(path: Path, text: str) -> str:
    m = re.search(r"^title:\s*(.+)$", text[:600], re.M)
    if m:
        return m.group(1).strip("\"' ")
    for line in text.splitlines()[:30]:
        h = HEADING_RE.match(line)
        if h and len(h.group(1)) == 1:
            return h.group(2)
    name = path.parent.name if path.stem == "index" else path.stem
    return name.replace("-", " ").replace("_", " ").title()


def _sections(text: str) -> List[Tuple[str, str]]:
    """Split into (breadcrumb, body) pairs, ignoring '#' inside code fences."""
    text = FRONTMATTER_RE.sub("", text)
    stack: List[Tuple[int, str]] = []
    out: List[Tuple[str, List[str]]] = [("", [])]
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        h = None if in_fence else HEADING_RE.match(line)
        if h:
            level, title = len(h.group(1)), h.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            out.append((" > ".join(t for _, t in stack), []))
        else:
            out[-1][1].append(line)
    return [(b, "\n".join(lines).strip()) for b, lines in out if "\n".join(lines).strip()]


def _split_words(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step) if words[i : i + size]]


def _split_paragraphs(text: str, max_words: int) -> List[str]:
    parts, cur, n = [], [], 0
    for para in re.split(r"\n\s*\n", text):
        w = len(para.split())
        if cur and n + w > max_words:
            parts.append("\n\n".join(cur))
            cur, n = [], 0
        if w > max_words:  # a single huge paragraph/table: fall back to windows
            parts.extend(_split_words(para, max_words, max_words // 7))
            continue
        cur.append(para)
        n += w
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def chunk_document(rel_path: str, raw: str, strategy: str = "heading_cap",
                   max_words: int = 220, min_words: int = 40) -> List[Chunk]:
    title = doc_title_from(Path(rel_path), raw)
    pieces: List[Tuple[str, str]] = []
    if strategy.startswith("fixed_"):
        size = int(strategy.split("_")[1])
        body = FRONTMATTER_RE.sub("", raw)
        pieces = [("", t) for t in _split_words(body, size, int(size * 0.15))]
    else:
        secs = _sections(raw)
        if strategy == "heading":
            pieces = secs
        elif strategy == "heading_cap":
            carry_b, carry_t = "", ""
            for crumb, body in secs:
                if carry_t:  # merge tiny previous section forward
                    body = carry_t + "\n\n" + body
                    crumb = carry_b if not crumb else crumb
                    carry_b = carry_t = ""
                if len(body.split()) < min_words:
                    carry_b, carry_t = crumb, body
                    continue
                for part in _split_paragraphs(body, max_words):
                    pieces.append((crumb, part))
            if carry_t:
                pieces.append((carry_b, carry_t))
        else:
            raise ValueError(f"unknown chunking strategy {strategy}")
    return [
        Chunk(f"{rel_path}#{i}", rel_path, title, crumb or title, t.strip(), i)
        for i, (crumb, t) in enumerate(pieces) if t.strip()
    ]


def load_corpus(root: str | Path, strategy: str = "heading_cap", **kw) -> Tuple[List[Chunk], Dict[str, str]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    import os
    exclude = [x for x in os.getenv("HR_EXCLUDE", "").split(",") if x]  # e.g. HR_EXCLUDE=synthetic/ -> real docs only
    chunks: List[Chunk] = []
    docs: Dict[str, str] = {}
    for fp in sorted(root.rglob("*.md")):
        rel = fp.relative_to(root).as_posix()
        if any(x in rel for x in exclude):
            continue
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            continue
        docs[rel] = raw
        chunks.extend(chunk_document(rel, raw, strategy, **kw))
    return chunks, docs
