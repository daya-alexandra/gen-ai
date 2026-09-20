"""Небольшой прозрачный BM25-поиск по синтетической базе знаний."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold().replace("ё", "е"))


def load_chunks(folder: str | Path) -> list[dict]:
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        source_id = path.stem
        text = path.read_text(encoding="utf-8").strip()
        title = text.splitlines()[0].strip()
        sections = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
        for number, section in enumerate(sections, 1):
            section = section.strip()
            if not section or section.startswith("# ") and "\n## " not in section and len(section) < 180:
                continue
            if section.startswith("## "):
                section = title + "\n\n" + section
            chunks.append(
                {
                    "chunk_id": f"{source_id}:{number:02d}",
                    "source_id": source_id,
                    "text": section,
                    "tokens": tokenize(section),
                }
            )
    if not chunks:
        raise ValueError("База знаний пуста")
    return chunks


class BM25Index:
    def __init__(self, folder: str | Path):
        self.chunks = load_chunks(folder)
        self.lengths = [len(item["tokens"]) for item in self.chunks]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        self.df = Counter()
        for item in self.chunks:
            self.df.update(set(item["tokens"]))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        terms = tokenize(query)
        n_docs = len(self.chunks)
        scored = []
        for chunk, length in zip(self.chunks, self.lengths):
            tf = Counter(chunk["tokens"])
            score = 0.0
            for term in terms:
                if not tf[term]:
                    continue
                idf = math.log(1 + (n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))
                numerator = tf[term] * 2.5
                denominator = tf[term] + 1.5 * (1 - 0.75 + 0.75 * length / self.avg_len)
                score += idf * numerator / denominator
            scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["chunk_id"]))
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "score": round(score, 6),
                "text": chunk["text"],
            }
            for score, chunk in scored[:top_k]
        ]
