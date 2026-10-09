from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


def grams(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[index : index + 3] for index in range(len(normalized) - 2)}


@dataclass
class Hit:
    index: int
    score: float


class CharTfidf:
    """Dependency-free Chinese-friendly character trigram retriever."""

    def __init__(self, documents: Iterable[str]):
        self.docs = [grams(doc) for doc in documents]
        self.df: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, document in enumerate(self.docs):
            self.df.update(document)
            for token in document:
                self.postings[token].append(index)
        self.size = len(self.docs)
        self.norms = [self._norm(document) for document in self.docs]

    def _weight(self, token: str) -> float:
        return math.log((self.size + 1) / (self.df[token] + 1)) + 1.0

    def _norm(self, document: set[str]) -> float:
        return math.sqrt(sum(self._weight(token) ** 2 for token in document)) or 1.0

    def search(self, query: str, limit: int = 8, allowed: set[int] | None = None) -> list[Hit]:
        tokens = grams(query)
        candidates: set[int] = set()
        for token in tokens:
            candidates.update(self.postings.get(token, []))
        if allowed is not None:
            candidates.intersection_update(allowed)
        qnorm = self._norm(tokens)
        ranked: list[Hit] = []
        for index in candidates:
            overlap = tokens.intersection(self.docs[index])
            score = sum(self._weight(token) ** 2 for token in overlap) / (qnorm * self.norms[index])
            ranked.append(Hit(index, score))
        return sorted(ranked, key=lambda hit: hit.score, reverse=True)[:limit]
