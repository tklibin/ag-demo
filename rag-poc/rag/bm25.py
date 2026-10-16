"""Okapi BM25 keyword search.

Policies are full of exact terms that embeddings blur: clause numbers ("4.2"), codes ("CO-197"),
acronyms ("CTAS"), amounts ("SAR 4,000"). BM25 catches those; the vector index catches paraphrases.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Clause numbers and decimals stay whole ("4.2", "4,000"); other words split on non-word characters.
# \w matches Arabic letters too.
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|\w+", re.UNICODE)
STOPWORDS = frozenset("""
a an and are as at be by for from has have if in is it its of on or that the this to was were will with
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.docs) if self.docs else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length) if self.avg_length else self.k1
            out.append(sum(self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm)
                           for t in terms if t in doc))
        return out
