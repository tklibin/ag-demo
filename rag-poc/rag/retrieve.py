"""Hybrid retrieval: vector search and BM25, merged with reciprocal rank fusion (RRF).

RRF only uses ranks, so the two retrievers' scores (cosine similarity vs BM25) never need to be put
on the same scale: score = sum over retrievers of 1 / (RRF_K + rank).
"""

from __future__ import annotations

import numpy as np

from . import config
from .index import Embedder, Index, ollama_embedder
from .models import Hit


def search(index: Index, query: str, k: int = config.TOP_K, embed: Embedder | None = None,
           doc_ids: set[str] | None = None, candidates: int = config.CANDIDATES) -> list[Hit]:
    if not index.chunks:
        return []
    allowed = np.array([doc_ids is None or c.doc_id in doc_ids for c in index.chunks])

    embed = embed or ollama_embedder(index.embed_model)  # queries must use the index's own model
    similarity = index.vectors @ embed([query])[0]
    keyword = np.asarray(index.bm25.scores(query))

    rankings = {
        "vector": _top(similarity, allowed, candidates),
        "bm25": _top(keyword, allowed & (keyword > 0), candidates),
    }
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, order in rankings.items():
        for rank, i in enumerate(order, 1):
            fused[i] = fused.get(i, 0.0) + 1 / (config.RRF_K + rank)
            ranks.setdefault(i, {})[name] = rank

    best = sorted(fused, key=lambda i: -fused[i])[:k]
    return [Hit(index.chunks[i], fused[i], ranks[i]) for i in best]


def _top(scores: np.ndarray, mask: np.ndarray, n: int) -> list[int]:
    idx = np.flatnonzero(mask)
    return idx[np.argsort(-scores[idx], kind="stable")][:n].tolist()
