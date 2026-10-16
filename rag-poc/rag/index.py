"""The document index: chunks, their embedding vectors, and a BM25 keyword index.

Stored as plain files in data/index/: chunks.jsonl, vectors.npy and meta.json. That is enough for
a handful of policy documents; a vector database is only needed at a much larger scale.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import config
from .bm25 import BM25
from .chunking import embedding_text
from .models import Chunk

Embedder = Callable[[list[str]], np.ndarray]  # texts -> (n, dim) unit vectors


def ollama_embedder(model: str = config.EMBED_MODEL, batch: int = 32) -> Embedder:
    import ollama

    client = ollama.Client(host=config.OLLAMA_HOST)

    def embed(texts: list[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch):
            vectors += client.embed(model=model, input=texts[i:i + batch], truncate=True).embeddings
        return normalize(np.asarray(vectors, dtype=np.float32))

    return embed


def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embed_model: str):
        self.embed_model = embed_model
        self._set(chunks, vectors)

    def _set(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25([embedding_text(c) for c in chunks])

    @classmethod
    def empty(cls, embed_model: str = config.EMBED_MODEL) -> Index:
        return cls([], np.zeros((0, 0), dtype=np.float32), embed_model)

    # --- documents ------------------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embed: Embedder) -> None:
        """Add a document's chunks, replacing any earlier version of the same document."""
        if not chunks:
            raise ValueError("No text found in the document (a scanned PDF? try --ocr)")
        doc_id = chunks[0].doc_id
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        new = embed([embedding_text(c) for c in chunks])
        old = self.vectors[keep] if keep else np.zeros((0, new.shape[1]), dtype=np.float32)
        self._set([self.chunks[i] for i in keep] + chunks, np.vstack([old, new]))

    def remove(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        self._set([self.chunks[i] for i in keep], self.vectors[keep])
        return removed

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.chunks:
            d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.doc_title, "chunks": 0, "pages": 0})
            d["chunks"] += 1
            d["pages"] = max(d["pages"], c.page_end or 0)
        return list(docs.values())

    # --- persistence ----------------------------------------------------------------------------

    def save(self, directory: Path = config.INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "meta.json").write_text(json.dumps({"embed_model": self.embed_model}, indent=2))

    @classmethod
    def load(cls, directory: Path = config.INDEX_DIR) -> Index:
        if not (directory / "meta.json").exists():
            return cls.empty()
        meta = json.loads((directory / "meta.json").read_text())
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        return cls(chunks, np.load(directory / "vectors.npy"), meta["embed_model"])
