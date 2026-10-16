"""Settings. Every value can be overridden with an environment variable of the same name."""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DATA = Path(_env("RAG_DATA", "data"))
INDEX_DIR = DATA / "index"
UPLOADS = DATA / "uploads"

OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = _env("RAG_CHAT_MODEL", "gemma4:31b")
EMBED_MODEL = _env("RAG_EMBED_MODEL", "bge-m3")  # multilingual (Arabic + English), 8k-token input

# Chunking: a section is split only when it is longer than this, at paragraph boundaries.
CHUNK_MAX_CHARS = int(_env("RAG_CHUNK_MAX_CHARS", "2400"))  # about 600 tokens

# Retrieval: candidates from each retriever, then the number of chunks given to the model.
CANDIDATES = int(_env("RAG_CANDIDATES", "30"))
TOP_K = int(_env("RAG_TOP_K", "6"))
RRF_K = 60  # reciprocal rank fusion constant; 60 is the value from the original paper

# Generation. Ollama's default context window is small and it silently truncates longer prompts,
# so the answer step always sets num_ctx explicitly (see answer.py).
TEMPERATURE = float(_env("RAG_TEMPERATURE", "0.1"))
# gemma4 can think before answering. Off by default: thinking makes every answer slower, and the
# quote check catches unsupported statements anyway. Set RAG_THINK=1 to compare on the eval set.
THINK = _env("RAG_THINK", "0") == "1"
MIN_CTX = 8192
