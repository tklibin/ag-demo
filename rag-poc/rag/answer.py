"""Answer a question from retrieved chunks with a local model, and verify every citation.

Local models have no built-in citation feature, so the POC enforces it:
  1. Each retrieved chunk is shown to the model with an id: [S1] §4.2 Non-emergency services, p.5
  2. The model must reply in JSON (Ollama's `format` takes a JSON schema): a list of statements,
     each with the id of the excerpt it relies on and a word-for-word quote from that excerpt.
  3. verify() checks each quote really is in the cited excerpt. A statement whose quote can't be
     found anywhere in the retrieved excerpts stays unverified, and the UI shows it as such.
"""

from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata

from . import config
from .models import Answer, Hit, Statement

SYSTEM = """You answer questions about policy documents using only the numbered excerpts provided.

Rules:
- Use only the excerpts. Do not add outside knowledge or assume what policies usually say.
- Break the answer into short factual statements. Each statement cites exactly one excerpt id (S1, S2, ...)
  and includes a quote copied word for word from that excerpt that supports it (one sentence or phrase,
  at most 40 words). Do not paraphrase inside the quote.
- Keep numbers, time limits, amounts, codes and conditions exactly as the excerpt states them.
  Include exceptions and conditions that change the answer.
- If the excerpts do not answer the question, set "covered" to false and return no statements.
  If they answer only part of it, answer that part and describe what is missing in "gaps".
- Write the statements in the language of the question."""

QUOTE_MATCH = 0.9  # share of a quote that must match the excerpt verbatim (after normalising)


def schema(n_sources: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "covered": {"type": "boolean"},
            "statements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source": {"type": "string", "enum": [f"S{i}" for i in range(1, n_sources + 1)]},
                        "quote": {"type": "string"},
                    },
                    "required": ["text", "source", "quote"],
                },
            },
            "gaps": {"type": "string"},
        },
        "required": ["covered", "statements", "gaps"],
    }


def build_prompt(question: str, hits: list[Hit]) -> str:
    excerpts = "\n\n".join(f"[S{i}] {h.chunk.label}\n{h.chunk.text}" for i, h in enumerate(hits, 1))
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def answer(question: str, hits: list[Hit], model: str = config.CHAT_MODEL) -> Answer:
    if not hits:
        return Answer(question, False, [], [], model, 0.0)
    import ollama

    prompt = build_prompt(question, hits)
    start = time.perf_counter()
    response = ollama.Client(host=config.OLLAMA_HOST).chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        format=schema(len(hits)),
        think=config.THINK,
        options={"temperature": config.TEMPERATURE, "num_ctx": context_size(SYSTEM + prompt)},
    )
    seconds = time.perf_counter() - start
    return parse_reply(question, response.message.content, hits, model, seconds)


def context_size(prompt: str, reply_tokens: int = 2048) -> int:
    """Tokens to reserve. Ollama silently drops whatever doesn't fit, so err on the large side:
    ~3 characters per token covers English and Arabic text."""
    needed = len(prompt) // 3 + reply_tokens
    return max(config.MIN_CTX, -(-needed // 4096) * 4096)


def parse_reply(question: str, raw: str, hits: list[Hit], model: str, seconds: float) -> Answer:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"covered": False, "statements": []}
    covered = bool(data.get("covered"))
    statements = [verify(Statement(s.get("text", ""), s.get("source", ""), s.get("quote", "")), hits)
                  for s in (data.get("statements") or []) if covered and s.get("text")]
    return Answer(question, covered, statements, hits, model, seconds, raw, (data.get("gaps") or "").strip())


# --- citation verification ----------------------------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                         "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", text).strip(" .\"'")


def quote_in(quote: str, text: str) -> bool:
    """True if the quote appears in the text. Quotes the model shortened with '...' must match
    piece by piece, in order; a few characters of difference (a dropped comma) are tolerated."""
    pieces = [p for p in (normalize(x) for x in re.split(r"\.\.\.|…", quote)) if len(p) >= 3]
    if not pieces:
        return False
    haystack = normalize(text)
    pos = 0
    for piece in pieces:
        found = haystack.find(piece, pos)
        if found < 0:
            match = difflib.SequenceMatcher(None, haystack, piece, autojunk=False).find_longest_match(
                pos, len(haystack), 0, len(piece))
            if match.size < QUOTE_MATCH * len(piece):
                return False
            found = match.a
        pos = found + 1
    return True


def verify(s: Statement, hits: list[Hit]) -> Statement:
    """Check the quote in the cited excerpt. If the model cited the wrong id but the quote is in
    another retrieved excerpt, re-attribute it: the evidence is real, only the label was wrong."""
    cited = int(s.source[1:]) - 1 if re.fullmatch(r"S\d+", s.source) else -1
    order = ([cited] if 0 <= cited < len(hits) else []) + [i for i in range(len(hits)) if i != cited]
    for i in order:
        if quote_in(s.quote, hits[i].chunk.text):
            s.verified, s.source = True, f"S{i + 1}"
            s.chunk_id, s.label = hits[i].chunk.chunk_id, hits[i].chunk.label
            return s
    if 0 <= cited < len(hits):
        s.chunk_id, s.label = hits[cited].chunk.chunk_id, hits[cited].chunk.label
    return s
