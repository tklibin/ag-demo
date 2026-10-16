from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Page:
    number: int | None  # 1-based PDF page; None for formats without pages (DOCX)
    text: str


@dataclass
class Chunk:
    chunk_id: str        # "<doc_id>:<n>"
    doc_id: str
    doc_title: str
    section: str         # clause number such as "4.2", or "" before the first heading
    heading: str         # "4.2 Non-emergency services", full heading line
    path: list[str]      # parent headings, outermost first
    page_start: int | None
    page_end: int | None
    text: str

    @property
    def label(self) -> str:
        """Human-readable location used in prompts and citations: '§4.2 Non-emergency services, p.5'."""
        where = self.heading or self.doc_title
        if self.page_start is None:
            return where
        pages = (f"p.{self.page_start}" if self.page_start == self.page_end
                 else f"pp.{self.page_start}-{self.page_end}")
        return f"{where}, {pages}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


@dataclass
class Hit:
    chunk: Chunk
    score: float                                      # fused (RRF) score
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank per retriever, e.g. {"vector": 2, "bm25": 1}


@dataclass
class Statement:
    text: str
    source: str                # "S1".."Sk": which retrieved chunk the model cited
    quote: str                 # verbatim text the model says supports the statement
    verified: bool = False     # the quote was found in the cited chunk
    chunk_id: str | None = None
    label: str | None = None


@dataclass
class Answer:
    question: str
    covered: bool              # the model judged the policy to answer the question
    statements: list[Statement]
    sources: list[Hit]
    model: str
    seconds: float
    raw: str = ""              # the model's raw JSON, kept for debugging
    gaps: str = ""             # what the model says the excerpts don't cover

    @property
    def verified_statements(self) -> list[Statement]:
        return [s for s in self.statements if s.verified]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "covered": self.covered,
            "gaps": self.gaps,
            "statements": [asdict(s) for s in self.statements],
            "sources": [{"id": f"S{i}", "chunk_id": h.chunk.chunk_id, "label": h.chunk.label,
                         "score": round(h.score, 5), "ranks": h.ranks}
                        for i, h in enumerate(self.sources, 1)],
            "model": self.model,
            "seconds": round(self.seconds, 1),
        }
