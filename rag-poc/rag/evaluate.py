"""Score retrieval and answers on a question set (eval/questions.jsonl).

Each question lists the section(s) that answer it, a few phrases a correct answer must contain,
and whether the policy answers it at all. The unanswerable ones matter most for a policy tool:
saying "not covered" beats inventing a rule.

Metrics
  retrieval   hit@k: an expected section is among the k chunks given to the model; MRR
  answers     correct: covered, and every must_mention phrase appears in the statements or quotes
              cited right: at least one verified statement cites an expected section
              declined: unanswerable questions where the model returned no statements
  citations   verified: share of all statements whose quote was found in the cited excerpt
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .answer import answer, normalize
from .index import Index, ollama_embedder
from .retrieve import search


def evaluate(path: Path, index: Index, k: int, model: str, retrieval_only: bool = False,
             limit: int | None = None) -> dict:
    questions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()][:limit]
    embed = ollama_embedder(index.embed_model)
    rows = []
    for q in questions:
        hits = search(index, q["question"], k=k, embed=embed)
        sections = [h.chunk.section for h in hits]
        expected = set(q.get("sections", []))
        rank = next((i for i, s in enumerate(sections, 1) if s in expected), None)
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
               "expected": sorted(expected), "retrieved": sections, "rank": rank}
        if not retrieval_only:
            a = answer(q["question"], hits, model=model)
            said = normalize(" ".join(f"{s.text} {s.quote}" for s in a.statements))
            row.update({
                "covered": a.covered and bool(a.statements),
                "statements": len(a.statements),
                "verified": len(a.verified_statements),
                "cited_sections": sorted({hits[int(s.source[1:]) - 1].chunk.section
                                          for s in a.verified_statements}),
                "missing": [m for m in q.get("must_mention", []) if normalize(m) not in said],
                "seconds": round(a.seconds, 1),
                "answer": [s.text for s in a.statements],
            })
            row["correct"] = row["covered"] and not row["missing"] if q["answerable"] else not row["covered"]
            row["cited_right"] = bool(expected & set(row["cited_sections"]))
        rows.append(row)
        _progress(row)
    return {"k": k, "model": None if retrieval_only else model, "summary": summarize(rows), "rows": rows}


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    s = {
        "questions": len(rows),
        "hit_at_k": _mean(r["rank"] is not None for r in answerable),
        "mrr": _mean((1 / r["rank"]) if r["rank"] else 0.0 for r in answerable),
    }
    if rows and "covered" in rows[0]:
        statements = sum(r["statements"] for r in rows)
        s.update({
            "answer_correct": _mean(r["correct"] for r in answerable),
            "cited_right": _mean(r["cited_right"] for r in answerable),
            "declined_unanswerable": _mean(r["correct"] for r in unanswerable),
            "quotes_verified": sum(r["verified"] for r in rows) / statements if statements else None,
            "seconds_median": statistics.median(r["seconds"] for r in rows),
            "seconds_max": max(r["seconds"] for r in rows),
        })
    return s


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"\n== {s['questions']} questions, k={report['k']}"
          + (f", model {report['model']}" if report["model"] else ", retrieval only") + " ==")
    print(f"  Retrieval  hit@{report['k']} {_pct(s['hit_at_k'])}   MRR {s['mrr']:.3f}")
    if "answer_correct" in s:
        print(f"  Answers    correct {_pct(s['answer_correct'])}   cited the right section {_pct(s['cited_right'])}")
        print(f"  Unanswerable questions declined {_pct(s['declined_unanswerable'])}")
        print(f"  Quotes verified {_pct(s['quotes_verified'])}")
        print(f"  Latency    median {s['seconds_median']:.1f}s   max {s['seconds_max']:.1f}s")

    misses = [r for r in report["rows"] if (r["answerable"] and r["rank"] is None) or r.get("correct") is False]
    if misses:
        print("\n== To look at ==")
    for r in misses:
        print(f"  {r['id']} {r['question']}")
        if r["answerable"] and r["rank"] is None:
            print(f"      retrieval missed {r['expected']}; got {r['retrieved']}")
        if r.get("correct") is False:
            if not r["answerable"]:
                print(f"      should have declined; answered: {r['answer']}")
            elif not r["covered"]:
                print("      declined, but the policy answers it")
            else:
                print(f"      missing {r['missing']}; answered: {r['answer']}")


def _progress(row: dict) -> None:
    mark = "" if "correct" not in row else ("  ok" if row["correct"] else "  WRONG")
    found = f"rank {row['rank']}" if row["rank"] else ("-" if not row["answerable"] else "MISSED")
    print(f"  {row['id']}  {found:<7}{mark}" + (f"  {row['seconds']}s" if "seconds" in row else ""), flush=True)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"
