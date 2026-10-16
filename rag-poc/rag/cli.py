"""Command line: python -m rag {sample,ingest,docs,remove,chunks,search,ask,eval,serve}"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config
from .answer import answer
from .chunking import chunk
from .index import Index, ollama_embedder
from .models import Answer
from .parse import parse
from .retrieve import search

SAMPLE_PDF = config.DATA / "sample" / "ed_claims_policy.pdf"
SAMPLE_DOCX = config.DATA / "sample" / "ed_claims_policy.docx"


# --- commands -----------------------------------------------------------------------------------

def cmd_sample(args) -> None:
    from . import sample_policy

    sample_policy.write_pdf(SAMPLE_PDF)
    sample_policy.write_docx(SAMPLE_DOCX)
    print(f"Wrote {SAMPLE_PDF} and {SAMPLE_DOCX} (fictional policy for testing)")


def cmd_ingest(args) -> None:
    for path in map(Path, args.paths):
        result = ingest(path, title=args.title, ocr=args.ocr)
        print(f"Ingested {path.name}: {result['chunks']} chunks from {result['pages']} page(s) "
              f"as '{result['doc_id']}' in {result['seconds']:.1f}s")


def ingest(path: Path, title: str | None = None, ocr: bool = False, index: Index | None = None) -> dict:
    import time

    start = time.perf_counter()
    pages = parse(path, ocr=ocr)
    doc_id = doc_id_for(path)
    chunks = chunk(pages, doc_id, title or _title(path, pages))
    index = index or Index.load()
    index.add(chunks, ollama_embedder(index.embed_model))
    index.save()
    return {"doc_id": doc_id, "chunks": len(chunks), "pages": len(pages),
            "seconds": time.perf_counter() - start}


def cmd_docs(args) -> None:
    docs = Index.load().documents()
    if not docs:
        print("No documents yet. Try: python -m rag sample && python -m rag ingest data/sample/ed_claims_policy.pdf")
    for d in docs:
        print(f"{d['doc_id']:<32} {d['chunks']:>4} chunks  {d['pages'] or '-':>4} pages  {d['title']}")


def cmd_remove(args) -> None:
    index = Index.load()
    removed = index.remove(args.doc_id)
    index.save()
    print(f"Removed {removed} chunks of {args.doc_id}")


def cmd_chunks(args) -> None:
    for c in Index.load().chunks:
        if args.doc_id and c.doc_id != args.doc_id:
            continue
        print(f"--- {c.chunk_id}  [{c.section or '-'}]  {c.label}")
        print("    path: " + " > ".join(c.path))
        print("    " + c.text[:args.width].replace("\n", " ") + ("…" if len(c.text) > args.width else ""))


def cmd_search(args) -> None:
    hits = search(Index.load(), args.question, k=args.k, doc_ids=_doc_filter(args))
    for i, h in enumerate(hits, 1):
        ranks = ", ".join(f"{k} #{v}" for k, v in h.ranks.items())
        print(f"[S{i}] {h.score:.4f}  ({ranks})  {h.chunk.label}")
        print("     " + h.chunk.text[:160].replace("\n", " "))


def cmd_ask(args) -> None:
    index = Index.load()
    if not index.chunks:
        raise SystemExit("The index is empty. Ingest a document first: python -m rag ingest <file>")
    hits = search(index, args.question, k=args.k, doc_ids=_doc_filter(args))
    result = answer(args.question, hits, model=args.model)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_answer(result, show_sources=args.sources)


def cmd_eval(args) -> None:
    from .evaluate import evaluate, print_report

    report = evaluate(Path(args.questions), Index.load(), k=args.k, model=args.model,
                      retrieval_only=args.retrieval_only, limit=args.limit)
    print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.out}")


def cmd_serve(args) -> None:
    app = Path(__file__).with_name("app.py")
    streamlit = shutil.which("streamlit", path=str(Path(sys.executable).parent)) or "streamlit"
    subprocess.run([streamlit, "run", str(app), "--server.port", str(args.port)], check=False)


# --- output -------------------------------------------------------------------------------------

def print_answer(a: Answer, show_sources: bool = False) -> None:
    print(f"Q: {a.question}\n")
    if not a.covered or not a.statements:
        print("The policy excerpts found do not answer this question.")
        if a.gaps:
            print(f"  ({a.gaps})")
    for s in a.statements:
        mark = "" if s.verified else "  [unverified: quote not found in the cited excerpt]"
        print(f"- {s.text} [{s.source}]{mark}")
    if a.covered and a.gaps:
        print(f"\nNot covered: {a.gaps}")
    cited = {s.source for s in a.statements}
    print("\nSources:")
    for i, h in enumerate(a.sources, 1):
        sid = f"S{i}"
        if sid in cited or show_sources:
            print(f"  [{sid}] {h.chunk.label}")
            for s in a.statements:
                if s.source == sid and s.verified:
                    print(f"        \"{s.quote}\"")
    print(f"\n({a.model}, {a.seconds:.1f}s, {len(a.verified_statements)}/{len(a.statements)} statements verified)")


# --- helpers ------------------------------------------------------------------------------------

def doc_id_for(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "document"


def _title(path: Path, pages) -> str:
    """The first heading in the document, or else the file name."""
    for page in pages:
        for line in page.text.splitlines():
            if line.startswith("#"):
                return line.lstrip("# ").strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _doc_filter(args) -> set[str] | None:
    return set(args.doc) if getattr(args, "doc", None) else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m rag", description="Ask questions about policy documents (local models)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("sample", help="write a fictional sample policy (PDF + DOCX) to data/sample/").set_defaults(func=cmd_sample)

    i = sub.add_parser("ingest", help="parse, chunk and index PDF / DOCX / MD / TXT files")
    i.add_argument("paths", nargs="+")
    i.add_argument("--title", help="document title (default: first heading)")
    i.add_argument("--ocr", action="store_true", help="transcribe pages without a text layer with the vision model")
    i.set_defaults(func=cmd_ingest)

    sub.add_parser("docs", help="list indexed documents").set_defaults(func=cmd_docs)

    r = sub.add_parser("remove", help="remove a document from the index")
    r.add_argument("doc_id")
    r.set_defaults(func=cmd_remove)

    c = sub.add_parser("chunks", help="show chunks with their section, pages and heading path")
    c.add_argument("--doc-id")
    c.add_argument("--width", type=int, default=140)
    c.set_defaults(func=cmd_chunks)

    for name, fn, help_text in [("search", cmd_search, "retrieval only: show the chunks a question finds"),
                                ("ask", cmd_ask, "answer a question with verified citations")]:
        q = sub.add_parser(name, help=help_text)
        q.add_argument("question")
        q.add_argument("--k", type=int, default=config.TOP_K)
        q.add_argument("--doc", action="append", help="limit to a document id (repeatable)")
        if name == "ask":
            q.add_argument("--model", default=config.CHAT_MODEL)
            q.add_argument("--sources", action="store_true", help="list every retrieved chunk, not only cited ones")
            q.add_argument("--json", action="store_true")
        q.set_defaults(func=fn)

    e = sub.add_parser("eval", help="score retrieval and answers on a question set")
    e.add_argument("--questions", default="eval/questions.jsonl")
    e.add_argument("--k", type=int, default=config.TOP_K)
    e.add_argument("--model", default=config.CHAT_MODEL)
    e.add_argument("--retrieval-only", action="store_true", help="skip the answer model (fast)")
    e.add_argument("--limit", type=int, help="only the first N questions")
    e.add_argument("--out", help="write the full report as JSON")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("serve", help="start the upload-and-ask web page (Streamlit)")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)
