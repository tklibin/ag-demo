"""Upload-and-ask web page. Start with: python -m rag serve  (or: streamlit run rag/app.py)"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Streamlit runs this file as a script

from rag import config  # noqa: E402
from rag.answer import answer  # noqa: E402
from rag.cli import ingest  # noqa: E402
from rag.index import Index  # noqa: E402
from rag.models import Answer  # noqa: E402
from rag.parse import SUPPORTED  # noqa: E402
from rag.retrieve import search  # noqa: E402

st.set_page_config(page_title="Policy Q&A", page_icon="📄", layout="wide")
st.session_state.setdefault("history", [])


# --- sidebar: documents and settings ------------------------------------------------------------

with st.sidebar:
    st.header("Documents")
    upload = st.file_uploader("Upload a policy", type=[s.lstrip(".") for s in sorted(SUPPORTED)])
    ocr = st.checkbox("Scanned PDF (OCR with the vision model, slow)")
    if upload is not None and st.button("Add to index", type="primary"):
        path = config.UPLOADS / upload.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(upload.getvalue())
        with st.spinner(f"Reading and indexing {upload.name}…"):
            try:
                result = ingest(path, ocr=ocr)
                st.success(f"Indexed {result['chunks']} sections from {result['pages']} page(s) "
                           f"in {result['seconds']:.0f}s")
            except Exception as e:  # show the reason (e.g. no text layer) instead of a stack trace
                st.error(str(e))

    index = Index.load()
    docs = index.documents()
    for d in docs:
        col1, col2 = st.columns([5, 1])
        col1.markdown(f"**{d['title']}**  \n{d['chunks']} sections · {d['pages'] or '–'} pages")
        if col2.button("✕", key=f"rm-{d['doc_id']}", help="Remove from index"):
            index.remove(d["doc_id"])
            index.save()
            st.rerun()
    selected = st.multiselect("Search in", [d["doc_id"] for d in docs], default=[d["doc_id"] for d in docs],
                              format_func=lambda i: next(d["title"] for d in docs if d["doc_id"] == i))

    st.header("Settings")
    model = st.text_input("Answer model (Ollama)", config.CHAT_MODEL)
    k = st.slider("Sections given to the model", 3, 12, config.TOP_K)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()


# --- main: questions and answers ----------------------------------------------------------------

st.title("Policy Q&A")
st.caption("Answers come only from the uploaded documents. Every statement shows the clause it relies on; "
           "a statement whose quote could not be found in that clause is marked as unverified.")


def show(a: Answer) -> None:
    if not a.covered or not a.statements:
        st.warning("The policy sections found do not answer this question." + (f" {a.gaps}" if a.gaps else ""))
    for s in a.statements:
        tag = f"`{s.source}` {s.label or ''}"
        if s.verified:
            st.markdown(f"- {s.text}  \n  <small>{tag}</small>", unsafe_allow_html=True)
        else:
            st.markdown(f"- {s.text}  \n  <small>{tag} · ⚠️ unverified: quote not found</small>",
                        unsafe_allow_html=True)
    if a.covered and a.gaps:
        st.info(f"Not covered: {a.gaps}")

    cited = {s.source for s in a.statements}
    with st.expander(f"Sources ({len(cited)} cited of {len(a.sources)} retrieved) · {a.seconds:.0f}s · {a.model}"):
        for i, h in enumerate(a.sources, 1):
            sid = f"S{i}"
            quotes = [s.quote for s in a.statements if s.source == sid and s.verified]
            st.markdown(f"**[{sid}] {h.chunk.label}**" + ("" if sid in cited else "  · not cited"))
            text = html.escape(h.chunk.text)
            for q in quotes:  # highlight verified quotes when they appear verbatim
                text = text.replace(html.escape(q), f"<mark>{html.escape(q)}</mark>")
            st.markdown(f"<div style='font-size:0.9em;white-space:pre-wrap'>{text}</div>", unsafe_allow_html=True)


for past in st.session_state.history:
    with st.chat_message("user"):
        st.write(past.question)
    with st.chat_message("assistant"):
        show(past)

question = st.chat_input("Ask about the policy, e.g. “Does a CT for a CTAS 5 patient need prior authorisation?”")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if not index.chunks:
            st.error("Upload a document first.")
        else:
            with st.spinner("Searching the policy and writing the answer…"):
                hits = search(index, question, k=k, doc_ids=set(selected) or None)
                result = answer(question, hits, model=model)
            show(result)
            st.session_state.history.append(result)
