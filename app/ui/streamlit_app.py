"""ContextIQ Streamlit UI.

The UI talks to the FastAPI backend over HTTP only (see ``app.ui.api_client``); it never
imports the ingestion, retrieval or generation pipeline. Run it with::

    streamlit run app/ui/streamlit_app.py

Layout: a sidebar for connection settings and retrieval options, a left column that manages the
knowledge base (upload, index, list, delete, filter) and a right column with a chat that shows
grounded answers with verified citations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ``streamlit run app/ui/streamlit_app.py`` executes this file with *its own directory* as
# sys.path[0], so the ``app`` package is only importable if the repo root is on the path.
# Adding it here keeps the UI runnable from any working directory without an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.config import Settings
from app.models.schemas import (
    Citation,
    DocumentInfo,
    DocumentStatus,
    DocumentUploadResult,
    HealthResponse,
    QueryResponse,
    RetrievedChunk,
    UploadOutcome,
)
from app.ui import api_client
from app.ui.api_client import ApiError, ContextIQClient

RETRIEVAL_MODES = ("default", "dense", "hybrid")
OUTCOME_BADGES: dict[UploadOutcome, str] = {
    UploadOutcome.INDEXED: ":green-badge[indexed]",
    UploadOutcome.DUPLICATE: ":orange-badge[duplicate]",
    UploadOutcome.FAILED: ":red-badge[failed]",
}
STATUS_BADGES: dict[DocumentStatus, str] = {
    DocumentStatus.INDEXED: ":green-badge[indexed]",
    DocumentStatus.PROCESSING: ":blue-badge[processing]",
    DocumentStatus.FAILED: ":red-badge[failed]",
}
TABLE_WIDTHS = [3, 1, 1, 1.4, 1.8, 1.2]
TEXT_PREVIEW_CHARS = 160
EXTRACTIVE_WARNING = (
    "No LLM API key is configured on the backend, so answers are verbatim excerpts from the "
    "top-ranked passages (extractive mode). Set ANTHROPIC_API_KEY, OPENAI_API_KEY or "
    "GROQ_API_KEY for the API to get synthesized answers."
)


@dataclass(frozen=True)
class SidebarOptions:
    """Values chosen in the sidebar for the current run."""

    base_url: str
    api_key: str
    top_k: int
    mode: str | None
    debug: bool


# --------------------------------------------------------------------------------------------
# Pure helpers (no Streamlit calls; unit-tested directly)
# --------------------------------------------------------------------------------------------


def citation_title(citation: Citation) -> str:
    """Expander title for a citation, e.g. ``[S1] handbook.pdf — page 3``."""
    return f"[{citation.citation_id}] {citation.filename} — page {citation.page_number}"


def outcome_line(result: DocumentUploadResult) -> str:
    """One Markdown line with a badge, the filename and the outcome reason."""
    badge = OUTCOME_BADGES[result.outcome]
    reason = result.message or (result.document.error if result.document else None) or ""
    line = f"{badge} **{result.filename}**"
    if reason:
        line += f" — {reason}"
    if result.processing_ms is not None:
        line += f" ({result.processing_ms:.0f} ms)"
    return line


def format_added(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def document_rows(documents: list[DocumentInfo]) -> list[dict[str, Any]]:
    """Rows for the documents table: filename, pages, chunks, status, added."""
    return [
        {
            "filename": doc.filename,
            "pages": doc.page_count,
            "chunks": doc.chunk_count,
            "status": doc.status.value,
            "added": format_added(doc.created_at),
        }
        for doc in documents
    ]


def retrieved_frame(chunks: list[RetrievedChunk]) -> pd.DataFrame:
    """Debug table of retrieved chunks with their (uncalibrated) ranking signals."""
    columns = [
        "rank",
        "filename",
        "page",
        "score",
        "dense_score",
        "sparse_rank",
        "rerank_score",
        "text",
    ]
    rows = [
        {
            "rank": rank,
            "filename": chunk.filename,
            "page": chunk.page_number,
            "score": round(chunk.score, 4),
            "dense_score": None if chunk.dense_score is None else round(chunk.dense_score, 4),
            "sparse_rank": chunk.sparse_rank,
            "rerank_score": None if chunk.rerank_score is None else round(chunk.rerank_score, 4),
            "text": _preview(chunk.text),
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]
    return pd.DataFrame(rows, columns=columns)


def timings_caption(meta: dict[str, Any]) -> str:
    """Caption summarising latency, answer mode and model for one answer."""
    model = meta.get("model") or "n/a"
    return (
        f"retrieval {meta['retrieval_ms']:.0f} ms · generation {meta['generation_ms']:.0f} ms"
        f" · mode: {meta['answer_mode']} · model: {model}"
    )


def message_from_response(response: QueryResponse) -> dict[str, Any]:
    """Convert a ``QueryResponse`` into the plain-dict chat message stored in session state."""
    retrieved = response.retrieved
    return {
        "role": "assistant",
        "content": response.answer,
        "citations": [c.model_dump(mode="json") for c in response.citations],
        "meta": {
            "abstained": response.abstained,
            "answer_mode": response.answer_mode.value,
            "model": response.model,
            "invalid_citation_ids": list(response.invalid_citation_ids),
            "retrieval_ms": response.timings.retrieval_ms,
            "generation_ms": response.timings.generation_ms,
            "retrieved": None
            if retrieved is None
            else [r.model_dump(mode="json") for r in retrieved],
        },
    }


def _preview(text: str, limit: int = TEXT_PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------------------------
# Client and sidebar
# --------------------------------------------------------------------------------------------


def get_client(base_url: str, api_key: str) -> ContextIQClient:
    """Return the client cached in session state for ``(base_url, api_key)``."""
    cache_key = (base_url, api_key)
    if st.session_state.get("client_key") != cache_key:
        previous = st.session_state.get("client")
        if previous is not None:
            previous.close()
        st.session_state["client"] = api_client.ContextIQClient(base_url, api_key)
        st.session_state["client_key"] = cache_key
    return st.session_state["client"]


def render_sidebar(settings: Settings) -> SidebarOptions:
    default_key = settings.ui_api_key.get_secret_value() if settings.ui_api_key else ""
    with st.sidebar:
        st.title("ContextIQ")
        st.caption("Document-grounded answers with verified citations.")
        base_url = st.text_input("Backend URL", value=settings.api_url, key="api_url")
        api_key = st.text_input("API key", value=default_key, type="password", key="api_key")
        top_k = st.slider("Top-k passages", min_value=1, max_value=20, value=5, key="top_k")
        mode = st.selectbox(
            "Retrieval mode",
            RETRIEVAL_MODES,
            key="mode",
            help="'default' uses the mode configured on the backend.",
        )
        debug = st.toggle("Show retrieval debug", key="debug")
    return SidebarOptions(
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        top_k=int(top_k),
        mode=None if mode == "default" else mode,
        debug=bool(debug),
    )


def render_connection_status(client: ContextIQClient) -> HealthResponse | None:
    """Show backend health in the sidebar; return ``None`` when it is unreachable."""
    with st.sidebar:
        st.divider()
        st.subheader("Connection")
        try:
            health = client.health()
        except ApiError as exc:
            st.error(f"Backend unreachable: {exc}")
            return None
        notify = st.success if health.status == "ok" else st.warning
        notify(f"Connected · status {health.status} · v{health.version}")
        st.caption(f"LLM: {health.llm_provider} / {health.llm_model or 'none'}")
        dimension = f" ({health.embedding_dimension}-d)" if health.embedding_dimension else ""
        st.caption(f"Embeddings: {health.embedding_model}{dimension}")
        reranker = "on" if health.reranker_enabled else "off"
        st.caption(f"Retrieval: {health.retrieval_mode.value} · reranker {reranker}")
        if health.llm_provider == "extractive":
            st.warning(EXTRACTIVE_WARNING)
    return health


def render_flash() -> None:
    """Show a one-shot message stored before a ``st.rerun()``."""
    flash = st.session_state.pop("flash", None)
    if flash:
        level, text = flash
        getattr(st, level)(text)


# --------------------------------------------------------------------------------------------
# Knowledge base (left column)
# --------------------------------------------------------------------------------------------


def render_knowledge_base(client: ContextIQClient) -> list[str]:
    """Upload, index, list, delete and filter documents; return selected document ids."""
    st.subheader("Knowledge base")
    render_flash()
    render_uploader(client)
    render_upload_summary()
    documents = fetch_documents(client)
    render_documents_table(client, documents)
    return render_document_filter(documents)


def render_uploader(client: ContextIQClient) -> None:
    generation = st.session_state.get("uploader_generation", 0)
    uploads = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True, key=f"uploader_{generation}"
    )
    if st.button("Index documents", type="primary", disabled=not uploads) and uploads:
        results = index_files(client, uploads)
        st.session_state["upload_results"] = [r.model_dump(mode="json") for r in results]
        st.session_state["uploader_generation"] = generation + 1
        st.rerun()


def index_files(client: ContextIQClient, uploads: list[Any]) -> list[DocumentUploadResult]:
    """Upload files one request at a time so per-file progress is visible."""
    results: list[DocumentUploadResult] = []
    total = len(uploads)
    progress = st.progress(0.0, text=f"Indexing 0/{total} files")
    for done, upload in enumerate(uploads, start=1):
        results.extend(index_one(client, upload))
        progress.progress(done / total, text=f"Indexed {done}/{total} files")
    return results


def index_one(client: ContextIQClient, upload: Any) -> list[DocumentUploadResult]:
    with st.status(f"Indexing {upload.name}…") as status:
        try:
            response = client.upload([(upload.name, upload.getvalue())])
        except ApiError as exc:
            status.update(label=f"{upload.name}: request failed", state="error")
            failed = DocumentUploadResult(
                filename=upload.name, outcome=UploadOutcome.FAILED, message=str(exc)
            )
            return [failed]
        outcomes = ", ".join(r.outcome.value for r in response.results)
        any_failed = any(r.outcome == UploadOutcome.FAILED for r in response.results)
        status.update(
            label=f"{upload.name}: {outcomes}", state="error" if any_failed else "complete"
        )
        return response.results


def render_upload_summary() -> None:
    raw_results = st.session_state.get("upload_results") or []
    if not raw_results:
        return
    with st.expander("Last indexing run", expanded=True):
        for raw in raw_results:
            st.markdown(outcome_line(DocumentUploadResult.model_validate(raw)))


def fetch_documents(client: ContextIQClient) -> list[DocumentInfo]:
    try:
        return client.list_documents().documents
    except ApiError as exc:
        st.error(f"Could not load documents: {exc}")
        return []


def render_documents_table(client: ContextIQClient, documents: list[DocumentInfo]) -> None:
    if not documents:
        st.info("No documents indexed yet. Upload one or more PDFs to get started.")
        return
    header = st.columns(TABLE_WIDTHS)
    for column, name in zip(
        header, ("File", "Pages", "Chunks", "Status", "Added", ""), strict=True
    ):
        column.caption(name)
    for doc, row in zip(documents, document_rows(documents), strict=True):
        cells = st.columns(TABLE_WIDTHS, vertical_alignment="center")
        cells[0].markdown(f"**{row['filename']}**")
        cells[1].write(row["pages"])
        cells[2].write(row["chunks"])
        cells[3].markdown(STATUS_BADGES[doc.status])
        cells[4].write(row["added"])
        with cells[5]:
            render_delete_button(client, doc)
        if doc.error:
            st.caption(f"Reason: {doc.error}")


def render_delete_button(client: ContextIQClient, doc: DocumentInfo) -> None:
    """Two-click delete: the first click arms the button, the second confirms."""
    armed = st.session_state.get("pending_delete") == doc.document_id
    label = "Confirm" if armed else "Delete"
    help_text = "Click again to permanently delete this document." if armed else "Delete document"
    clicked = st.button(
        label,
        key=f"delete_{doc.document_id}",
        type="primary" if armed else "secondary",
        help=help_text,
    )
    if not clicked:
        return
    if not armed:
        st.session_state["pending_delete"] = doc.document_id
        st.rerun()
    st.session_state.pop("pending_delete", None)
    delete_document(client, doc)
    st.rerun()


def delete_document(client: ContextIQClient, doc: DocumentInfo) -> None:
    try:
        response = client.delete_document(doc.document_id)
    except ApiError as exc:
        st.session_state["flash"] = ("error", f"Could not delete {doc.filename}: {exc}")
        return
    st.session_state["flash"] = (
        "success",
        f"Deleted {doc.filename} ({response.chunks_removed} chunks removed).",
    )


def render_document_filter(documents: list[DocumentInfo]) -> list[str]:
    options = {d.document_id: d.filename for d in documents if d.status == DocumentStatus.INDEXED}
    if not options:
        return []
    previous = st.session_state.get("selected_docs", [])
    st.session_state["selected_docs"] = [d for d in previous if d in options]
    return st.multiselect(
        "Restrict question to selected documents",
        options=list(options),
        format_func=lambda doc_id: options[doc_id],
        key="selected_docs",
        placeholder="All documents",
    )


# --------------------------------------------------------------------------------------------
# Chat (right column)
# --------------------------------------------------------------------------------------------


def render_chat(client: ContextIQClient, selected_ids: list[str], options: SidebarOptions) -> None:
    heading, action = st.columns([4, 1], vertical_alignment="bottom")
    heading.subheader("Ask")
    if action.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()
    if selected_ids:
        st.caption(f"Questions are restricted to {len(selected_ids)} selected document(s).")
    for message in st.session_state.setdefault("messages", []):
        render_message(message)
    question = st.chat_input("Ask a question about your documents")
    if question:
        handle_question(client, question, selected_ids, options)


def handle_question(
    client: ContextIQClient, question: str, selected_ids: list[str], options: SidebarOptions
) -> None:
    user_message = {"role": "user", "content": question, "citations": [], "meta": {}}
    st.session_state["messages"].append(user_message)
    render_message(user_message)
    try:
        with st.spinner("Retrieving passages and generating an answer…"):
            response = client.query(
                question,
                top_k=options.top_k,
                document_ids=selected_ids or None,
                mode=options.mode,
                include_debug=options.debug,
            )
    except ApiError as exc:
        st.error(f"Query failed: {exc}")
        return
    assistant_message = message_from_response(response)
    st.session_state["messages"].append(assistant_message)
    render_message(assistant_message)


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(message)


def render_answer(message: dict[str, Any]) -> None:
    meta = message["meta"]
    if meta.get("abstained"):
        st.info(message["content"])
    else:
        st.markdown(message["content"])
    for raw in message["citations"]:
        render_citation(Citation.model_validate(raw))
    invalid = meta.get("invalid_citation_ids") or []
    if invalid:
        st.caption(f"Dropped {len(invalid)} unverifiable citation label(s): {', '.join(invalid)}")
    st.caption(timings_caption(meta))
    retrieved = meta.get("retrieved")
    if retrieved:
        chunks = [RetrievedChunk.model_validate(r) for r in retrieved]
        st.dataframe(retrieved_frame(chunks), hide_index=True, width="stretch")
        st.caption("Scores are uncalibrated ranking signals, not probabilities.")


def render_citation(citation: Citation) -> None:
    with st.expander(citation_title(citation)):
        st.text(citation.excerpt)
        if citation.score is not None:
            st.caption(f"retrieval score (uncalibrated): {citation.score:.3f}")


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="ContextIQ", layout="wide")
    options = render_sidebar(Settings())
    client = get_client(options.base_url, options.api_key)
    if render_connection_status(client) is None:
        st.warning(
            "The backend is not reachable. Check the backend URL in the sidebar and make sure "
            "the ContextIQ API is running."
        )
        st.stop()
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        selected_ids = render_knowledge_base(client)
    with right:
        render_chat(client, selected_ids, options)


if __name__ == "__main__":
    main()
