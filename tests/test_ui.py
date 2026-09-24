"""Tests for the Streamlit UI (``app.ui.streamlit_app``) and its HTTP client.

The app is driven with ``streamlit.testing.v1.AppTest``; the backend is replaced by a fake
client patched onto ``app.ui.api_client`` so no server or network is involved. The client itself
is exercised against ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from app.models.schemas import (
    AnswerMode,
    Citation,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentStatus,
    DocumentUploadResponse,
    DocumentUploadResult,
    HealthResponse,
    QueryResponse,
    RetrievalMode,
    RetrievedChunk,
    Timings,
    UploadOutcome,
)
from app.ui import api_client, streamlit_app
from app.ui.api_client import ApiError, ContextIQClient

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py"
DOC_ID = "a" * 32
QUESTION = "What is the PTO policy?"
ANSWER = "Employees accrue 20 days of paid time off per year [S1]."


# --------------------------------------------------------------------------------------------
# Canned responses
# --------------------------------------------------------------------------------------------


def make_health(provider: str = "anthropic", model: str | None = "claude-opus-5") -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.1.0",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        llm_provider=provider,
        llm_model=model,
        retrieval_mode=RetrievalMode.HYBRID,
        reranker_enabled=False,
        auth_mode="api_key",
    )


def make_document(**overrides: Any) -> DocumentInfo:
    fields: dict[str, Any] = {
        "document_id": DOC_ID,
        "workspace_id": "alpha",
        "filename": "handbook.pdf",
        "sha256": "f" * 64,
        "size_bytes": 1234,
        "page_count": 12,
        "chunk_count": 40,
        "status": DocumentStatus.INDEXED,
        "created_at": datetime(2026, 9, 23, 10, 30, tzinfo=UTC),
    }
    fields.update(overrides)
    return DocumentInfo(**fields)


def make_citation(score: float | None = 0.8123) -> Citation:
    return Citation(
        citation_id="S1",
        chunk_id=f"{DOC_ID}:p3:c7",
        document_id=DOC_ID,
        filename="handbook.pdf",
        page_number=3,
        excerpt="Full-time employees accrue 20 days of PTO per calendar year.",
        score=score,
    )


def make_retrieved() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{DOC_ID}:p3:c7",
        document_id=DOC_ID,
        filename="handbook.pdf",
        page_number=3,
        score=0.8123,
        dense_score=0.79,
        sparse_rank=1,
        text="Full-time employees accrue 20 days of PTO per calendar year. " * 5,
    )


def make_query_response(**overrides: Any) -> QueryResponse:
    fields: dict[str, Any] = {
        "request_id": "req-1",
        "workspace_id": "alpha",
        "question": QUESTION,
        "answer": ANSWER,
        "abstained": False,
        "answer_mode": AnswerMode.LLM,
        "citations": [make_citation()],
        "invalid_citation_ids": [],
        "retrieved": None,
        "model": "claude-opus-5",
        "timings": Timings(retrieval_ms=12.5, generation_ms=840.0, total_ms=852.5),
    }
    fields.update(overrides)
    return QueryResponse(**fields)


# --------------------------------------------------------------------------------------------
# Fake client
# --------------------------------------------------------------------------------------------


class FakeClient:
    """Stand-in for ``ContextIQClient`` returning canned pydantic responses."""

    health_response: ClassVar[HealthResponse] = make_health()
    documents: ClassVar[list[DocumentInfo]] = [make_document()]
    query_response: ClassVar[QueryResponse] = make_query_response()
    query_error: ClassVar[ApiError | None] = None
    health_error: ClassVar[ApiError | None] = None
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def close(self) -> None:
        return None

    def health(self) -> HealthResponse:
        self.calls.append(("health", {}))
        if self.health_error is not None:
            raise self.health_error
        return self.health_response

    def list_documents(self) -> DocumentListResponse:
        self.calls.append(("list_documents", {}))
        return DocumentListResponse(
            workspace_id="alpha", documents=list(self.documents), total=len(self.documents)
        )

    def upload(self, files: list[tuple[str, bytes]]) -> DocumentUploadResponse:
        self.calls.append(("upload", {"names": [name for name, _ in files]}))
        results = [
            DocumentUploadResult(
                filename=name, outcome=UploadOutcome.INDEXED, message="Indexed", processing_ms=5.0
            )
            for name, _ in files
        ]
        return DocumentUploadResponse(request_id="req-u", results=results)

    def delete_document(self, document_id: str) -> DeleteDocumentResponse:
        self.calls.append(("delete_document", {"document_id": document_id}))
        type(self).documents = [d for d in self.documents if d.document_id != document_id]
        return DeleteDocumentResponse(
            request_id="req-d", document_id=document_id, deleted=True, chunks_removed=40
        )

    def query(self, question: str, **kwargs: Any) -> QueryResponse:
        self.calls.append(("query", {"question": question, **kwargs}))
        if self.query_error is not None:
            raise self.query_error
        return self.query_response

    def search(self, query: str, **kwargs: Any) -> Any:
        raise AssertionError("search is not used by the UI")


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeClient]:
    """Patch the client class the app instantiates and reset the fake's canned state."""
    monkeypatch.setattr(api_client, "ContextIQClient", FakeClient)
    monkeypatch.setenv("CONTEXTIQ_API_URL", "http://fake-backend:8000")
    monkeypatch.setenv("CONTEXTIQ_UI_API_KEY", "dev-key-alpha-0001")
    FakeClient.health_response = make_health()
    FakeClient.documents = [make_document()]
    FakeClient.query_response = make_query_response()
    FakeClient.query_error = None
    FakeClient.health_error = None
    FakeClient.calls = []
    return FakeClient


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    return at


def all_text(at: AppTest) -> str:
    """Concatenate the visible text of the common element types for loose assertions."""
    parts: list[str] = []
    for kind in ("markdown", "caption", "text", "info", "warning", "error", "success"):
        parts.extend(str(el.value) for el in getattr(at, kind))
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------
# AppTest: rendering
# --------------------------------------------------------------------------------------------


def test_app_renders_health_and_documents(fake_client: type[FakeClient]) -> None:
    at = run_app()

    assert not at.exception
    sidebar_text = "\n".join(c.value for c in at.sidebar.caption)
    assert "anthropic / claude-opus-5" in sidebar_text
    assert "BAAI/bge-small-en-v1.5 (384-d)" in sidebar_text
    assert at.sidebar.success[0].value.startswith("Connected")
    assert not at.sidebar.warning
    assert "**handbook.pdf**" in [m.value for m in at.markdown]
    assert ":green-badge[indexed]" in [m.value for m in at.markdown]
    assert "2026-09-23 10:30" in all_text(at)
    assert at.button(key=f"delete_{DOC_ID}").label == "Delete"
    assert at.multiselect[0].label == "Restrict question to selected documents"
    assert at.sidebar.text_input(key="api_key").value == "dev-key-alpha-0001"
    assert at.sidebar.text_input(key="api_url").value == "http://fake-backend:8000"


def test_extractive_mode_shows_warning(fake_client: type[FakeClient]) -> None:
    fake_client.health_response = make_health(provider="extractive", model=None)
    at = run_app()

    assert not at.exception
    warnings = [w.value for w in at.sidebar.warning]
    assert any("extractive mode" in w for w in warnings)
    assert "extractive / none" in "\n".join(c.value for c in at.sidebar.caption)


def test_empty_knowledge_base_shows_hint(fake_client: type[FakeClient]) -> None:
    fake_client.documents = []
    at = run_app()

    assert not at.exception
    assert any("No documents indexed yet" in i.value for i in at.info)
    assert not at.multiselect


def test_failed_document_shows_reason(fake_client: type[FakeClient]) -> None:
    fake_client.documents = [
        make_document(status=DocumentStatus.FAILED, error="encrypted PDF", chunk_count=0)
    ]
    at = run_app()

    assert not at.exception
    assert ":red-badge[failed]" in [m.value for m in at.markdown]
    assert "Reason: encrypted PDF" in [c.value for c in at.caption]


def test_unreachable_backend_shows_error_not_traceback(fake_client: type[FakeClient]) -> None:
    fake_client.health_error = ApiError(None, "Could not reach the ContextIQ API (ConnectError).")
    at = run_app()

    assert not at.exception
    assert any("Backend unreachable" in e.value for e in at.sidebar.error)
    assert any("not reachable" in w.value for w in at.warning)
    assert not at.chat_input


# --------------------------------------------------------------------------------------------
# AppTest: chat
# --------------------------------------------------------------------------------------------


def test_chat_shows_answer_and_citation(fake_client: type[FakeClient]) -> None:
    at = run_app()
    at.chat_input[0].set_value(QUESTION).run()

    assert not at.exception
    roles = [m.name for m in at.chat_message]
    assert roles == ["user", "assistant"]
    markdown = [m.value for m in at.markdown]
    assert QUESTION in markdown
    assert ANSWER in markdown
    assert [e.label for e in at.expander] == ["[S1] handbook.pdf — page 3"]
    assert "Full-time employees accrue 20 days" in "\n".join(t.value for t in at.text)
    captions = [c.value for c in at.caption]
    assert "retrieval score (uncalibrated): 0.812" in captions
    assert any("retrieval 12 ms · generation 840 ms" in c for c in captions)
    assert any("mode: llm · model: claude-opus-5" in c for c in captions)
    assert not at.dataframe
    query_calls = [kw for name, kw in fake_client.calls if name == "query"]
    assert query_calls == [
        {
            "question": QUESTION,
            "top_k": 5,
            "document_ids": None,
            "mode": None,
            "include_debug": False,
        }
    ]


def test_chat_history_persists_across_reruns(fake_client: type[FakeClient]) -> None:
    at = run_app()
    at.chat_input[0].set_value(QUESTION).run()
    at.run()

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert len(at.session_state["messages"]) == 2


def test_abstained_answer_uses_info_callout_and_lists_dropped_labels(
    fake_client: type[FakeClient],
) -> None:
    no_answer = "I couldn't find enough information in the indexed documents to answer that."
    fake_client.query_response = make_query_response(
        answer=no_answer, abstained=True, citations=[], invalid_citation_ids=["S9", "S12"]
    )
    at = run_app()
    at.chat_input[0].set_value("Who is the CEO?").run()

    assert not at.exception
    assert no_answer in [i.value for i in at.info]
    assert no_answer not in [m.value for m in at.markdown]
    assert not at.expander
    assert any("Dropped 2 unverifiable citation label(s): S9, S12" in c.value for c in at.caption)


def test_debug_toggle_shows_retrieved_dataframe(fake_client: type[FakeClient]) -> None:
    fake_client.query_response = make_query_response(retrieved=[make_retrieved()])
    at = run_app()
    at.sidebar.toggle(key="debug").set_value(True)
    at.sidebar.slider(key="top_k").set_value(8)
    at.sidebar.selectbox(key="mode").set_value("dense")
    at.run()
    at.chat_input[0].set_value(QUESTION).run()

    assert not at.exception
    assert len(at.dataframe) == 1
    frame = at.dataframe[0].value
    assert list(frame.columns) == [
        "rank",
        "filename",
        "page",
        "score",
        "dense_score",
        "sparse_rank",
        "rerank_score",
        "text",
    ]
    assert frame.iloc[0]["filename"] == "handbook.pdf"
    query_call = next(kw for name, kw in fake_client.calls if name == "query")
    assert query_call["include_debug"] is True
    assert query_call["top_k"] == 8
    assert query_call["mode"] == "dense"


def test_query_api_error_shows_error_without_traceback(fake_client: type[FakeClient]) -> None:
    fake_client.query_error = ApiError(502, "LLM call failed", request_id="req-x")
    at = run_app()
    at.chat_input[0].set_value(QUESTION).run()

    assert not at.exception
    errors = [e.value for e in at.error]
    assert errors == ["Query failed: HTTP 502: LLM call failed (request id req-x)"]
    assert [m.name for m in at.chat_message] == ["user"]


def test_clear_chat_button_resets_history(fake_client: type[FakeClient]) -> None:
    at = run_app()
    at.chat_input[0].set_value(QUESTION).run()
    assert len(at.chat_message) == 2

    clear = next(b for b in at.button if b.label == "Clear chat")
    clear.click().run()

    assert not at.exception
    assert not at.chat_message
    assert at.session_state["messages"] == []


def test_restricting_to_selected_documents_passes_ids(fake_client: type[FakeClient]) -> None:
    at = run_app()
    at.multiselect[0].select(DOC_ID).run()
    at.chat_input[0].set_value(QUESTION).run()

    assert not at.exception
    assert any("restricted to 1 selected document" in c.value for c in at.caption)
    query_call = next(kw for name, kw in fake_client.calls if name == "query")
    assert query_call["document_ids"] == [DOC_ID]


# --------------------------------------------------------------------------------------------
# AppTest: delete flow
# --------------------------------------------------------------------------------------------


def test_delete_requires_second_click(fake_client: type[FakeClient]) -> None:
    at = run_app()
    at.button(key=f"delete_{DOC_ID}").click().run()

    assert not at.exception
    assert at.button(key=f"delete_{DOC_ID}").label == "Confirm"
    assert not [c for c in fake_client.calls if c[0] == "delete_document"]

    at.button(key=f"delete_{DOC_ID}").click().run()

    assert not at.exception
    assert ("delete_document", {"document_id": DOC_ID}) in fake_client.calls
    assert fake_client.documents == []
    assert any("Deleted handbook.pdf (40 chunks removed)" in s.value for s in at.success)
    assert any("No documents indexed yet" in i.value for i in at.info)


def test_client_is_cached_per_url_and_key(fake_client: type[FakeClient]) -> None:
    at = run_app()
    first = at.session_state["client"]
    at.run()
    assert at.session_state["client"] is first

    at.sidebar.text_input(key="api_key").set_value("dev-key-beta-0002").run()

    assert not at.exception
    assert at.session_state["client"] is not first
    assert at.session_state["client"].api_key == "dev-key-beta-0002"


# --------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------


def test_citation_title_and_document_rows() -> None:
    assert streamlit_app.citation_title(make_citation()) == "[S1] handbook.pdf — page 3"
    rows = streamlit_app.document_rows([make_document()])
    assert rows == [
        {
            "filename": "handbook.pdf",
            "pages": 12,
            "chunks": 40,
            "status": "indexed",
            "added": "2026-09-23 10:30",
        }
    ]


def test_outcome_line_badges() -> None:
    indexed = DocumentUploadResult(
        filename="a.pdf", outcome=UploadOutcome.INDEXED, message="Indexed", processing_ms=12.4
    )
    duplicate = DocumentUploadResult(
        filename="b.pdf",
        outcome=UploadOutcome.DUPLICATE,
        message="Already indexed",
        duplicate_of=DOC_ID,
    )
    failed = DocumentUploadResult(
        filename="c.pdf",
        outcome=UploadOutcome.FAILED,
        message="",
        document=make_document(status=DocumentStatus.FAILED, error="no extractable text"),
    )
    assert (
        streamlit_app.outcome_line(indexed) == ":green-badge[indexed] **a.pdf** — Indexed (12 ms)"
    )
    assert (
        streamlit_app.outcome_line(duplicate)
        == ":orange-badge[duplicate] **b.pdf** — Already indexed"
    )
    assert (
        streamlit_app.outcome_line(failed) == ":red-badge[failed] **c.pdf** — no extractable text"
    )


def test_message_from_response_and_timings_caption() -> None:
    message = streamlit_app.message_from_response(
        make_query_response(retrieved=[make_retrieved()], invalid_citation_ids=["S4"])
    )
    assert message["role"] == "assistant"
    assert message["content"] == ANSWER
    assert message["citations"][0]["citation_id"] == "S1"
    assert message["meta"]["invalid_citation_ids"] == ["S4"]
    assert len(message["meta"]["retrieved"]) == 1
    assert (
        streamlit_app.timings_caption(message["meta"])
        == "retrieval 12 ms · generation 840 ms · mode: llm · model: claude-opus-5"
    )
    assert streamlit_app.timings_caption({**message["meta"], "model": None}).endswith("model: n/a")


def test_retrieved_frame_truncates_text() -> None:
    frame = streamlit_app.retrieved_frame([make_retrieved()])
    assert len(frame) == 1
    assert frame.iloc[0]["rank"] == 1
    assert frame.iloc[0]["score"] == 0.8123
    assert len(frame.iloc[0]["text"]) <= streamlit_app.TEXT_PREVIEW_CHARS
    assert frame.iloc[0]["text"].endswith("…")


# --------------------------------------------------------------------------------------------
# HTTP client against httpx.MockTransport
# --------------------------------------------------------------------------------------------


def make_client(handler: Any) -> ContextIQClient:
    return ContextIQClient(
        "http://testserver/", "secret-key-000", transport=httpx.MockTransport(handler)
    )


def test_client_sends_api_key_and_parses_health() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("X-API-Key")
        return httpx.Response(200, json=make_health().model_dump(mode="json"))

    with make_client(handler) as client:
        health = client.health()

    assert seen == {"path": "/health", "key": "secret-key-000"}
    assert health.llm_provider == "anthropic"
    assert "secret-key-000" not in repr(client)


def test_client_upload_sends_repeated_files_parts() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = request.read()
        results = [
            DocumentUploadResult(filename="a.pdf", outcome=UploadOutcome.INDEXED, message="ok"),
            DocumentUploadResult(filename="b.pdf", outcome=UploadOutcome.DUPLICATE, message="dup"),
        ]
        response = DocumentUploadResponse(request_id="r", results=results)
        return httpx.Response(201, json=response.model_dump(mode="json"))

    with make_client(handler) as client:
        response = client.upload([("a.pdf", b"%PDF-1.4 a"), ("b.pdf", b"%PDF-1.4 b")])

    assert seen["content_type"].startswith("multipart/form-data")
    assert seen["body"].count(b'name="files"') == 2
    assert b'filename="a.pdf"' in seen["body"] and b'filename="b.pdf"' in seen["body"]
    assert response.indexed_count == 1
    with pytest.raises(ValueError, match="must not be empty"), make_client(handler) as client:
        client.upload([])


def test_client_query_body_and_search() -> None:
    bodies: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        bodies[request.url.path] = json.loads(request.read())
        if request.url.path == "/query":
            return httpx.Response(200, json=make_query_response().model_dump(mode="json"))
        payload = {
            "request_id": "r",
            "workspace_id": "alpha",
            "query": "pto",
            "mode": "dense",
            "results": [make_retrieved().model_dump(mode="json")],
            "timings": {"retrieval_ms": 3.0, "generation_ms": 0.0, "total_ms": 3.0},
        }
        return httpx.Response(200, json=payload)

    with make_client(handler) as client:
        answer = client.query(QUESTION, top_k=3, document_ids=[], mode="hybrid", include_debug=True)
        search = client.search("pto", mode="dense")

    assert bodies["/query"] == {
        "question": QUESTION,
        "top_k": 3,
        "mode": "hybrid",
        "include_debug": True,
    }
    assert bodies["/search"] == {"query": "pto", "mode": "dense"}
    assert answer.citations[0].citation_id == "S1"
    assert search.results[0].page_number == 3


def test_client_rejects_invalid_request_before_sending() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request must not be sent")

    with make_client(handler) as client, pytest.raises(ApiError) as excinfo:
        client.query("hi")

    assert excinfo.value.status_code is None
    assert "question" in excinfo.value.message


def test_client_maps_error_response_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"request_id": "req-9", "error": "unauthorized", "detail": "invalid API key"}
        return httpx.Response(401, json=body)

    with make_client(handler) as client, pytest.raises(ApiError) as excinfo:
        client.list_documents()

    error = excinfo.value
    assert (error.status_code, error.message, error.request_id) == (
        401,
        "unauthorized: invalid API key",
        "req-9",
    )
    assert str(error) == "HTTP 401: unauthorized: invalid API key (request id req-9)"


def test_client_maps_fastapi_detail_and_plain_text_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/documents/missing":
            return httpx.Response(404, json={"detail": "document not found"})
        if request.url.path == "/documents":
            detail = [{"loc": ["body", "files"], "msg": "field required", "type": "missing"}]
            return httpx.Response(422, json={"detail": detail})
        return httpx.Response(503, text="upstream down", headers={"X-Request-ID": "rid-1"})

    with make_client(handler) as client:
        with pytest.raises(ApiError) as not_found:
            client.delete_document("missing")
        with pytest.raises(ApiError) as invalid:
            client.upload([("a.pdf", b"x")])
        with pytest.raises(ApiError) as plain:
            client.health()

    assert (not_found.value.status_code, not_found.value.message) == (404, "document not found")
    assert (invalid.value.status_code, invalid.value.message) == (422, "field required")
    assert (plain.value.status_code, plain.value.message, plain.value.request_id) == (
        503,
        "upstream down",
        "rid-1",
    )


def test_client_wraps_transport_errors_and_bad_bodies() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(refuse) as client, pytest.raises(ApiError) as excinfo:
        client.health()
    assert excinfo.value.status_code is None
    assert "Could not reach the ContextIQ API at http://testserver" in excinfo.value.message
    assert "ConnectError" in excinfo.value.message

    def garbage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with make_client(garbage) as client, pytest.raises(ApiError) as bad_body:
        client.health()
    assert bad_body.value.status_code == 200
    assert "Unexpected response body from /health" in bad_body.value.message


# --------------------------------------------------------------------------------------------
# Indexing flow (the uploader widget cannot be driven by AppTest, so call the helpers directly)
# --------------------------------------------------------------------------------------------


class FakeUpload:
    """Minimal stand-in for ``st.runtime.uploaded_file_manager.UploadedFile``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def getvalue(self) -> bytes:
        return b"%PDF-1.4 " + self.name.encode()


class FailingUploadClient(FakeClient):
    def upload(self, files: list[tuple[str, bytes]]) -> DocumentUploadResponse:
        raise ApiError(413, "file too large", request_id="req-big")


def _indexing_script() -> None:
    import streamlit as st

    from app.ui import streamlit_app
    from tests.test_ui import FailingUploadClient, FakeClient, FakeUpload

    client = (
        FailingUploadClient("http://x", "k")
        if st.session_state.get("fail")
        else FakeClient("http://x", "k")
    )
    results = streamlit_app.index_files(client, [FakeUpload("a.pdf"), FakeUpload("b.pdf")])
    st.session_state["upload_results"] = [r.model_dump(mode="json") for r in results]
    streamlit_app.render_upload_summary()


def test_index_files_reports_per_file_status(fake_client: type[FakeClient]) -> None:
    at = AppTest.from_function(_indexing_script, default_timeout=60)
    at.run()

    assert not at.exception
    assert [(s.label, s.state) for s in at.status] == [
        ("a.pdf: indexed", "complete"),
        ("b.pdf: indexed", "complete"),
    ]
    assert [kw["names"] for name, kw in fake_client.calls if name == "upload"] == [
        ["a.pdf"],
        ["b.pdf"],
    ]
    assert [e.label for e in at.expander] == ["Last indexing run"]
    assert [m.value for m in at.markdown] == [
        ":green-badge[indexed] **a.pdf** — Indexed (5 ms)",
        ":green-badge[indexed] **b.pdf** — Indexed (5 ms)",
    ]


def test_index_files_marks_failed_uploads(fake_client: type[FakeClient]) -> None:
    at = AppTest.from_function(_indexing_script, default_timeout=60)
    at.session_state["fail"] = True
    at.run()

    assert not at.exception
    assert [(s.label, s.state) for s in at.status] == [
        ("a.pdf: request failed", "error"),
        ("b.pdf: request failed", "error"),
    ]
    lines = [m.value for m in at.markdown]
    assert len(lines) == 2
    assert lines[0].startswith(":red-badge[failed] **a.pdf** — HTTP 413: file too large")
