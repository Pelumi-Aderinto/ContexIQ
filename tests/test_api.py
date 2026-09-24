"""HTTP-level tests for the ContextIQ FastAPI application.

Every app is built with ``create_app(settings, embedder=HashingEmbedder(64), llm=<fake>)`` and
driven through ``TestClient`` as a context manager so the lifespan runs (services built on
entry, store closed on exit). No real LLM, embedding model or network is ever touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field

from app import __version__
from app.api.deps import Services
from app.api.main import API_TITLE, build_services, create_app
from app.api.main import app as module_app
from app.api.routes.documents import _safe_filename
from app.core.config import Settings
from app.core.logging import REQUEST_ID_HEADER
from app.generation.prompts import NO_ANSWER_TEXT
from app.models.schemas import ErrorResponse, QueryResponse, SearchResponse
from app.retrieval.embeddings import HashingEmbedder

ALPHA_KEY = "test-key-alpha-000"
BETA_KEY = "test-key-beta-0000"
ALPHA = {"X-API-Key": ALPHA_KEY}
BETA = {"X-API-Key": BETA_KEY}
WRONG = {"X-API-Key": "definitely-not-a-key"}

# Settings overrides that make the app report an LLM without ever building a real one.
LLM_SETTINGS: dict[str, Any] = {"llm_provider": "openai", "llm_model": "fake-model"}

POLICY_PAGES = [
    "Security policy. Passwords must be at least 12 characters long and include a number. "
    "Passwords are rotated every 90 days and may not be reused for one year.",
    "Remote access. Employees connect through the corporate VPN using hardware tokens. "
    "Shared accounts are prohibited and every login is audited quarterly.",
]
LEAVE_PAGES = [
    "Leave policy. Full-time employees accrue 25 days of paid time off per year, plus public "
    "holidays. Unused days carry over up to a maximum of 5 days into the next year.",
    "Parental leave. Primary caregivers receive 16 weeks of paid parental leave. Requests must "
    "be submitted at least 30 days in advance to the people team.",
]
GARBAGE = b"this is definitely not a pdf file"

MakePdf = Callable[..., bytes]
ClientFactory = Callable[..., TestClient]


# --------------------------------------------------------------------------------------------
# Fakes and helpers
# --------------------------------------------------------------------------------------------


class CountingFakeLLM(FakeListChatModel):
    """Fake chat model that counts invocations (``FakeListChatModel`` alone does not)."""

    calls: int = 0
    prompts: list[str] = Field(default_factory=list)

    def _call(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return super()._call(messages, *args, **kwargs)


class RaisingFakeLLM(FakeListChatModel):
    """Fake chat model whose generation always fails, simulating a provider outage."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("provider timed out")


def llm_json(answer: str, citations: list[str], *, insufficient: bool = False) -> str:
    return json.dumps(
        {"answer": answer, "citations": citations, "insufficient_evidence": insufficient}
    )


def fake_llm(answer: str, citations: list[str], *, insufficient: bool = False) -> CountingFakeLLM:
    return CountingFakeLLM(responses=[llm_json(answer, citations, insufficient=insufficient)])


def pdf_part(name: str, data: bytes) -> tuple[str, tuple[str, bytes, str]]:
    """One ``files`` entry for a multipart upload."""
    return ("files", (name, data, "application/pdf"))


def upload(client: TestClient, headers: dict[str, str], *parts: tuple[str, Any]) -> httpx.Response:
    return client.post("/documents", headers=headers, files=list(parts))


def upload_one(
    client: TestClient, headers: dict[str, str], make_pdf: MakePdf, pages: list[str], name: str
) -> dict[str, Any]:
    """Upload a freshly built PDF, assert it was indexed and return its ``DocumentInfo`` dict."""
    response = upload(client, headers, pdf_part(name, make_pdf(pages)))
    assert response.status_code == 201, response.text
    result = response.json()["results"][0]
    assert result["outcome"] == "indexed", result
    return result["document"]


def assert_error_body(response: httpx.Response) -> ErrorResponse:
    """The body validates as ``ErrorResponse`` and its request id matches the header."""
    body = ErrorResponse.model_validate(response.json())
    assert body.request_id
    assert body.request_id == response.headers[REQUEST_ID_HEADER]
    return body


def services_of(client: TestClient) -> Services:
    app = client.app
    assert isinstance(app, FastAPI)
    return app.state.services


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture
def make_client(settings: Settings) -> Iterator[ClientFactory]:
    """Factory: ``make_client(llm=None, *, raise_server_exceptions=True, **overrides)``.

    Builds an app over the shared test settings (small chunks so short PDFs yield several
    chunks) with a ``HashingEmbedder`` and the given LLM, and enters ``TestClient`` so the
    lifespan runs. Every client is closed at teardown.
    """
    with ExitStack() as stack:

        def _make(
            llm: BaseChatModel | None = None,
            *,
            raise_server_exceptions: bool = True,
            **overrides: Any,
        ) -> TestClient:
            app_settings = settings.model_copy(
                update={"chunk_size": 200, "chunk_overlap": 20, **overrides}
            )
            app = create_app(app_settings, embedder=HashingEmbedder(64), llm=llm)
            return stack.enter_context(
                TestClient(app, raise_server_exceptions=raise_server_exceptions)
            )

        yield _make


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    """Extractive-mode client (no LLM)."""
    return make_client(None)


# --------------------------------------------------------------------------------------------
# Module-level app, build_services, health
# --------------------------------------------------------------------------------------------


def test_module_level_app_is_configured() -> None:
    assert isinstance(module_app, FastAPI)
    assert module_app.title == API_TITLE
    assert module_app.version == __version__


def test_build_services_creates_dirs_and_wires_components(settings: Settings) -> None:
    services = build_services(settings, embedder=HashingEmbedder(64), llm=None)
    try:
        assert settings.index_dir.is_dir()
        assert settings.db_path.exists()
        assert services.settings is settings
        assert services.embedder.dimension == services.vector_store.dimension == 64
        assert services.llm_model_name is None
        assert services.retrieval.reranker is None
    finally:
        services.store.close()


def test_build_services_auto_llm_resolves_extractive(settings: Settings) -> None:
    services = build_services(settings, embedder=HashingEmbedder(64), llm="auto")
    try:
        assert services.llm_model_name is None
    finally:
        services.store.close()


def test_build_services_rejects_unknown_llm_option(settings: Settings) -> None:
    with pytest.raises(ValueError, match="'auto'"):
        build_services(settings, embedder=HashingEmbedder(64), llm="bogus")  # type: ignore[arg-type]


def test_health_needs_no_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["embedding_model"] == "hashing-test-embedder"
    assert body["embedding_dimension"] == 64
    assert body["llm_provider"] == "extractive"
    assert body["llm_model"] is None
    assert body["retrieval_mode"] == "hybrid"
    assert body["reranker_enabled"] is False
    assert body["auth_mode"] == "api_key"
    assert response.headers[REQUEST_ID_HEADER]


def test_health_reports_configured_llm(make_client: ClientFactory) -> None:
    client = make_client(fake_llm("x", ["S1"]), **LLM_SETTINGS)
    body = client.get("/health").json()
    assert body["llm_provider"] == "openai"
    assert body["llm_model"] == "fake-model"


def test_health_reports_custom_provider_for_injected_model(make_client: ClientFactory) -> None:
    client = make_client(fake_llm("x", ["S1"]))  # settings say extractive, but a model is wired
    body = client.get("/health").json()
    assert body["llm_provider"] == "custom"
    assert body["llm_model"] == "CountingFakeLLM"


def test_services_unavailable_before_startup(settings: Settings) -> None:
    app = create_app(settings, embedder=HashingEmbedder(64), llm=None)
    response = TestClient(app).get("/health")  # no context manager: lifespan never ran
    assert response.status_code == 503
    assert_error_body(response)


def test_openapi_schema_lists_every_endpoint(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/documents", "/documents/{document_id}", "/query", "/search"}


# --------------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------------

PROTECTED_ROUTES: list[tuple[str, str, dict[str, Any]]] = [
    ("GET", "/documents", {}),
    ("POST", "/documents", {"files": [pdf_part("a.pdf", b"%PDF-1.4 tiny")]}),
    ("GET", "/documents/abc", {}),
    ("DELETE", "/documents/abc", {}),
    ("POST", "/query", {"json": {"question": "What is the policy?"}}),
    ("POST", "/search", {"json": {"query": "policy"}}),
]


@pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED_ROUTES)
@pytest.mark.parametrize("headers", [{}, WRONG], ids=["missing", "wrong"])
def test_protected_routes_reject_missing_or_wrong_key(
    client: TestClient, method: str, path: str, kwargs: dict[str, Any], headers: dict[str, str]
) -> None:
    response = client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    body = assert_error_body(response)
    assert body.detail == "Invalid or missing API key"


def test_auth_disabled_maps_to_default_workspace(make_client: ClientFactory) -> None:
    client = make_client(None, auth_mode="disabled", default_workspace_id="solo")
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"workspace_id": "solo", "documents": [], "total": 0}


# --------------------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------------------


def test_upload_indexes_pdf_and_lists_it(client: TestClient, make_pdf: MakePdf) -> None:
    response = upload(client, ALPHA, pdf_part("policy.pdf", make_pdf(POLICY_PAGES)))
    assert response.status_code == 201
    body = response.json()
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    [result] = body["results"]
    assert result["outcome"] == "indexed"
    assert result["filename"] == "policy.pdf"
    assert result["processing_ms"] >= 0
    document = result["document"]
    assert document["status"] == "indexed"
    assert document["workspace_id"] == "alpha"
    assert document["page_count"] == 2
    assert document["chunk_count"] >= 2

    listing = client.get("/documents", headers=ALPHA).json()
    assert listing["workspace_id"] == "alpha"
    assert listing["total"] == 1
    assert listing["documents"][0]["document_id"] == document["document_id"]


def test_upload_same_bytes_again_is_duplicate(client: TestClient, make_pdf: MakePdf) -> None:
    data = make_pdf(POLICY_PAGES)
    first = upload(client, ALPHA, pdf_part("policy.pdf", data)).json()["results"][0]
    response = upload(client, ALPHA, pdf_part("renamed.pdf", data))
    assert response.status_code == 200
    [result] = response.json()["results"]
    assert result["outcome"] == "duplicate"
    assert result["duplicate_of"] == first["document"]["document_id"]
    assert client.get("/documents", headers=ALPHA).json()["total"] == 1


def test_upload_garbage_fails_with_visible_reason(client: TestClient) -> None:
    response = upload(client, ALPHA, pdf_part("junk.pdf", GARBAGE))
    assert response.status_code == 422
    [result] = response.json()["results"]
    assert result["outcome"] == "failed"
    assert "not_pdf" in result["message"]
    assert result["document"] is None
    assert client.get("/documents", headers=ALPHA).json()["total"] == 0


def test_upload_mixed_batch_reports_per_file_outcomes(
    client: TestClient, make_pdf: MakePdf
) -> None:
    response = upload(
        client,
        ALPHA,
        pdf_part("good.pdf", make_pdf(POLICY_PAGES)),
        pdf_part("bad.pdf", GARBAGE),
    )
    assert response.status_code == 201
    outcomes = {r["filename"]: r["outcome"] for r in response.json()["results"]}
    assert outcomes == {"good.pdf": "indexed", "bad.pdf": "failed"}


def test_upload_oversize_file_is_a_failed_outcome(make_client: ClientFactory) -> None:
    client = make_client(None, max_upload_mb=1)
    payload = b"%PDF-1.4\n" + b"0" * (1_500_000)
    response = upload(client, ALPHA, pdf_part("huge.pdf", payload))
    assert response.status_code == 422
    [result] = response.json()["results"]
    assert result["outcome"] == "failed"
    assert result["message"] == "File exceeds 1 MB limit"
    assert result["document"] is None
    assert client.get("/documents", headers=ALPHA).json()["total"] == 0


def test_upload_exactly_at_limit_is_accepted_by_size_check(make_client: ClientFactory) -> None:
    client = make_client(None, max_upload_mb=1)
    payload = b"%PDF-1.4\n" + b"0" * (1024 * 1024 - 9)  # exactly 1 MiB: not oversize
    [result] = upload(client, ALPHA, pdf_part("edge.pdf", payload)).json()["results"]
    assert result["outcome"] == "failed"
    assert "exceeds" not in result["message"]  # rejected by the parser, not by the size check


def test_upload_too_many_files(make_client: ClientFactory, make_pdf: MakePdf) -> None:
    client = make_client(None, max_files_per_upload=1)
    response = upload(
        client,
        ALPHA,
        pdf_part("a.pdf", make_pdf(POLICY_PAGES)),
        pdf_part("b.pdf", make_pdf(LEAVE_PAGES)),
    )
    assert response.status_code == 422
    body = assert_error_body(response)
    assert "Too many files" in (body.detail or "")
    assert client.get("/documents", headers=ALPHA).json()["total"] == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "upload.pdf"),
        ("", "upload.pdf"),
        ("   ", "upload.pdf"),
        ("..", "upload.pdf"),
        ("report.pdf", "report.pdf"),
        ("dir/sub/report.pdf", "report.pdf"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("x" * 300 + ".pdf", "x" * 255),
    ],
)
def test_safe_filename(raw: str | None, expected: str) -> None:
    # A part without a filename is parsed as a plain form field upstream, so the ``None``
    # fallback cannot be reached over HTTP; the helper is checked directly instead.
    assert _safe_filename(raw) == expected


def test_upload_strips_directories_from_filename(client: TestClient, make_pdf: MakePdf) -> None:
    response = upload(client, ALPHA, pdf_part("../../etc/passwd.pdf", make_pdf(POLICY_PAGES)))
    assert response.json()["results"][0]["filename"] == "passwd.pdf"


# --------------------------------------------------------------------------------------------
# Get / delete
# --------------------------------------------------------------------------------------------


def test_get_document_only_in_owning_workspace(client: TestClient, make_pdf: MakePdf) -> None:
    document = upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    path = f"/documents/{document['document_id']}"

    response = client.get(path, headers=ALPHA)
    assert response.status_code == 200
    assert response.json()["document_id"] == document["document_id"]

    foreign = client.get(path, headers=BETA)
    assert foreign.status_code == 404
    assert assert_error_body(foreign).detail == "Document not found"
    assert client.get("/documents/does-not-exist", headers=ALPHA).status_code == 404


def test_delete_document_removes_metadata_and_chunks(client: TestClient, make_pdf: MakePdf) -> None:
    document = upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    path = f"/documents/{document['document_id']}"
    assert client.post("/search", headers=ALPHA, json={"query": "password"}).json()["results"]

    response = client.delete(path, headers=ALPHA)
    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is True
    assert body["document_id"] == document["document_id"]
    assert body["chunks_removed"] == document["chunk_count"]
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]

    assert client.get(path, headers=ALPHA).status_code == 404
    assert client.delete(path, headers=ALPHA).status_code == 404
    assert client.get("/documents", headers=ALPHA).json()["total"] == 0
    search = client.post("/search", headers=ALPHA, json={"query": "password"}).json()
    assert search["results"] == []


def test_delete_from_other_workspace_is_404_and_keeps_document(
    client: TestClient, make_pdf: MakePdf
) -> None:
    document = upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    path = f"/documents/{document['document_id']}"
    assert client.delete(path, headers=BETA).status_code == 404
    assert client.get(path, headers=ALPHA).status_code == 200


# --------------------------------------------------------------------------------------------
# Query / search
# --------------------------------------------------------------------------------------------


def test_query_on_empty_workspace_abstains_without_llm_call(make_client: ClientFactory) -> None:
    llm = fake_llm("should never be used [S1]", ["S1"])
    client = make_client(llm, **LLM_SETTINGS)
    response = client.post("/query", headers=ALPHA, json={"question": "What is the PTO policy?"})
    assert response.status_code == 200
    body = QueryResponse.model_validate(response.json())
    assert body.abstained is True
    assert body.answer == NO_ANSWER_TEXT
    assert body.citations == []
    assert body.answer_mode == "llm"
    assert body.timings.generation_ms == 0
    assert llm.calls == 0


def test_query_with_valid_citation(make_client: ClientFactory, make_pdf: MakePdf) -> None:
    llm = fake_llm("Passwords must be at least 12 characters long [S1].", ["S1"])
    client = make_client(llm, **LLM_SETTINGS)
    document = upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    question = "How long must passwords be?"

    response = client.post("/query", headers=ALPHA, json={"question": question})
    assert response.status_code == 200
    body = QueryResponse.model_validate(response.json())
    assert llm.calls == 1
    assert body.abstained is False
    assert body.answer_mode == "llm"
    assert body.model == "fake-model"
    assert "[S1]" in body.answer
    assert body.invalid_citation_ids == []
    assert body.request_id == response.headers[REQUEST_ID_HEADER]

    [citation] = body.citations
    assert citation.citation_id == "S1"
    assert citation.document_id == document["document_id"]
    assert citation.filename == "policy.pdf"
    assert citation.chunk_id.startswith(f"{document['document_id']}:p{citation.page_number}:")
    assert citation.excerpt

    # S1 is the top retrieved chunk: the same retrieval that /search exposes.
    search = SearchResponse.model_validate(
        client.post("/search", headers=ALPHA, json={"query": question}).json()
    )
    top = search.results[0]
    assert (citation.chunk_id, citation.filename, citation.page_number) == (
        top.chunk_id,
        top.filename,
        top.page_number,
    )


def test_query_with_invalid_citation_abstains(
    make_client: ClientFactory, make_pdf: MakePdf
) -> None:
    llm = fake_llm("Passwords are 42 characters [S9].", ["S9"])
    client = make_client(llm, **LLM_SETTINGS)
    upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")

    body = QueryResponse.model_validate(
        client.post("/query", headers=ALPHA, json={"question": "How long are passwords?"}).json()
    )
    assert body.invalid_citation_ids == ["S9"]
    assert body.abstained is True
    assert body.citations == []
    assert body.answer.startswith(NO_ANSWER_TEXT)


def test_query_include_debug_returns_retrieved_chunks(
    make_client: ClientFactory, make_pdf: MakePdf
) -> None:
    client = make_client(fake_llm("Twelve characters [S1].", ["S1"]), **LLM_SETTINGS)
    upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    question = "How long must passwords be?"

    plain = client.post("/query", headers=ALPHA, json={"question": question}).json()
    assert plain["retrieved"] is None

    debug = QueryResponse.model_validate(
        client.post(
            "/query", headers=ALPHA, json={"question": question, "include_debug": True}
        ).json()
    )
    assert debug.retrieved
    assert all(isinstance(chunk.score, float) for chunk in debug.retrieved)
    assert debug.retrieved[0].chunk_id == debug.citations[0].chunk_id
    assert all(chunk.text for chunk in debug.retrieved)


def test_query_extractive_mode_without_llm(client: TestClient, make_pdf: MakePdf) -> None:
    upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    body = QueryResponse.model_validate(
        client.post("/query", headers=ALPHA, json={"question": "How long are passwords?"}).json()
    )
    assert body.answer_mode == "extractive"
    assert body.abstained is False
    assert body.model is None
    assert body.citations and all(c.citation_id.startswith("S") for c in body.citations)


def test_search_returns_scored_results(client: TestClient, make_pdf: MakePdf) -> None:
    document = upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    response = client.post(
        "/search", headers=ALPHA, json={"query": "password rotation", "top_k": 2, "mode": "dense"}
    )
    assert response.status_code == 200
    body = SearchResponse.model_validate(response.json())
    assert body.request_id == response.headers[REQUEST_ID_HEADER]
    assert body.workspace_id == "alpha"
    assert body.query == "password rotation"
    assert body.mode == "dense"
    assert 1 <= len(body.results) <= 2
    assert all(isinstance(r.score, float) for r in body.results)
    assert all(r.dense_score is not None for r in body.results)
    assert all(r.document_id == document["document_id"] for r in body.results)
    assert body.timings.total_ms >= body.timings.retrieval_ms


def test_search_defaults_to_configured_mode(client: TestClient, make_pdf: MakePdf) -> None:
    upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    body = client.post("/search", headers=ALPHA, json={"query": "vpn tokens"}).json()
    assert body["mode"] == "hybrid"
    assert body["results"]


# --------------------------------------------------------------------------------------------
# Workspace isolation over HTTP
# --------------------------------------------------------------------------------------------


def test_workspaces_are_isolated_over_http(make_client: ClientFactory, make_pdf: MakePdf) -> None:
    client = make_client(fake_llm("25 days [S1].", ["S1"]), **LLM_SETTINGS)
    doc_a = upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    doc_b = upload_one(client, BETA, make_pdf, LEAVE_PAGES, "leave.pdf")

    # Alpha asks about beta's content: only alpha's chunks can ever come back.
    for mode in ("dense", "hybrid"):
        search = SearchResponse.model_validate(
            client.post(
                "/search",
                headers=ALPHA,
                json={"query": "paid time off parental leave", "top_k": 20, "mode": mode},
            ).json()
        )
        assert search.results
        assert {r.document_id for r in search.results} == {doc_a["document_id"]}
        assert all(r.chunk_id.startswith(doc_a["document_id"]) for r in search.results)

    # Beta's citations never point at alpha's document, even when asking about it.
    answer = QueryResponse.model_validate(
        client.post(
            "/query",
            headers=BETA,
            json={"question": "How long must passwords be?", "include_debug": True},
        ).json()
    )
    assert answer.workspace_id == "beta"
    assert answer.citations
    assert {c.document_id for c in answer.citations} == {doc_b["document_id"]}
    assert answer.retrieved is not None
    assert {r.document_id for r in answer.retrieved} == {doc_b["document_id"]}

    # Listings and lookups are scoped too.
    assert [d["document_id"] for d in client.get("/documents", headers=BETA).json()["documents"]]
    assert client.get(f"/documents/{doc_a['document_id']}", headers=BETA).status_code == 404
    assert client.get(f"/documents/{doc_b['document_id']}", headers=ALPHA).status_code == 404


def test_document_filter_cannot_reach_other_workspace(
    client: TestClient, make_pdf: MakePdf
) -> None:
    upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    doc_b = upload_one(client, BETA, make_pdf, LEAVE_PAGES, "leave.pdf")
    body = client.post(
        "/search",
        headers=ALPHA,
        json={"query": "leave", "document_ids": [doc_b["document_id"]]},
    ).json()
    assert body["results"] == []


# --------------------------------------------------------------------------------------------
# Validation, request ids and error mapping
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "payload", "field"),
    [
        ("/query", {"question": "hi"}, "question"),
        ("/query", {"question": "What is the policy?", "top_k": 0}, "top_k"),
        ("/query", {"question": "What is the policy?", "mode": "sparse"}, "mode"),
        ("/search", {"query": "   "}, "query"),
        ("/search", {"query": "ok", "top_k": 51}, "top_k"),
        ("/search", {}, "query"),
    ],
)
def test_validation_errors_return_error_body_with_request_id(
    client: TestClient, path: str, payload: dict[str, Any], field: str
) -> None:
    response = client.post(path, headers=ALPHA, json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert body["error"] == "Validation error"
    assert isinstance(body["detail"], list) and body["detail"]
    assert any(field in error["loc"] for error in body["detail"])


def test_upload_without_files_is_validation_error(client: TestClient) -> None:
    response = client.post("/documents", headers=ALPHA, files=[("other", ("x.pdf", b"%PDF-"))])
    assert response.status_code == 422
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_every_response_carries_request_id(client: TestClient) -> None:
    for response in (
        client.get("/health"),
        client.get("/documents", headers=ALPHA),
        client.get("/documents"),
        client.post("/query", headers=ALPHA, json={"question": "hi"}),
        client.get("/nope"),
    ):
        assert response.headers[REQUEST_ID_HEADER], response.url


def test_provided_request_id_is_echoed(client: TestClient) -> None:
    headers = {**ALPHA, REQUEST_ID_HEADER: "trace-abc_123"}
    response = client.post("/search", headers=headers, json={"query": "anything"})
    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "trace-abc_123"
    assert response.json()["request_id"] == "trace-abc_123"

    error = client.get("/documents/missing", headers=headers)
    assert error.headers[REQUEST_ID_HEADER] == "trace-abc_123"
    assert error.json()["request_id"] == "trace-abc_123"


def test_malformed_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "bad id with spaces"})
    assert response.headers[REQUEST_ID_HEADER] != "bad id with spaces"
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_llm_failure_maps_to_502(make_client: ClientFactory, make_pdf: MakePdf) -> None:
    client = make_client(RaisingFakeLLM(responses=["unused"]), **LLM_SETTINGS)
    upload_one(client, ALPHA, make_pdf, POLICY_PAGES, "policy.pdf")
    response = client.post("/query", headers=ALPHA, json={"question": "How long are passwords?"})
    assert response.status_code == 502
    body = assert_error_body(response)
    assert body.error == "Bad Gateway"
    assert body.detail is not None and body.detail.startswith("LLM generation failed (")


def test_unexpected_error_maps_to_500_with_request_id(
    make_client: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(None, raise_server_exceptions=False)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(services_of(client).answer, "search", boom)
    response = client.post("/search", headers=ALPHA, json={"query": "anything"})
    assert response.status_code == 500
    body = assert_error_body(response)
    assert body.detail == "Internal server error"
    assert "secret" not in response.text


def test_http_404_for_unknown_route_uses_error_body(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert assert_error_body(response).error == "Not Found"


# --------------------------------------------------------------------------------------------
# Persistence across restarts
# --------------------------------------------------------------------------------------------


def test_documents_survive_app_restart(settings: Settings, make_pdf: MakePdf) -> None:
    app_settings = settings.model_copy(update={"chunk_size": 200, "chunk_overlap": 20})

    data = make_pdf(POLICY_PAGES)  # built once: every make_pdf call yields different bytes

    with TestClient(create_app(app_settings, embedder=HashingEmbedder(64), llm=None)) as first:
        response = upload(first, ALPHA, pdf_part("policy.pdf", data))
        assert response.status_code == 201
        document = response.json()["results"][0]["document"]
        assert first.post("/search", headers=ALPHA, json={"query": "password"}).json()["results"]

    with TestClient(create_app(app_settings, embedder=HashingEmbedder(64), llm=None)) as second:
        listing = second.get("/documents", headers=ALPHA).json()
        assert [d["document_id"] for d in listing["documents"]] == [document["document_id"]]
        search = second.post("/search", headers=ALPHA, json={"query": "password"}).json()
        assert search["results"]
        assert {r["document_id"] for r in search["results"]} == {document["document_id"]}
        # Duplicate detection also survives: the same bytes are recognised after restart.
        again = upload(second, ALPHA, pdf_part("policy.pdf", data))
        assert again.status_code == 200
        assert again.json()["results"][0]["outcome"] == "duplicate"
