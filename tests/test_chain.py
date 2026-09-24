"""Tests for ``AnswerService`` (LCEL chain + citation verification) and the chat model factory.

No real LLM is ever called: the chain runs against ``FakeListChatModel`` and retrieval is a
stub that returns prepared ``ScoredChunk`` lists.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field

from app.core.config import Settings
from app.generation.chain import EXTRACTIVE_HEADER, AnswerService, GenerationError
from app.generation.llm import ConfigurationError, create_chat_model
from app.generation.prompts import NO_ANSWER_TEXT
from app.models.domain import Chunk, ScoredChunk
from app.models.schemas import AnswerMode, QueryRequest, RetrievalMode, SearchRequest

WORKSPACE = "acme"
REQUEST_ID = "req-123"
DOC_A = uuid.uuid4().hex
DOC_B = uuid.uuid4().hex


# ---- fixtures / helpers ----------------------------------------------------------------------


def make_scored(
    index: int,
    text: str,
    *,
    document_id: str = DOC_A,
    filename: str = "handbook.pdf",
    page: int = 1,
    score: float = 0.9,
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=Chunk.make_id(document_id, page, index),
        document_id=document_id,
        workspace_id=WORKSPACE,
        filename=filename,
        page_number=page,
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
    )
    return ScoredChunk(
        chunk=chunk, vector_id=index + 1, score=score, dense_score=score, sparse_rank=index + 1
    )


def sample_chunks() -> list[ScoredChunk]:
    return [
        make_scored(0, "Employees accrue 25 days of paid leave per year.", page=3, score=0.91),
        make_scored(1, "Unused leave may be carried over up to 5 days.", page=4, score=0.85),
        make_scored(
            0,
            "Remote work requires manager approval.",
            document_id=DOC_B,
            filename="remote.pdf",
            page=1,
            score=0.62,
        ),
        make_scored(
            1,
            "Equipment is shipped within 10 days.",
            document_id=DOC_B,
            filename="remote.pdf",
            page=2,
            score=0.55,
        ),
    ]


class StubRetrieval:
    """Duck-typed stand-in for ``RetrievalService`` returning prepared chunks."""

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        *,
        k: int | None = None,
        document_ids: list[str] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[ScoredChunk]:
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "query": query,
                "k": k,
                "document_ids": document_ids,
                "mode": mode,
            }
        )
        return list(self.chunks)


class CountingFakeLLM(FakeListChatModel):
    """Fake chat model that counts invocations and records the prompts it received."""

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


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "_env_file": None,
        "environment": "test",
        "llm_provider": "openai",
        "llm_model": "fake-model",
        "citation_excerpt_chars": 80,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def llm_json(answer: str, citations: list[str], *, insufficient: bool = False) -> str:
    return json.dumps(
        {"answer": answer, "citations": citations, "insufficient_evidence": insufficient}
    )


def make_service(
    chunks: list[ScoredChunk], llm: BaseChatModel | None, **settings_overrides: Any
) -> tuple[AnswerService, StubRetrieval]:
    retrieval = StubRetrieval(chunks)
    service = AnswerService(
        settings=make_settings(**settings_overrides),
        retrieval=retrieval,  # type: ignore[arg-type]
        llm=llm,
    )
    return service, retrieval


def query(**overrides: Any) -> QueryRequest:
    payload: dict[str, Any] = {"question": "How many days of leave do employees get?"}
    payload.update(overrides)
    return QueryRequest(**payload)


# ---- AnswerService.answer: LLM mode ------------------------------------------------------------


def test_happy_path_maps_citations_to_retrieved_chunks() -> None:
    chunks = sample_chunks()
    llm = CountingFakeLLM(
        responses=[llm_json("Employees get 25 days [S1], carrying over 5 [S2].", ["S1", "S2"])]
    )
    service, _ = make_service(chunks, llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.request_id == REQUEST_ID
    assert response.workspace_id == WORKSPACE
    assert response.abstained is False
    assert response.answer_mode == AnswerMode.LLM
    assert response.answer == "Employees get 25 days [S1], carrying over 5 [S2]."
    assert response.model == "fake-model"
    assert response.invalid_citation_ids == []
    assert response.retrieved is None
    assert [c.citation_id for c in response.citations] == ["S1", "S2"]
    assert [c.chunk_id for c in response.citations] == [
        chunks[0].chunk.chunk_id,
        chunks[1].chunk.chunk_id,
    ]
    assert [c.page_number for c in response.citations] == [3, 4]
    assert response.citations[0].filename == "handbook.pdf"
    assert response.citations[0].excerpt.startswith("Employees accrue 25 days")
    assert llm.calls == 1


def test_llm_receives_labelled_sources_and_question() -> None:
    llm = CountingFakeLLM(responses=[llm_json("25 days [S1].", ["S1"])])
    service, _ = make_service(sample_chunks(), llm)

    service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    prompt = llm.prompts[0]
    assert '<source id="S1" file="handbook.pdf" page="3">' in prompt
    assert '<source id="S4" file="remote.pdf" page="2">' in prompt
    assert "How many days of leave do employees get?" in prompt
    assert "untrusted" in prompt


def test_retrieval_receives_request_parameters() -> None:
    llm = CountingFakeLLM(responses=[llm_json("x [S1]", ["S1"])])
    service, retrieval = make_service(sample_chunks(), llm)
    request = query(top_k=7, document_ids=[DOC_A], mode=RetrievalMode.DENSE)

    service.answer(WORKSPACE, request, request_id=REQUEST_ID)

    assert retrieval.calls == [
        {
            "workspace_id": WORKSPACE,
            "query": request.question,
            "k": 7,
            "document_ids": [DOC_A],
            "mode": RetrievalMode.DENSE,
        }
    ]


def test_invalid_label_is_dropped_and_reported() -> None:
    chunks = sample_chunks()
    llm = CountingFakeLLM(responses=[llm_json("25 days [S1]; see also [S9].", ["S1", "S9"])])
    service, _ = make_service(chunks, llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.abstained is False
    assert [c.citation_id for c in response.citations] == ["S1"]
    assert response.citations[0].chunk_id == chunks[0].chunk.chunk_id
    assert response.invalid_citation_ids == ["S9"]


def test_inline_marker_missing_from_list_is_still_honoured() -> None:
    chunks = sample_chunks()
    llm = CountingFakeLLM(responses=[llm_json("Carry-over is 5 days [S2].", [])])
    service, _ = make_service(chunks, llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.abstained is False
    assert [c.chunk_id for c in response.citations] == [chunks[1].chunk.chunk_id]


def test_insufficient_evidence_abstains_with_explanation() -> None:
    llm = CountingFakeLLM(
        responses=[llm_json("The sources do not mention parental leave.", [], insufficient=True)]
    )
    service, _ = make_service(sample_chunks(), llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.abstained is True
    assert response.answer_mode == AnswerMode.LLM
    assert response.answer == f"{NO_ANSWER_TEXT}\nThe sources do not mention parental leave."
    assert response.citations == []
    assert response.model == "fake-model"


def test_insufficient_evidence_with_citations_still_abstains() -> None:
    llm = CountingFakeLLM(responses=[llm_json("Maybe [S1].", ["S1"], insufficient=True)])
    service, _ = make_service(sample_chunks(), llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.abstained is True
    assert response.citations == []
    assert response.answer.startswith(NO_ANSWER_TEXT)


def test_answer_without_valid_citations_abstains() -> None:
    llm = CountingFakeLLM(responses=[llm_json("Leave is 30 days, I believe.", ["S42"])])
    service, _ = make_service(sample_chunks(), llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.abstained is True
    assert response.answer == NO_ANSWER_TEXT
    assert response.citations == []
    assert response.invalid_citation_ids == ["S42"]


def test_no_retrieved_chunks_abstains_without_calling_llm() -> None:
    llm = CountingFakeLLM(responses=[llm_json("should never be used [S1]", ["S1"])])
    service, _ = make_service([], llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.abstained is True
    assert response.answer == NO_ANSWER_TEXT
    assert response.citations == []
    assert response.answer_mode == AnswerMode.LLM
    assert response.model is None
    assert response.timings.generation_ms == 0.0
    assert llm.calls == 0


def test_llm_failure_raises_generation_error_without_prompt_text() -> None:
    llm = RaisingFakeLLM(responses=["unused"])
    service, _ = make_service(sample_chunks(), llm)

    with pytest.raises(GenerationError) as excinfo:
        service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert str(excinfo.value) == "TimeoutError"
    assert "leave" not in str(excinfo.value).lower()
    assert isinstance(excinfo.value.__cause__, TimeoutError)


def test_include_debug_populates_retrieved() -> None:
    chunks = sample_chunks()
    llm = CountingFakeLLM(responses=[llm_json("25 days [S1].", ["S1"])])
    service, _ = make_service(chunks, llm)

    response = service.answer(WORKSPACE, query(include_debug=True), request_id=REQUEST_ID)

    assert response.retrieved is not None
    assert len(response.retrieved) == len(chunks)
    first = response.retrieved[0]
    assert first.chunk_id == chunks[0].chunk.chunk_id
    assert first.document_id == DOC_A
    assert first.filename == "handbook.pdf"
    assert first.page_number == 3
    assert first.score == pytest.approx(0.91)
    assert first.dense_score == pytest.approx(0.91)
    assert first.sparse_rank == 1
    assert first.rerank_score is None
    assert first.text == chunks[0].chunk.text


def test_timings_are_non_negative_and_consistent() -> None:
    llm = CountingFakeLLM(responses=[llm_json("25 days [S1].", ["S1"])])
    service, _ = make_service(sample_chunks(), llm)

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    timings = response.timings
    assert timings.retrieval_ms >= 0
    assert timings.generation_ms >= 0
    assert timings.total_ms >= timings.retrieval_ms
    assert timings.total_ms >= timings.generation_ms


def test_context_respects_max_context_chars() -> None:
    chunks = [make_scored(i, f"passage {i} " + "x" * 400, page=i + 1) for i in range(6)]
    llm = CountingFakeLLM(responses=[llm_json("x [S1]", ["S1"])])
    service, _ = make_service(chunks, llm, max_context_chars=1000)

    service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    prompt = llm.prompts[0]
    assert '<source id="S1"' in prompt
    assert '<source id="S6"' not in prompt


# ---- AnswerService.answer: extractive mode -----------------------------------------------------


def test_extractive_mode_returns_top_three_passages() -> None:
    chunks = sample_chunks()
    service, _ = make_service(chunks, None, llm_provider="extractive")

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.answer_mode == AnswerMode.EXTRACTIVE
    assert response.abstained is False
    assert response.model is None
    assert response.invalid_citation_ids == []
    assert response.answer.startswith(EXTRACTIVE_HEADER + "\n\n")
    assert "[S1] (handbook.pdf, p. 3): Employees accrue 25 days" in response.answer
    assert "[S2] (handbook.pdf, p. 4): Unused leave" in response.answer
    assert "[S3] (remote.pdf, p. 1): Remote work" in response.answer
    assert "[S4]" not in response.answer
    assert [c.citation_id for c in response.citations] == ["S1", "S2", "S3"]
    assert [c.chunk_id for c in response.citations] == [sc.chunk.chunk_id for sc in chunks[:3]]
    assert response.timings.generation_ms == 0.0


def test_extractive_mode_with_no_chunks_abstains() -> None:
    service, _ = make_service([], None, llm_provider="extractive")

    response = service.answer(WORKSPACE, query(), request_id=REQUEST_ID)

    assert response.abstained is True
    assert response.answer == NO_ANSWER_TEXT
    assert response.answer_mode == AnswerMode.EXTRACTIVE
    assert response.citations == []


# ---- AnswerService.search ----------------------------------------------------------------------


def test_search_returns_retrieved_chunks_with_default_mode() -> None:
    chunks = sample_chunks()
    service, retrieval = make_service(
        chunks, None, llm_provider="extractive", retrieval_mode="dense"
    )
    request = SearchRequest(query="remote work approval", top_k=2)

    response = service.search(WORKSPACE, request, request_id=REQUEST_ID)

    assert response.request_id == REQUEST_ID
    assert response.workspace_id == WORKSPACE
    assert response.query == "remote work approval"
    assert response.mode == RetrievalMode.DENSE
    assert [r.chunk_id for r in response.results] == [sc.chunk.chunk_id for sc in chunks]
    assert response.timings.generation_ms == 0.0
    assert response.timings.total_ms >= response.timings.retrieval_ms >= 0
    assert retrieval.calls[0]["k"] == 2
    assert retrieval.calls[0]["mode"] is None


def test_search_honours_explicit_mode() -> None:
    service, retrieval = make_service([], None, llm_provider="extractive", retrieval_mode="dense")
    request = SearchRequest(query="anything", mode=RetrievalMode.HYBRID, document_ids=[DOC_B])

    response = service.search(WORKSPACE, request, request_id=REQUEST_ID)

    assert response.mode == RetrievalMode.HYBRID
    assert response.results == []
    assert retrieval.calls[0]["mode"] == RetrievalMode.HYBRID
    assert retrieval.calls[0]["document_ids"] == [DOC_B]


# ---- create_chat_model (offline: models are constructed, never invoked) -----------------------


@pytest.fixture
def no_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_create_chat_model_extractive_returns_none(no_provider_env: None) -> None:
    assert create_chat_model(Settings(_env_file=None, llm_provider="extractive")) is None


def test_create_chat_model_auto_without_keys_is_extractive(no_provider_env: None) -> None:
    assert create_chat_model(Settings(_env_file=None, llm_provider="auto")) is None


def test_create_chat_model_anthropic_has_no_temperature(no_provider_env: None) -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", ANTHROPIC_API_KEY="sk-ant-test")

    model = create_chat_model(settings)

    assert isinstance(model, BaseChatModel)
    assert type(model).__name__ == "ChatAnthropic"
    assert getattr(model, "temperature", None) is None
    assert model.max_tokens == 1024
    assert model.max_retries == 2


def test_create_chat_model_openai_uses_configured_model(no_provider_env: None) -> None:
    settings = Settings(
        _env_file=None, llm_provider="openai", llm_model="gpt-4.1-mini", OPENAI_API_KEY="sk-test"
    )

    model = create_chat_model(settings)

    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "gpt-4.1-mini"
    assert model.max_retries == 2


def test_create_chat_model_unknown_provider_raises_configuration_error(
    no_provider_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Settings, "resolved_llm_provider", lambda self: "bogus-provider")
    settings = Settings(_env_file=None, llm_model="some-model")

    with pytest.raises(ConfigurationError) as excinfo:
        create_chat_model(settings)

    assert "bogus-provider" in str(excinfo.value)
    assert "some-model" in str(excinfo.value)
