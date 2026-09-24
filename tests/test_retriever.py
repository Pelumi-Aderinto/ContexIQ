"""Tests for RetrievalService, reciprocal rank fusion, reranking and the LangChain adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest
import structlog
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.models.domain import Chunk, ScoredChunk
from app.models.schemas import DocumentInfo, RetrievalMode, UploadOutcome
from app.retrieval.retriever import (
    CrossEncoderReranker,
    RetrievalService,
    WorkspaceRetriever,
    reciprocal_rank_fusion,
    scored_chunk_to_retrieved,
)

if TYPE_CHECKING:
    from pathlib import Path

    from app.core.config import Settings
    from tests.conftest import MakePdf, RagStack

WS = "alpha"

DOC_A_PAGES = [
    "Zephyrite is a rare blue mineral. Zephyrite deposits form in ancient riverbeds and "
    "collectors prize zephyrite for its faint glow under ultraviolet light.",
    "Kestrel migration routes cross the northern plains every autumn. Kestrels nest in cliff "
    "hollows and abandoned barns and hunt over open farmland.",
]
DOC_B_PAGES = [
    "Maintenance bulletin. Replace the pressure valve AX2-7731 every twelve months. Torque "
    "the AX2-7731 retaining nut to 12 Nm and record the serial number.",
    "General safety notes. Wear gloves when handling hot components and keep the work area "
    "free of clutter, spilled oil and loose tools.",
]
DOCUMENT_METADATA_KEYS = {
    "chunk_id",
    "document_id",
    "filename",
    "page_number",
    "score",
    "dense_score",
    "sparse_rank",
    "rerank_score",
    "vector_id",
}


@dataclass
class Corpus:
    stack: RagStack
    doc_a: DocumentInfo
    doc_b: DocumentInfo

    @property
    def retrieval(self) -> RetrievalService:
        return self.stack.retrieval

    def service_with(self, reranker: object = None, **overrides: object) -> RetrievalService:
        """A second service over the same stores with tweaked settings and/or a reranker."""
        return RetrievalService(
            settings=self.stack.settings.model_copy(update=overrides),
            store=self.stack.store,
            vector_store=self.stack.vector_store,
            embedder=self.stack.embedder,
            reranker=reranker,  # type: ignore[arg-type]
        )


@dataclass
class KeywordReranker:
    """Fake reranker: texts containing ``keyword`` score 1.0, everything else 0.0."""

    keyword: str
    calls: list[int] = field(default_factory=list)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append(len(texts))
        return [1.0 if self.keyword in text else 0.0 for text in texts]


@pytest.fixture
def corpus(rag_stack: RagStack, make_pdf: MakePdf) -> Corpus:
    a = rag_stack.pipeline.ingest(make_pdf(DOC_A_PAGES), "minerals.pdf", WS)
    b = rag_stack.pipeline.ingest(make_pdf(DOC_B_PAGES), "bulletin.pdf", WS)
    assert a.outcome is UploadOutcome.INDEXED and b.outcome is UploadOutcome.INDEXED
    assert a.document is not None and b.document is not None
    assert rag_stack.store.count_chunks(WS) >= 4
    return Corpus(rag_stack, a.document, b.document)


def _make_scored_chunk(**overrides: object) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=Chunk.make_id("doc", 3, 7),
        document_id="doc",
        workspace_id=WS,
        filename="file.pdf",
        page_number=3,
        chunk_index=7,
        text="some text",
        char_start=0,
        char_end=9,
    )
    values: dict[str, object] = {
        "chunk": chunk,
        "vector_id": 42,
        "score": 0.5,
        "dense_score": 0.4,
        "sparse_rank": 2,
        "rerank_score": None,
    }
    values.update(overrides)
    return ScoredChunk(**values)  # type: ignore[arg-type]


# ---- dense and hybrid retrieval --------------------------------------------------------------


def test_dense_retrieval_finds_the_page_with_a_distinctive_term(corpus: Corpus) -> None:
    results = corpus.retrieval.retrieve(WS, "zephyrite mineral glow", mode=RetrievalMode.DENSE)

    assert results
    top = results[0]
    assert "zephyrite" in top.chunk.text.lower()
    assert top.chunk.page_number == 1
    assert top.chunk.document_id == corpus.doc_a.document_id
    assert top.chunk.filename == "minerals.pdf"
    assert top.dense_score is not None
    assert top.score == top.dense_score
    assert top.sparse_rank is None and top.rerank_score is None
    scores = [sc.score for sc in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_retrieval_finds_an_exact_rare_token(corpus: Corpus) -> None:
    results = corpus.retrieval.retrieve(WS, "AX2-7731", mode=RetrievalMode.HYBRID)

    assert results
    top = results[0]
    assert "AX2-7731" in top.chunk.text
    assert top.sparse_rank == 1
    assert top.chunk.filename == "bulletin.pdf"
    assert top.chunk.page_number == 1
    rrf_k = corpus.stack.settings.rrf_k
    for sc in results:
        assert 0.0 < sc.score <= 2.0 / (rrf_k + 1)
        assert sc.dense_score is not None or sc.sparse_rank is not None
    assert [sc.score for sc in results] == sorted((sc.score for sc in results), reverse=True)


def test_default_mode_comes_from_settings(corpus: Corpus) -> None:
    assert corpus.stack.settings.retrieval_mode == "hybrid"
    default = corpus.retrieval.retrieve(WS, "AX2-7731")
    hybrid = corpus.retrieval.retrieve(WS, "AX2-7731", mode=RetrievalMode.HYBRID)
    assert [sc.vector_id for sc in default] == [sc.vector_id for sc in hybrid]
    assert default[0].sparse_rank == 1


@pytest.mark.parametrize("mode", [RetrievalMode.DENSE, RetrievalMode.HYBRID])
def test_document_ids_filter_restricts_results(corpus: Corpus, mode: RetrievalMode) -> None:
    query = "zephyrite valve AX2-7731"
    only_a = corpus.retrieval.retrieve(
        WS, query, document_ids=[corpus.doc_a.document_id], mode=mode
    )
    only_b = corpus.retrieval.retrieve(
        WS, query, document_ids=[corpus.doc_b.document_id], mode=mode
    )

    assert only_a and {sc.chunk.document_id for sc in only_a} == {corpus.doc_a.document_id}
    assert only_b and {sc.chunk.document_id for sc in only_b} == {corpus.doc_b.document_id}
    assert corpus.retrieval.retrieve(WS, query, document_ids=["no-such-doc"], mode=mode) == []
    unfiltered = corpus.retrieval.retrieve(WS, query, document_ids=[], mode=mode)
    assert {sc.chunk.document_id for sc in unfiltered} == {
        corpus.doc_a.document_id,
        corpus.doc_b.document_id,
    }


def test_k_is_clamped_to_max_top_k(corpus: Corpus) -> None:
    service = corpus.service_with(max_top_k=2)
    assert len(service.retrieve(WS, "zephyrite", k=50, mode=RetrievalMode.DENSE)) == 2
    assert len(service.retrieve(WS, "zephyrite", k=50, mode=RetrievalMode.HYBRID)) == 2
    assert len(service.retrieve(WS, "zephyrite", k=1)) == 1


def test_k_defaults_to_settings_top_k(corpus: Corpus) -> None:
    service = corpus.service_with(top_k=3)
    assert len(service.retrieve(WS, "zephyrite", mode=RetrievalMode.DENSE)) == 3
    assert len(corpus.retrieval.retrieve(WS, "zephyrite", mode=RetrievalMode.DENSE)) == min(
        corpus.stack.settings.top_k, corpus.stack.store.count_chunks(WS)
    )


def test_empty_unknown_workspace_or_blank_query_returns_nothing(corpus: Corpus) -> None:
    assert corpus.retrieval.retrieve("beta", "zephyrite") == []
    assert corpus.retrieval.retrieve("never-seen", "zephyrite") == []
    assert corpus.retrieval.retrieve(WS, "   ") == []


def test_min_dense_score_drops_weak_hits(corpus: Corpus) -> None:
    service = corpus.service_with(min_dense_score=0.99)
    _, chunk = next(corpus.stack.store.iter_chunks(WS))

    exact = service.retrieve(WS, chunk.text, mode=RetrievalMode.DENSE)
    assert [sc.chunk.chunk_id for sc in exact] == [chunk.chunk_id]
    assert exact[0].dense_score == pytest.approx(1.0, abs=1e-5)
    assert service.retrieve(WS, "completely unrelated wording", mode=RetrievalMode.DENSE) == []


def test_missing_sqlite_rows_are_dropped_with_a_warning(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = corpus.stack.store.get_chunks

    def get_chunks_missing_first(workspace_id: str, vector_ids: list[int]) -> dict[int, Chunk]:
        chunks = original(workspace_id, vector_ids)
        chunks.pop(vector_ids[0], None)
        return chunks

    monkeypatch.setattr(corpus.stack.store, "get_chunks", get_chunks_missing_first)
    with structlog.testing.capture_logs() as logs:
        results = corpus.retrieval.retrieve(WS, "zephyrite", k=3, mode=RetrievalMode.DENSE)

    assert len(results) == 2
    drift = [entry for entry in logs if entry["event"] == "retrieval.index_db_drift"]
    assert drift and drift[0]["missing_chunks"] == 1 and drift[0]["workspace_id"] == WS


# ---- fusion ----------------------------------------------------------------------------------


def test_reciprocal_rank_fusion_with_known_rankings() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)

    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[4] == pytest.approx(1 / 63)
    assert sorted(fused, key=fused.__getitem__, reverse=True) == [1, 3, 2, 4]


def test_reciprocal_rank_fusion_edge_cases() -> None:
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}
    assert reciprocal_rank_fusion([[5, 5]], k=60) == {5: pytest.approx(1 / 61)}
    assert reciprocal_rank_fusion([[7]], k=1) == {7: pytest.approx(0.5)}
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([[1]], k=0)


def test_scored_chunk_to_retrieved_copies_every_field() -> None:
    sc = _make_scored_chunk(rerank_score=0.9)
    retrieved = scored_chunk_to_retrieved(sc)

    assert retrieved.chunk_id == sc.chunk.chunk_id
    assert retrieved.document_id == "doc"
    assert retrieved.filename == "file.pdf"
    assert retrieved.page_number == 3
    assert retrieved.text == "some text"
    assert (retrieved.score, retrieved.dense_score, retrieved.sparse_rank) == (0.5, 0.4, 2)
    assert retrieved.rerank_score == 0.9


# ---- reranking -------------------------------------------------------------------------------


def test_reranker_reorders_results_and_sets_scores(corpus: Corpus) -> None:
    reranker = KeywordReranker("Kestrel")
    service = corpus.service_with(rerank_candidates=20, reranker=reranker)

    results = service.retrieve(WS, "zephyrite", k=2, mode=RetrievalMode.DENSE)

    assert len(results) == 2
    assert "Kestrel" in results[0].chunk.text
    assert results[0].rerank_score == 1.0 == results[0].score
    assert results[1].rerank_score == 0.0 == results[1].score
    assert results[0].dense_score is not None  # original signal is preserved for debugging
    assert reranker.calls == [corpus.stack.store.count_chunks(WS)]


def test_reranker_is_limited_to_rerank_candidates(corpus: Corpus) -> None:
    reranker = KeywordReranker("Kestrel")
    service = corpus.service_with(rerank_candidates=3, reranker=reranker)
    results = service.retrieve(WS, "zephyrite", k=1, mode=RetrievalMode.HYBRID)
    assert len(results) == 1
    assert reranker.calls == [3]


def test_reranker_returning_wrong_count_raises(corpus: Corpus) -> None:
    class BrokenReranker:
        def rerank(self, query: str, texts: list[str]) -> list[float]:
            return [1.0]

    service = corpus.service_with(reranker=BrokenReranker())
    with pytest.raises(RuntimeError, match="reranker returned"):
        service.retrieve(WS, "zephyrite", k=2)


def test_reranker_is_created_from_settings_only_when_enabled(corpus: Corpus) -> None:
    assert corpus.retrieval.reranker is None
    enabled = corpus.service_with(rerank_enabled=True, rerank_model="fake/cross-encoder")
    assert isinstance(enabled.reranker, CrossEncoderReranker)
    assert enabled.reranker.model_name == "fake/cross-encoder"
    assert enabled.reranker._model is None  # nothing loaded until the first rerank


def test_cross_encoder_reranker_loads_lazily_and_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[str] = []

    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
            return np.array([0.1 * len(text) for _, text in pairs], dtype=np.float32)

    def fake_load(self: CrossEncoderReranker) -> FakeCrossEncoder:
        loads.append(self.model_name)
        return FakeCrossEncoder()

    monkeypatch.setattr(CrossEncoderReranker, "_load_model", fake_load)
    reranker = CrossEncoderReranker("fake/model")

    assert reranker.rerank("q", []) == []
    assert loads == []
    scores = reranker.rerank("q", ["ab", "abcd"])
    assert scores == pytest.approx([0.2, 0.4])
    assert all(isinstance(s, float) for s in scores)
    reranker.rerank("q", ["x"])
    assert loads == ["fake/model"]


# ---- LangChain adapter -----------------------------------------------------------------------


def test_workspace_retriever_invoke_returns_documents_with_metadata(corpus: Corpus) -> None:
    retriever = corpus.retrieval.as_langchain_retriever(WS, k=3)
    assert isinstance(retriever, WorkspaceRetriever)

    docs = retriever.invoke("zephyrite mineral glow")

    assert docs and all(isinstance(doc, Document) for doc in docs)
    top = docs[0]
    assert set(top.metadata) == DOCUMENT_METADATA_KEYS
    assert "zephyrite" in top.page_content.lower()
    assert top.metadata["page_number"] == 1
    assert top.metadata["filename"] == "minerals.pdf"
    assert top.metadata["document_id"] == corpus.doc_a.document_id
    assert top.metadata["chunk_id"].startswith(f"{corpus.doc_a.document_id}:p1:c")
    assert isinstance(top.metadata["vector_id"], int)
    assert len(docs) <= 3


def test_workspace_retriever_forwards_document_ids(corpus: Corpus) -> None:
    retriever = corpus.retrieval.as_langchain_retriever(WS, document_ids=[corpus.doc_b.document_id])
    docs = retriever.invoke("zephyrite AX2-7731")
    assert docs
    assert {doc.metadata["document_id"] for doc in docs} == {corpus.doc_b.document_id}
    assert {doc.metadata["page_number"] for doc in docs} <= {1, 2}


def test_workspace_retriever_works_inside_an_lcel_chain(corpus: Corpus) -> None:
    retriever = corpus.retrieval.as_langchain_retriever(WS, k=2)
    chain = retriever | RunnableLambda(len)
    assert chain.invoke("zephyrite") == len(retriever.invoke("zephyrite")) == 2


async def test_workspace_retriever_supports_async_invoke(corpus: Corpus) -> None:
    retriever = corpus.retrieval.as_langchain_retriever(WS, k=2)
    docs = await retriever.ainvoke("zephyrite")
    assert [d.metadata["chunk_id"] for d in docs] == [
        d.metadata["chunk_id"] for d in retriever.invoke("zephyrite")
    ]


def test_workspace_retriever_hides_the_service(corpus: Corpus) -> None:
    retriever = corpus.retrieval.as_langchain_retriever(WS, k=2, document_ids=["x"])
    dumped = retriever.model_dump()
    assert "service" not in dumped
    assert dumped["workspace_id"] == WS and dumped["k"] == 2 and dumped["document_ids"] == ["x"]
    assert "RetrievalService" not in repr(retriever)


# ---- real embedding model ---------------------------------------------------------------------

SAMPLE_QUESTIONS: list[tuple[str, str, set[int]]] = [
    (
        "What is the minimum password length for standard user accounts?",
        "halcyon_information_security_policy.pdf",
        {1},
    ),
    (
        "How many days of paid time off do employees with less than two years of service accrue?",
        "halcyon_employee_handbook.pdf",
        {2},
    ),
    (
        "How many Aurora X200 units were shipped in Q2 FY2026?",
        "halcyon_q2_fy2026_business_review.pdf",
        {1, 2},
    ),
    (
        "What does error code E-310 mean?",
        "aurora_x200_installation_and_maintenance_guide.pdf",
        {3, 5},
    ),
    (
        "What is the maximum take-off weight of the Aurora X200?",
        "aurora_x200_technical_specification.pdf",
        {1, 5},
    ),
]


@pytest.mark.slow
def test_real_embedder_retrieves_the_expected_sample_documents(
    settings: Settings, sample_pdf_paths: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # local cache only, never the network
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.embeddings import SentenceTransformerEmbedder
    from app.retrieval.vector_store import VectorStore
    from app.storage.metadata_store import MetadataStore

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model, query_prefix=settings.embedding_query_prefix
    )
    store = MetadataStore(settings.db_path)
    vector_store = VectorStore(settings.index_dir, embedder.dimension)
    pipeline = IngestionPipeline(
        settings=settings, store=store, vector_store=vector_store, embedder=embedder
    )
    retrieval = RetrievalService(
        settings=settings, store=store, vector_store=vector_store, embedder=embedder
    )
    try:
        for path in sample_pdf_paths:
            result = pipeline.ingest(path.read_bytes(), path.name, WS)
            assert result.outcome is UploadOutcome.INDEXED, (path.name, result.message)
        assert store.count_chunks(WS) == vector_store.count(WS) > 0

        for question, expected_file, expected_pages in SAMPLE_QUESTIONS:
            results = retrieval.retrieve(WS, question, k=3, mode=RetrievalMode.HYBRID)
            files = [sc.chunk.filename for sc in results]
            assert expected_file in files, (question, files)
            pages = {sc.chunk.page_number for sc in results if sc.chunk.filename == expected_file}
            assert pages & expected_pages, (question, pages)
    finally:
        store.close()
