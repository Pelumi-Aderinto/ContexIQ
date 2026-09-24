"""Retrieval over one workspace: dense, hybrid (dense + BM25 fused with RRF), optional rerank.

``RetrievalService.retrieve`` is the only entry point the generation layer and the API use.
Every candidate id comes from the workspace's own FAISS index or from an FTS query filtered by
``workspace_id``, and hydration goes through ``MetadataStore.get_chunks`` which filters by
workspace again, so a result can never belong to another workspace.

Scores are exposed for debugging only: cosine similarity (dense), a reciprocal-rank-fusion
score (hybrid) or a cross-encoder logit (rerank). None of them is a probability.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Protocol

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor
from pydantic import ConfigDict, Field

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.domain import ScoredChunk
from app.models.schemas import RetrievalMode, RetrievedChunk
from app.retrieval.embeddings import Embedder, resolve_device
from app.retrieval.vector_store import VectorStore
from app.storage.metadata_store import MetadataStore

log = get_logger(__name__)

Hit = tuple[int, float]


# --------------------------------------------------------------------------------------------
# Fusion and conversion helpers
# --------------------------------------------------------------------------------------------


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Fuse several rankings: each item scores ``sum(1 / (k + rank))`` over the lists it is in.

    ``rank`` is 1-based. An item repeated inside one ranking counts once, at its best rank.
    Higher is better; ``k`` dampens the advantage of top ranks (60 is the usual default).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    fused: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def scored_chunk_to_retrieved(sc: ScoredChunk) -> RetrievedChunk:
    """Convert an internal ``ScoredChunk`` into the public ``RetrievedChunk`` schema."""
    chunk = sc.chunk
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        score=sc.score,
        dense_score=sc.dense_score,
        sparse_rank=sc.sparse_rank,
        rerank_score=sc.rerank_score,
        text=chunk.text,
    )


def scored_chunk_to_document(sc: ScoredChunk) -> Document:
    """Convert a ``ScoredChunk`` into a LangChain ``Document`` carrying full provenance."""
    chunk = sc.chunk
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "score": sc.score,
            "dense_score": sc.dense_score,
            "sparse_rank": sc.sparse_rank,
            "rerank_score": sc.rerank_score,
            "vector_id": sc.vector_id,
        },
    )


# --------------------------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------------------------


class Reranker(Protocol):
    """Scores ``texts`` against ``query``; higher means more relevant. One score per text."""

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker(Reranker):
    """Sentence-Transformers cross-encoder, loaded on first use so construction stays cheap."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> Any:
        from sentence_transformers import CrossEncoder  # heavy import, deferred on purpose

        started = time.perf_counter()
        model = CrossEncoder(self.model_name, device=resolve_device(self.device))
        log.info(
            "reranker.loaded",
            model_name=self.model_name,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )
        return model

    @property
    def model(self) -> Any:
        """The underlying ``CrossEncoder``, loaded exactly once even under concurrent calls."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, text) for text in texts])
        return [float(s) for s in np.asarray(scores, dtype=np.float32).reshape(-1)]


# --------------------------------------------------------------------------------------------
# Retrieval service
# --------------------------------------------------------------------------------------------


class RetrievalService:
    """Dense / hybrid retrieval for a workspace with optional cross-encoder reranking.

    An explicitly passed ``reranker`` is always used. Without one, a
    :class:`CrossEncoderReranker` is created from settings only when ``rerank_enabled`` is on.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        store: MetadataStore,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._vector_store = vector_store
        self._embedder = embedder
        if reranker is None and settings.rerank_enabled:
            reranker = CrossEncoderReranker(settings.rerank_model, device=settings.embedding_device)
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    # ---- public API ------------------------------------------------------------------

    def retrieve(
        self,
        workspace_id: str,
        query: str,
        *,
        k: int | None = None,
        document_ids: list[str] | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[ScoredChunk]:
        """Return the top ``k`` chunks of ``workspace_id`` for ``query``, best first.

        ``k`` defaults to ``settings.top_k`` and is clamped to ``settings.max_top_k``. ``mode``
        defaults to ``settings.retrieval_mode``. ``document_ids`` restricts results to those
        documents; ids outside the workspace match nothing and an empty list means no filter.
        Blank queries and unknown or empty workspaces yield ``[]``.
        """
        started = time.perf_counter()
        top_k = self._resolve_k(k)
        retrieval_mode = RetrievalMode(mode or self._settings.retrieval_mode)
        text = query.strip()
        if not text or self._store.count_chunks(workspace_id) == 0:
            return []

        doc_filter = list(document_ids) if document_ids else None
        allowed = self._allowed_vector_ids(workspace_id, doc_filter)
        if allowed is not None and not allowed:
            return []

        candidates = self._candidate_count(top_k, retrieval_mode)
        dense = self._dense_search(workspace_id, text, candidates, allowed)
        sparse: list[Hit] = []
        if retrieval_mode is RetrievalMode.HYBRID:
            sparse = self._store.keyword_search(workspace_id, text, candidates, doc_filter)

        ranked = self._rank(dense, sparse, retrieval_mode)
        keep = self._settings.rerank_candidates if self._reranker is not None else top_k
        results = self._hydrate(workspace_id, ranked[:keep], dict(dense), sparse)
        if self._reranker is not None and results:
            results = self._rerank(text, results)
        results = results[:top_k]

        log.info(
            "retrieval.completed",
            workspace_id=workspace_id,
            mode=retrieval_mode.value,
            k=top_k,
            dense_hits=len(dense),
            sparse_hits=len(sparse),
            results=len(results),
            reranked=self._reranker is not None,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        return results

    def as_langchain_retriever(
        self,
        workspace_id: str,
        *,
        k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> WorkspaceRetriever:
        """Wrap this service as an LCEL-compatible retriever bound to one workspace."""
        return WorkspaceRetriever(
            service=self, workspace_id=workspace_id, k=k, document_ids=document_ids
        )

    # ---- internals -------------------------------------------------------------------

    def _resolve_k(self, k: int | None) -> int:
        requested = k if k is not None and k > 0 else self._settings.top_k
        return max(1, min(requested, self._settings.max_top_k))

    def _candidate_count(self, top_k: int, mode: RetrievalMode) -> int:
        """How many hits each retriever fetches before fusion / reranking."""
        count = top_k
        if mode is RetrievalMode.HYBRID:
            count *= self._settings.hybrid_candidate_multiplier
        if self._reranker is not None:
            count = max(count, self._settings.rerank_candidates)
        return count

    def _allowed_vector_ids(
        self, workspace_id: str, document_ids: list[str] | None
    ) -> set[int] | None:
        """FAISS ids belonging to ``document_ids`` within the workspace; ``None`` = no filter."""
        if document_ids is None:
            return None
        return set(self._store.list_vector_ids(workspace_id, document_ids))

    def _dense_search(
        self, workspace_id: str, query: str, k: int, allowed: set[int] | None
    ) -> list[Hit]:
        query_vector = self._embedder.embed_query(query)
        hits = self._vector_store.search(workspace_id, query_vector, k, allowed)
        threshold = self._settings.min_dense_score
        if threshold > 0:
            hits = [(vid, score) for vid, score in hits if score >= threshold]
        return hits

    def _rank(self, dense: list[Hit], sparse: list[Hit], mode: RetrievalMode) -> list[Hit]:
        """Candidate ``(vector_id, score)`` pairs best first: cosine (dense) or RRF (hybrid)."""
        if mode is RetrievalMode.DENSE:
            return list(dense)
        dense_scores = dict(dense)
        fused = reciprocal_rank_fusion(
            [[vid for vid, _ in dense], [vid for vid, _ in sparse]], k=self._settings.rrf_k
        )
        # Ties on the fused score go to the better dense score, then to the lower id (stable).
        order = sorted(fused, key=lambda vid: (-fused[vid], -dense_scores.get(vid, -math.inf), vid))
        return [(vid, fused[vid]) for vid in order]

    def _hydrate(
        self,
        workspace_id: str,
        ranked: list[Hit],
        dense_scores: dict[int, float],
        sparse: list[Hit],
    ) -> list[ScoredChunk]:
        """Load chunk rows for the ranked ids, dropping ids SQLite no longer knows (drift)."""
        sparse_ranks = {vid: rank for rank, (vid, _) in enumerate(sparse, start=1)}
        chunks = self._store.get_chunks(workspace_id, [vid for vid, _ in ranked])
        results: list[ScoredChunk] = []
        missing = 0
        for vid, score in ranked:
            chunk = chunks.get(vid)
            if chunk is None or chunk.workspace_id != workspace_id:
                missing += 1
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    vector_id=vid,
                    score=score,
                    dense_score=dense_scores.get(vid),
                    sparse_rank=sparse_ranks.get(vid),
                )
            )
        if missing:
            log.warning(
                "retrieval.index_db_drift", workspace_id=workspace_id, missing_chunks=missing
            )
        return results

    def _rerank(self, query: str, results: list[ScoredChunk]) -> list[ScoredChunk]:
        """Re-score with the cross-encoder; the rerank score becomes the ranking score."""
        if self._reranker is None:  # pragma: no cover - guarded by the caller
            return results
        scores = self._reranker.rerank(query, [sc.chunk.text for sc in results])
        if len(scores) != len(results):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(results)} candidates"
            )
        reranked = [
            sc.model_copy(update={"rerank_score": float(score), "score": float(score)})
            for sc, score in zip(results, scores, strict=True)
        ]
        reranked.sort(key=lambda sc: sc.score, reverse=True)
        return reranked


# --------------------------------------------------------------------------------------------
# LangChain adapter
# --------------------------------------------------------------------------------------------


class WorkspaceRetriever(BaseRetriever):
    """LCEL-compatible retriever pinned to one workspace (``retriever | prompt | llm``).

    The bound service is excluded from serialisation and repr; ``k`` and ``document_ids`` are
    forwarded to :meth:`RetrievalService.retrieve` unchanged.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    service: Any = Field(exclude=True, repr=False, description="The RetrievalService to query.")
    workspace_id: str
    k: int | None = None
    document_ids: list[str] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks = self.service.retrieve(
            self.workspace_id, query, k=self.k, document_ids=self.document_ids
        )
        return [scored_chunk_to_document(sc) for sc in chunks]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return await run_in_executor(
            None, self._get_relevant_documents, query, run_manager=run_manager.get_sync()
        )
