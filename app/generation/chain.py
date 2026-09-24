"""Answer generation service: retrieval -> grounded LLM answer -> verified citations.

The LCEL chain is ``ANSWER_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_llm_answer)``.
Post-processing verifies every cited label against the chunks that were actually shown to the
model, abstains honestly when evidence is missing, and falls back to an extractive answer
(top passages, each cited) when no LLM is configured.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from app.core.config import Settings
from app.generation.citations import LLMAnswer, build_citations, make_excerpt, parse_llm_answer
from app.generation.prompts import ANSWER_PROMPT, NO_ANSWER_TEXT, format_context
from app.models.domain import ScoredChunk
from app.models.schemas import (
    AnswerMode,
    Citation,
    QueryRequest,
    QueryResponse,
    RetrievalMode,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    Timings,
)

if TYPE_CHECKING:
    from app.retrieval.retriever import RetrievalService

logger = structlog.get_logger(__name__)

EXTRACTIVE_HEADER = (
    "No LLM is configured, so here are the most relevant passages instead of a synthesized answer:"
)
_EXTRACTIVE_PASSAGES = 3
_EXPLANATION_CHARS = 300


_SECRET_RE = re.compile(r"(?i)(bearer\s+\S+|\b(?:sk|gsk|xai|hf)[-_][A-Za-z0-9_-]{8,})")
_REASON_CHARS = 300


def sanitize_reason(exc: BaseException) -> str:
    """Provider error text safe to expose: secrets redacted, whitespace collapsed, capped."""
    text = " ".join(str(exc).split()) or "no details"
    text = _SECRET_RE.sub("<redacted>", text)
    return text if len(text) <= _REASON_CHARS else text[: _REASON_CHARS - 3] + "..."


class GenerationError(Exception):
    """Raised when the LLM call fails. The API layer maps it to HTTP 502.

    ``error_type`` is the provider exception class (e.g. ``NotFoundError``) and ``reason`` a
    sanitised copy of its message, so operators and users can see *why* (a retired model, a
    bad key, a rate limit) without leaking secrets.
    """

    def __init__(self, error_type: str, reason: str = "") -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.reason = reason


def _elapsed_ms(start: float) -> float:
    return max(0.0, (time.perf_counter() - start) * 1000.0)


def _to_retrieved(sc: ScoredChunk) -> RetrievedChunk:
    """Convert an internal ``ScoredChunk`` into the public debug representation."""
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


def _build_chain(llm: BaseChatModel) -> Runnable[dict[str, str], LLMAnswer]:
    return ANSWER_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_llm_answer)


def _extractive_answer(
    chunks: list[ScoredChunk], *, excerpt_chars: int, focus: str
) -> tuple[str, list[Citation]]:
    """Header plus the top passages, each labelled and cited (used when no LLM is configured).

    Excerpts are focused on the question so the passage shown is the one that answers it.
    """
    top = chunks[:_EXTRACTIVE_PASSAGES]
    label_map = {f"S{index}": sc for index, sc in enumerate(top, start=1)}
    citations, _ = build_citations(
        list(label_map), label_map, excerpt_chars=excerpt_chars, focus=focus
    )
    lines = [
        f"[{c.citation_id}] ({c.filename}, p. {c.page_number}): {c.excerpt}" for c in citations
    ]
    return EXTRACTIVE_HEADER + "\n\n" + "\n\n".join(lines), citations


def _abstention_text(parsed: LLMAnswer) -> str:
    """No-answer message, with the model's brief explanation appended when it gave one."""
    explanation = parsed.answer.strip() if parsed.insufficient_evidence else ""
    if not explanation:
        return NO_ANSWER_TEXT
    return f"{NO_ANSWER_TEXT}\n{make_excerpt(explanation, _EXPLANATION_CHARS)}"


class AnswerService:
    """Answers questions over a workspace with verified citations, or searches it raw."""

    def __init__(
        self, *, settings: Settings, retrieval: RetrievalService, llm: BaseChatModel | None
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._llm = llm
        self._chain = _build_chain(llm) if llm is not None else None

    # ---- public API ------------------------------------------------------------------------
    def answer(self, workspace_id: str, request: QueryRequest, *, request_id: str) -> QueryResponse:
        """Retrieve, generate and verify an answer for ``request`` within ``workspace_id``."""
        started = time.perf_counter()
        chunks = self._retrieval.retrieve(
            workspace_id,
            request.question,
            k=request.top_k,
            document_ids=request.document_ids,
            mode=request.mode,
        )
        retrieval_ms = _elapsed_ms(started)
        retrieved = [_to_retrieved(sc) for sc in chunks] if request.include_debug else None
        base = {
            "request_id": request_id,
            "workspace_id": workspace_id,
            "question": request.question,
            "retrieved": retrieved,
        }
        if not chunks:
            response = QueryResponse(
                **base,
                answer=NO_ANSWER_TEXT,
                abstained=True,
                answer_mode=self._answer_mode(),
                citations=[],
                timings=self._timings(started, retrieval_ms, generation_ms=0.0),
            )
        elif self._chain is None:
            answer_text, citations = _extractive_answer(
                chunks,
                excerpt_chars=self._settings.citation_excerpt_chars,
                focus=request.question,
            )
            response = QueryResponse(
                **base,
                answer=answer_text,
                abstained=False,
                answer_mode=AnswerMode.EXTRACTIVE,
                citations=citations,
                timings=self._timings(started, retrieval_ms, generation_ms=0.0),
            )
        else:
            response = self._generate(self._chain, chunks, request, started, retrieval_ms, base)
        self._log_answer(response, chunk_count=len(chunks))
        return response

    def search(
        self, workspace_id: str, request: SearchRequest, *, request_id: str
    ) -> SearchResponse:
        """Raw retrieval without generation (debug-oriented)."""
        started = time.perf_counter()
        chunks = self._retrieval.retrieve(
            workspace_id,
            request.query,
            k=request.top_k,
            document_ids=request.document_ids,
            mode=request.mode,
        )
        retrieval_ms = _elapsed_ms(started)
        mode = request.mode or RetrievalMode(self._settings.retrieval_mode)
        logger.info(
            "search_completed",
            request_id=request_id,
            workspace_id=workspace_id,
            mode=str(mode),
            results=len(chunks),
            retrieval_ms=round(retrieval_ms, 1),
        )
        return SearchResponse(
            request_id=request_id,
            workspace_id=workspace_id,
            query=request.query,
            mode=mode,
            results=[_to_retrieved(sc) for sc in chunks],
            timings=self._timings(started, retrieval_ms, generation_ms=0.0),
        )

    # ---- internals -------------------------------------------------------------------------
    def _generate(
        self,
        chain: Runnable[dict[str, str], LLMAnswer],
        chunks: list[ScoredChunk],
        request: QueryRequest,
        started: float,
        retrieval_ms: float,
        base: dict[str, object],
    ) -> QueryResponse:
        """Run the LCEL chain over the formatted context and verify the citations."""
        context, label_map = format_context(chunks, max_chars=self._settings.max_context_chars)
        generation_started = time.perf_counter()
        try:
            parsed = chain.invoke({"question": request.question, "context": context})
        except Exception as exc:
            reason = sanitize_reason(exc)
            logger.warning(
                "llm_call_failed",
                request_id=base["request_id"],
                error_type=type(exc).__name__,
                reason=reason,
                generation_ms=round(_elapsed_ms(generation_started), 1),
            )
            raise GenerationError(type(exc).__name__, reason) from exc
        generation_ms = _elapsed_ms(generation_started)

        citations, invalid = build_citations(
            parsed.citations,
            label_map,
            excerpt_chars=self._settings.citation_excerpt_chars,
            focus=request.question,
        )
        abstained = parsed.insufficient_evidence or not citations
        if abstained:
            answer_text = _abstention_text(parsed)
            citations = []
        else:
            answer_text = parsed.answer
        return QueryResponse(
            **base,
            answer=answer_text,
            abstained=abstained,
            answer_mode=AnswerMode.LLM,
            citations=citations,
            invalid_citation_ids=invalid,
            model=self._settings.resolved_llm_model(),
            timings=self._timings(started, retrieval_ms, generation_ms=generation_ms),
        )

    def _answer_mode(self) -> AnswerMode:
        return AnswerMode.LLM if self._llm is not None else AnswerMode.EXTRACTIVE

    @staticmethod
    def _timings(started: float, retrieval_ms: float, *, generation_ms: float) -> Timings:
        return Timings(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=_elapsed_ms(started),
        )

    @staticmethod
    def _log_answer(response: QueryResponse, *, chunk_count: int) -> None:
        logger.info(
            "answer_completed",
            request_id=response.request_id,
            workspace_id=response.workspace_id,
            answer_mode=str(response.answer_mode),
            abstained=response.abstained,
            retrieved=chunk_count,
            citations=len(response.citations),
            invalid_citations=len(response.invalid_citation_ids),
            retrieval_ms=round(response.timings.retrieval_ms, 1),
            generation_ms=round(response.timings.generation_ms, 1),
            total_ms=round(response.timings.total_ms, 1),
        )
