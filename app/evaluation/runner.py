"""End-to-end evaluation runner: ingest the dataset corpus, ask every question, score, report.

The runner builds the same service stack the API uses (metadata store, embedder, FAISS store,
retrieval, ingestion and answer service) directly from ``Settings`` so it never depends on the
HTTP layer. It indexes the dataset's documents into a dedicated workspace, then for each
question:

* runs raw retrieval (``AnswerService.search``) with ``top_k = max(k_values)`` and scores
  recall@k / hit@k / MRR over ``(filename, page)`` pairs;
* runs the full answer path (``AnswerService.answer`` with ``include_debug=True``) and scores
  citation validity, citation page accuracy, keyword-based answer correctness, abstention and
  the ``must_not_contain`` hard check, recording latencies.

Results are aggregated into an :class:`EvaluationReport` that can be written as JSON and as a
Markdown summary. All metric definitions live in :mod:`app.evaluation.metrics`; this module
only orchestrates and formats. Only identifiers, counts and durations are logged.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import get_logger
from app.evaluation.metrics import (
    SourceRef,
    abstention_metrics,
    answer_correctness,
    citation_page_accuracy,
    citation_validity,
    forbidden_terms_found,
    hit_at_k,
    latency_summary,
    mrr,
    normalize_for_matching,
    recall_at_k,
)
from app.generation.chain import AnswerService
from app.generation.llm import create_chat_model
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import (
    AnswerMode,
    DocumentUploadResult,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    UploadOutcome,
)
from app.retrieval.embeddings import Embedder, create_embedder
from app.retrieval.retriever import RetrievalService
from app.retrieval.vector_store import VectorStore
from app.storage.metadata_store import MetadataStore

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE_ID = "eval"
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
QuestionType = Literal["answerable", "unanswerable", "cross_document"]
QUESTION_TYPES: tuple[str, ...] = ("answerable", "cross_document", "unanswerable")
LlmSpec = BaseChatModel | None | Literal["auto"]

_ANSWER_PREVIEW_CHARS = 200
_MISSED_PAGES_SHOWN = 3
_GIT_TIMEOUT_SECONDS = 5.0
_MAX_SEARCH_K = 50  # SearchRequest.top_k upper bound


class EvaluationError(RuntimeError):
    """Raised when the evaluation cannot produce trustworthy results (e.g. ingestion failed)."""


# --------------------------------------------------------------------------------------------
# Dataset models
# --------------------------------------------------------------------------------------------


class ExpectedSource(BaseModel):
    """One page that holds (part of) the answer to a question."""

    filename: str
    page: int = Field(ge=1)

    @property
    def pair(self) -> SourceRef:
        return (self.filename, self.page)


class EvalQuestion(BaseModel):
    """One dataset question. ``must_not_contain`` is a hard check on the final answer text."""

    id: str
    question: str = Field(min_length=3)
    type: QuestionType
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    notes: str = ""
    must_not_contain: list[str] = Field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    @property
    def expected_pairs(self) -> list[SourceRef]:
        return [source.pair for source in self.expected_sources]


class EvalDataset(BaseModel):
    """The evaluation dataset (``evaluation/dataset.json``)."""

    version: int = 1
    documents: list[str] = Field(min_length=1)
    questions: list[EvalQuestion] = Field(min_length=1)
    source_path: str | None = Field(default=None, exclude=True)


def load_dataset(path: Path) -> EvalDataset:
    """Read and validate a dataset file. Raises ``FileNotFoundError`` / ``ValueError``."""
    dataset_path = Path(path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{dataset_path} is not valid JSON: {exc}") from exc
    dataset = EvalDataset.model_validate(payload)
    ids = [question.id for question in dataset.questions]
    if len(set(ids)) != len(ids):
        raise ValueError("dataset question ids must be unique")
    return dataset.model_copy(update={"source_path": _display_path(dataset_path)})


def _display_path(path: Path) -> str:
    """``path`` relative to the repository root when it lies inside it (keeps reports portable)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------
# Result models
# --------------------------------------------------------------------------------------------


class PageRef(BaseModel):
    filename: str
    page: int


class LatencyStats(BaseModel):
    p50: float
    p95: float
    mean: float
    max: float
    n: int


class QuestionResult(BaseModel):
    """Everything measured for one question."""

    id: str
    type: QuestionType
    question: str
    answerable: bool
    retrieved_pages: list[PageRef] = Field(
        description="Distinct (filename, page) pairs from the top-max(k) retrieval, best first."
    )
    recall_at_k: dict[int, float] | None = Field(
        description="None for questions without expected sources (unanswerable)."
    )
    hit_at_k: dict[int, bool] | None
    mrr: float | None
    citation_validity: float
    citation_page_accuracy: float | None
    answer_correctness: float
    abstained: bool
    answer_mode: AnswerMode
    citations: list[PageRef]
    invalid_citation_ids: list[str]
    forbidden_terms_found: list[str]
    injection_check_passed: bool | None = Field(
        description="None when the question has no must_not_contain terms."
    )
    search_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    answer_preview: str
    notes: str


class TypeSummary(BaseModel):
    """Aggregate metrics for the questions of one type."""

    type: str
    count: int
    recall_at_k: dict[int, float] | None
    hit_rate_at_k: dict[int, float] | None
    mrr: float | None
    citation_validity: float | None
    citation_page_accuracy: float | None
    answer_correctness: float | None
    abstention_rate: float | None


class AggregateMetrics(BaseModel):
    """Dataset-level metrics. Ratios with an empty denominator are ``None``."""

    question_count: int
    retrieval_question_count: int = Field(description="Questions with expected sources.")
    answerable_count: int
    unanswerable_count: int
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float | None
    citation_validity: float | None
    citation_page_accuracy: float | None = Field(description="Over questions with citations.")
    answer_correctness: float | None = Field(description="Over answerable questions.")
    abstention_accuracy: float | None = Field(description="Unanswerable questions abstained.")
    false_abstention_rate: float | None = Field(description="Answerable questions abstained.")
    abstention_precision: float | None
    abstention_recall: float | None
    injection_check_count: int
    injection_check_pass_rate: float | None
    latency_ms: dict[str, LatencyStats] = Field(
        description="Keys: retrieval, generation, total (answer path) and search."
    )


class IngestedDocument(BaseModel):
    filename: str
    outcome: UploadOutcome
    page_count: int
    chunk_count: int
    processing_ms: float | None


class RunConfig(BaseModel):
    """Everything needed to interpret and reproduce a run."""

    timestamp: str
    git_commit: str | None
    dataset_path: str | None
    dataset_version: int
    document_count: int
    question_count: int
    question_limit: int | None
    workspace_id: str
    data_dir: str
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    retrieval_mode: str
    top_k: int
    k_values: list[int]
    rerank_enabled: bool
    rerank_model: str | None
    llm_provider: str
    llm_model: str | None
    answer_mode: AnswerMode
    ingestion: list[IngestedDocument]


class EvaluationReport(BaseModel):
    config: RunConfig
    aggregate: AggregateMetrics
    by_type: list[TypeSummary]
    questions: list[QuestionResult]


# --------------------------------------------------------------------------------------------
# Service construction (kept here so the runner never imports the API layer)
# --------------------------------------------------------------------------------------------


@dataclass
class EvalServices:
    """The service stack used by the runner; ``close`` releases the SQLite connection."""

    settings: Settings
    store: MetadataStore
    vector_store: VectorStore
    embedder: Embedder
    retrieval: RetrievalService
    ingestion: IngestionPipeline
    answer: AnswerService
    llm: BaseChatModel | None
    llm_provider: str
    llm_model_name: str | None

    @property
    def answer_mode(self) -> AnswerMode:
        return AnswerMode.LLM if self.llm is not None else AnswerMode.EXTRACTIVE

    def close(self) -> None:
        self.store.close()


def _resolve_llm(settings: Settings, llm: LlmSpec) -> tuple[BaseChatModel | None, str, str | None]:
    """Return ``(chat_model, provider_label, model_name)`` for the requested LLM spec."""
    if llm == "auto":
        provider, model_name = settings.resolved_llm_provider(), settings.resolved_llm_model()
        return create_chat_model(settings), provider, model_name
    if llm is None:
        return None, "extractive", None
    model_name = settings.resolved_llm_model() or type(llm).__name__
    return llm, "custom", model_name


def build_eval_services(
    settings: Settings, embedder: Embedder | None = None, llm: LlmSpec = "auto"
) -> EvalServices:
    """Construct the full stack from ``settings``.

    ``embedder=None`` loads the configured embedding model. ``llm="auto"`` resolves the chat
    model from settings (``None`` = extractive); pass a model instance to override, or ``None``
    to force extractive mode.
    """
    resolved_embedder = embedder if embedder is not None else create_embedder(settings)
    store = MetadataStore(settings.db_path)
    try:
        vector_store = VectorStore(settings.index_dir, resolved_embedder.dimension)
        retrieval = RetrievalService(
            settings=settings, store=store, vector_store=vector_store, embedder=resolved_embedder
        )
        ingestion = IngestionPipeline(
            settings=settings, store=store, vector_store=vector_store, embedder=resolved_embedder
        )
        chat_model, provider, model_name = _resolve_llm(settings, llm)
        answer = AnswerService(settings=settings, retrieval=retrieval, llm=chat_model)
    except Exception:
        store.close()
        raise
    return EvalServices(
        settings=settings,
        store=store,
        vector_store=vector_store,
        embedder=resolved_embedder,
        retrieval=retrieval,
        ingestion=ingestion,
        answer=answer,
        llm=chat_model,
        llm_provider=provider,
        llm_model_name=model_name,
    )


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _dedupe_pairs(pairs: Sequence[SourceRef]) -> list[SourceRef]:
    return list(dict.fromkeys(pairs))


def _short_name(filename: str) -> str:
    return filename[:-4] if filename.lower().endswith(".pdf") else filename


def _page_label(pair: SourceRef) -> str:
    return f"{_short_name(pair[0])} p{pair[1]}"


def _missing_keywords(answer: str, keywords: Sequence[str]) -> list[str]:
    haystack = normalize_for_matching(answer)
    return [k for k in dict.fromkeys(keywords) if normalize_for_matching(k) not in haystack]


def _git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """Short hash of HEAD, or ``None`` when git or the repository is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _validate_k_values(k_values: Sequence[int], max_top_k: int) -> tuple[int, ...]:
    values = tuple(sorted({int(k) for k in k_values}))
    if not values or values[0] < 1:
        raise ValueError("k_values must contain positive integers")
    ceiling = min(max_top_k, _MAX_SEARCH_K)
    if values[-1] > ceiling:
        raise ValueError(
            f"max(k_values)={values[-1]} exceeds the retrieval depth limit {ceiling} "
            f"(settings.max_top_k={max_top_k}, API cap {_MAX_SEARCH_K}); lower k_values"
        )
    return values


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


class EvaluationRunner:
    """Run the dataset end to end against one configuration.

    The stack is built from ``settings`` on :meth:`run` and closed afterwards. ``embedder``
    defaults to the configured model; ``llm`` follows :func:`build_eval_services`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        llm: LlmSpec = "auto",
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._llm = llm
        self._k_values = _validate_k_values(k_values, settings.max_top_k)
        self._workspace_id = workspace_id

    @property
    def k_values(self) -> tuple[int, ...]:
        return self._k_values

    @property
    def max_k(self) -> int:
        return self._k_values[-1]

    # ---- public API ------------------------------------------------------------------

    def run(
        self,
        dataset: EvalDataset,
        *,
        documents_root: Path | None = None,
        limit: int | None = None,
        on_question: Callable[[QuestionResult], None] | None = None,
    ) -> EvaluationReport:
        """Ingest the corpus, evaluate the (first ``limit``) questions and build the report.

        Relative document paths are resolved against ``documents_root`` (default: repo root).
        ``on_question`` is called after each question, e.g. for CLI progress output.
        """
        started = time.perf_counter()
        services = build_eval_services(self._settings, embedder=self._embedder, llm=self._llm)
        try:
            self._reset_workspace(services)
            ingested = self._ingest_documents(services, dataset, documents_root or REPO_ROOT)
            questions = dataset.questions[:limit] if limit is not None else dataset.questions
            results: list[QuestionResult] = []
            for question in questions:
                result = self._evaluate_question(services, question)
                results.append(result)
                if on_question is not None:
                    on_question(result)
            report = EvaluationReport(
                config=self._build_config(services, dataset, ingested, limit),
                aggregate=aggregate_results(results, self._k_values),
                by_type=summarize_by_type(results, self._k_values),
                questions=results,
            )
        finally:
            services.close()
        log.info(
            "evaluation.completed",
            workspace_id=self._workspace_id,
            questions=len(report.questions),
            documents=len(report.config.ingestion),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )
        return report

    # ---- corpus ----------------------------------------------------------------------

    def _reset_workspace(self, services: EvalServices) -> None:
        """Make the evaluation workspace empty so a reused data_dir cannot skew results."""
        existing = services.store.list_documents(self._workspace_id)
        for doc in existing:
            services.ingestion.delete_document(self._workspace_id, doc.document_id)
        services.vector_store.drop_workspace(self._workspace_id)
        if existing:
            log.info(
                "evaluation.workspace_reset",
                workspace_id=self._workspace_id,
                documents_removed=len(existing),
            )

    def _ingest_documents(
        self, services: EvalServices, dataset: EvalDataset, documents_root: Path
    ) -> list[IngestedDocument]:
        ingested: list[IngestedDocument] = []
        failures: list[str] = []
        for relative in dataset.documents:
            path = Path(relative)
            if not path.is_absolute():
                path = documents_root / path
            if not path.is_file():
                raise FileNotFoundError(f"dataset document not found: {path}")
            result = services.ingestion.ingest(path.read_bytes(), path.name, self._workspace_id)
            ingested.append(_to_ingested(result))
            if result.outcome is UploadOutcome.FAILED:
                failures.append(f"{path.name}: {result.message}")
        if failures:
            raise EvaluationError("document ingestion failed: " + "; ".join(failures))
        log.info(
            "evaluation.ingested",
            workspace_id=self._workspace_id,
            documents=len(ingested),
            chunks=sum(doc.chunk_count for doc in ingested),
        )
        return ingested

    # ---- one question ----------------------------------------------------------------

    def _evaluate_question(self, services: EvalServices, question: EvalQuestion) -> QuestionResult:
        workspace = self._workspace_id
        search = services.answer.search(
            workspace,
            SearchRequest(query=question.question, top_k=self.max_k),
            request_id=f"eval-{question.id}-search",
        )
        retrieved_pairs = _dedupe_pairs([(r.filename, r.page_number) for r in search.results])

        response = services.answer.answer(
            workspace,
            QueryRequest(
                question=question.question, top_k=self._settings.top_k, include_debug=True
            ),
            request_id=f"eval-{question.id}-answer",
        )
        result = self._score(question, retrieved_pairs, search.timings.retrieval_ms, response)
        log.info(
            "evaluation.question_completed",
            question_id=question.id,
            question_type=question.type,
            hit=(result.hit_at_k or {}).get(self.max_k),
            abstained=result.abstained,
            citations=len(result.citations),
            injection_check_passed=result.injection_check_passed,
            total_ms=round(result.total_ms, 1),
        )
        return result

    def _score(
        self,
        question: EvalQuestion,
        retrieved_pairs: list[SourceRef],
        search_ms: float,
        response: QueryResponse,
    ) -> QuestionResult:
        expected = question.expected_pairs
        scored_retrieval = bool(expected)
        recall = {k: recall_at_k(retrieved_pairs, expected, k) for k in self._k_values}
        hits = {k: hit_at_k(retrieved_pairs, expected, k) for k in self._k_values}

        retrieved_ids = {r.chunk_id for r in response.retrieved or []}
        forbidden = forbidden_terms_found(response.answer, question.must_not_contain)
        correctness = answer_correctness(
            response.answer,
            question.expected_keywords,
            abstained=response.abstained,
            answerable=question.answerable,
        )
        page_accuracy = (
            citation_page_accuracy(response.citations, expected) if scored_retrieval else None
        )
        result = QuestionResult(
            id=question.id,
            type=question.type,
            question=question.question,
            answerable=question.answerable,
            retrieved_pages=[PageRef(filename=f, page=p) for f, p in retrieved_pairs],
            recall_at_k=recall if scored_retrieval else None,
            hit_at_k=hits if scored_retrieval else None,
            mrr=mrr(retrieved_pairs, expected) if scored_retrieval else None,
            citation_validity=citation_validity(response.citations, retrieved_ids),
            citation_page_accuracy=page_accuracy,
            answer_correctness=correctness,
            abstained=response.abstained,
            answer_mode=response.answer_mode,
            citations=[
                PageRef(filename=c.filename, page=c.page_number) for c in response.citations
            ],
            invalid_citation_ids=list(response.invalid_citation_ids),
            forbidden_terms_found=forbidden,
            injection_check_passed=(not forbidden) if question.must_not_contain else None,
            search_ms=search_ms,
            retrieval_ms=response.timings.retrieval_ms,
            generation_ms=response.timings.generation_ms,
            total_ms=response.timings.total_ms,
            answer_preview=_preview(response.answer),
            notes="",
        )
        result.notes = _describe(question, result, response.answer, self._settings.top_k)
        return result

    # ---- config ----------------------------------------------------------------------

    def _build_config(
        self,
        services: EvalServices,
        dataset: EvalDataset,
        ingested: list[IngestedDocument],
        limit: int | None,
    ) -> RunConfig:
        settings = self._settings
        return RunConfig(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            git_commit=_git_commit(),
            dataset_path=dataset.source_path,
            dataset_version=dataset.version,
            document_count=len(dataset.documents),
            question_count=len(dataset.questions),
            question_limit=limit,
            workspace_id=self._workspace_id,
            data_dir=str(settings.data_dir),
            embedding_model=services.embedder.model_name,
            embedding_dimension=services.embedder.dimension,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            retrieval_mode=settings.retrieval_mode,
            top_k=settings.top_k,
            k_values=list(self._k_values),
            rerank_enabled=services.retrieval.reranker is not None,
            rerank_model=settings.rerank_model if services.retrieval.reranker else None,
            llm_provider=services.llm_provider,
            llm_model=services.llm_model_name,
            answer_mode=services.answer_mode,
            ingestion=ingested,
        )


def _to_ingested(result: DocumentUploadResult) -> IngestedDocument:
    doc = result.document
    return IngestedDocument(
        filename=result.filename,
        outcome=result.outcome,
        page_count=doc.page_count if doc else 0,
        chunk_count=doc.chunk_count if doc else 0,
        processing_ms=result.processing_ms,
    )


def _preview(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _ANSWER_PREVIEW_CHARS:
        return collapsed
    return collapsed[: _ANSWER_PREVIEW_CHARS - 3].rstrip() + "..."


def _describe(question: EvalQuestion, result: QuestionResult, answer: str, top_k: int) -> str:
    """Short human-readable diagnosis used in the per-question table."""
    parts: list[str] = []
    if result.forbidden_terms_found:
        parts.append("must_not_contain violated: " + ", ".join(result.forbidden_terms_found))
    if result.invalid_citation_ids:
        parts.append("invalid citation labels dropped: " + ", ".join(result.invalid_citation_ids))
    if not question.answerable:
        parts.append("abstained" if result.abstained else "answered instead of abstaining")
        return "; ".join(parts)

    parts.extend(_retrieval_notes(question, result, top_k))
    if result.abstained:
        parts.append("abstained on an answerable question")
    elif result.answer_correctness < 1.0:
        missing = _missing_keywords(answer, question.expected_keywords)
        parts.append("keywords not in answer: " + ", ".join(missing))
    return "; ".join(parts) or "ok"


def _retrieval_notes(question: EvalQuestion, result: QuestionResult, top_k: int) -> list[str]:
    """Explain retrieval misses: expected pages never retrieved, or ranked below ``top_k``."""
    if result.hit_at_k is None:
        return []
    max_k = max(result.hit_at_k)
    retrieved = [(p.filename, p.page) for p in result.retrieved_pages]
    ranks = {
        pair: retrieved.index(pair) + 1 for pair in question.expected_pairs if pair in retrieved
    }
    not_retrieved = [pair for pair in question.expected_pairs if pair not in ranks]
    late = [(pair, rank) for pair, rank in ranks.items() if rank > top_k]
    notes: list[str] = []
    if not_retrieved:
        notes.append(f"not in top-{max_k}: " + ", ".join(_page_label(p) for p in not_retrieved))
    if late:
        ranked = ", ".join(f"{_page_label(p)} at rank {r}" for p, r in late)
        notes.append(f"below top-{top_k}: {ranked}")
    if notes and not any(rank <= top_k for rank in ranks.values()):
        top = ", ".join(_page_label(p) for p in retrieved[:_MISSED_PAGES_SHOWN])
        notes.append(f"top retrieved: {top or 'nothing'}")
    return notes


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------


def _latency(values: Sequence[float]) -> LatencyStats:
    return LatencyStats(**latency_summary(values))


def _retrieval_means(
    results: Sequence[QuestionResult], k_values: Sequence[int]
) -> tuple[dict[int, float], dict[int, float], float | None]:
    """Mean recall@k, hit rate@k and MRR over the questions that have expected sources."""
    scored = [r for r in results if r.recall_at_k is not None and r.hit_at_k is not None]
    if not scored:
        return {}, {}, None
    count = len(scored)
    recall = {k: sum(r.recall_at_k[k] for r in scored) / count for k in k_values}
    hit_rate = {k: sum(1 for r in scored if r.hit_at_k[k]) / count for k in k_values}
    return recall, hit_rate, _mean([r.mrr for r in scored if r.mrr is not None])


def aggregate_results(
    results: Sequence[QuestionResult], k_values: Sequence[int]
) -> AggregateMetrics:
    """Roll per-question results up into dataset-level metrics."""
    recall, hit_rate, mean_mrr = _retrieval_means(results, k_values)
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    with_page_accuracy = [
        r.citation_page_accuracy for r in results if r.citation_page_accuracy is not None
    ]
    injection_checks = [r for r in results if r.injection_check_passed is not None]
    abstention = abstention_metrics((r.abstained, r.answerable) for r in results)
    return AggregateMetrics(
        question_count=len(results),
        retrieval_question_count=sum(1 for r in results if r.recall_at_k is not None),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        recall_at_k=recall,
        hit_rate_at_k=hit_rate,
        mrr=mean_mrr,
        citation_validity=_mean([r.citation_validity for r in results]),
        citation_page_accuracy=_mean(with_page_accuracy),
        answer_correctness=_mean([r.answer_correctness for r in answerable]),
        abstention_accuracy=_ratio(sum(r.abstained for r in unanswerable), len(unanswerable)),
        false_abstention_rate=_ratio(sum(r.abstained for r in answerable), len(answerable)),
        abstention_precision=abstention["precision"],
        abstention_recall=abstention["recall"],
        injection_check_count=len(injection_checks),
        injection_check_pass_rate=_mean(
            [float(bool(r.injection_check_passed)) for r in injection_checks]
        ),
        latency_ms={
            "retrieval": _latency([r.retrieval_ms for r in results]),
            "generation": _latency([r.generation_ms for r in results]),
            "total": _latency([r.total_ms for r in results]),
            "search": _latency([r.search_ms for r in results]),
        },
    )


def summarize_by_type(
    results: Sequence[QuestionResult], k_values: Sequence[int]
) -> list[TypeSummary]:
    """One :class:`TypeSummary` per question type present, in canonical order."""
    summaries: list[TypeSummary] = []
    for question_type in QUESTION_TYPES:
        group = [r for r in results if r.type == question_type]
        if not group:
            continue
        recall, hit_rate, mean_mrr = _retrieval_means(group, k_values)
        page_accuracy = [
            r.citation_page_accuracy for r in group if r.citation_page_accuracy is not None
        ]
        summaries.append(
            TypeSummary(
                type=question_type,
                count=len(group),
                recall_at_k=recall or None,
                hit_rate_at_k=hit_rate or None,
                mrr=mean_mrr,
                citation_validity=_mean([r.citation_validity for r in group]),
                citation_page_accuracy=_mean(page_accuracy),
                answer_correctness=_mean([r.answer_correctness for r in group]),
                abstention_rate=_ratio(sum(r.abstained for r in group), len(group)),
            )
        )
    return summaries


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------

EXTRACTIVE_NOTICE = (
    "**Extractive mode (no LLM).** The answer step returned the top passages verbatim instead "
    "of a synthesized answer, so *answer correctness* here means the expected keywords appear "
    "in the returned excerpts (each truncated to `citation_excerpt_chars`), not that a "
    "question was answered. In this mode the system only abstains when nothing is retrieved, "
    "so *abstention accuracy* on unanswerable questions is expected to be 0 and *false "
    "abstention* 0; use the retrieval and citation metrics to compare configurations."
)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_bool(value: bool | None) -> str:
    return "n/a" if value is None else ("yes" if value else "no")


def _cell(text: str) -> str:
    """Escape a value for a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows)
    return lines


def _report_k(report: EvaluationReport) -> int:
    """The k shown in the per-question table: ``top_k`` when evaluated, else the largest k."""
    k_values = report.config.k_values
    return report.config.top_k if report.config.top_k in k_values else max(k_values)


def _questions_evaluated(config: RunConfig) -> str:
    if config.question_limit is None:
        return str(config.question_count)
    return f"{min(config.question_limit, config.question_count)} of {config.question_count}"


def _config_rows(config: RunConfig) -> list[list[str]]:
    return [
        ["Timestamp (UTC)", config.timestamp],
        ["Git commit", config.git_commit or "n/a"],
        ["Dataset", f"{config.dataset_path or 'inline'} (v{config.dataset_version})"],
        ["Questions evaluated", _questions_evaluated(config)],
        ["Documents", str(config.document_count)],
        ["Embedding model", f"{config.embedding_model} ({config.embedding_dimension}-d)"],
        ["Chunking", f"{config.chunk_size} chars, overlap {config.chunk_overlap}"],
        ["Retrieval mode", config.retrieval_mode],
        ["top_k (answer path)", str(config.top_k)],
        ["k values (retrieval metrics)", ", ".join(str(k) for k in config.k_values)],
        ["Reranker", config.rerank_model if config.rerank_enabled else "disabled"],
        ["LLM provider / model", f"{config.llm_provider} / {config.llm_model or 'none'}"],
        ["Answer mode", config.answer_mode.value],
    ]


def _summary_rows(agg: AggregateMetrics, k_values: Sequence[int]) -> list[list[str]]:
    rows = [[f"Recall@{k}", _fmt(agg.recall_at_k.get(k))] for k in k_values]
    rows += [[f"Hit rate@{k}", _fmt(agg.hit_rate_at_k.get(k))] for k in k_values]
    latency = agg.latency_ms
    rows += [
        ["MRR", _fmt(agg.mrr)],
        ["Citation validity", _fmt(agg.citation_validity)],
        ["Citation page accuracy", _fmt(agg.citation_page_accuracy)],
        ["Answer correctness (answerable)", _fmt(agg.answer_correctness)],
        ["Abstention accuracy (unanswerable)", _fmt(agg.abstention_accuracy)],
        ["False abstention rate (answerable)", _fmt(agg.false_abstention_rate)],
        [
            "Abstention precision / recall",
            f"{_fmt(agg.abstention_precision)} / {_fmt(agg.abstention_recall)}",
        ],
        [
            "Injection checks passed",
            f"{_fmt(agg.injection_check_pass_rate)} ({agg.injection_check_count} checks)",
        ],
        [
            "Retrieval latency p50 / p95 (ms)",
            f"{_fmt(latency['retrieval'].p50, 1)} / {_fmt(latency['retrieval'].p95, 1)}",
        ],
        [
            "Generation latency p50 / p95 (ms)",
            f"{_fmt(latency['generation'].p50, 1)} / {_fmt(latency['generation'].p95, 1)}",
        ],
        [
            "Total latency p50 / p95 (ms)",
            f"{_fmt(latency['total'].p50, 1)} / {_fmt(latency['total'].p95, 1)}",
        ],
    ]
    return rows


def _type_rows(summaries: Sequence[TypeSummary], k: int) -> list[list[str]]:
    return [
        [
            s.type,
            str(s.count),
            _fmt(s.recall_at_k.get(k) if s.recall_at_k else None),
            _fmt(s.hit_rate_at_k.get(k) if s.hit_rate_at_k else None),
            _fmt(s.mrr),
            _fmt(s.citation_validity),
            _fmt(s.citation_page_accuracy),
            _fmt(s.answer_correctness),
            _fmt(s.abstention_rate),
        ]
        for s in summaries
    ]


def _question_rows(results: Sequence[QuestionResult], k: int) -> list[list[str]]:
    return [
        [
            r.id,
            r.type,
            _fmt_bool(r.hit_at_k.get(k) if r.hit_at_k else None),
            _fmt(r.citation_validity, 2),
            _fmt(r.answer_correctness, 2),
            _fmt_bool(r.abstained),
            r.notes,
        ]
        for r in results
    ]


def _ingestion_rows(config: RunConfig) -> list[list[str]]:
    return [
        [
            d.filename,
            d.outcome.value,
            str(d.page_count),
            str(d.chunk_count),
            _fmt(d.processing_ms, 0),
        ]
        for d in config.ingestion
    ]


def render_markdown(report: EvaluationReport) -> str:
    """Render the report as a Markdown document (summary, per-type, per-question, ingestion)."""
    k = _report_k(report)
    lines = ["# ContextIQ evaluation report", ""]
    lines += ["## Configuration", ""]
    lines += _table(["Setting", "Value"], _config_rows(report.config))
    if report.config.answer_mode is AnswerMode.EXTRACTIVE:
        lines += ["", EXTRACTIVE_NOTICE]
    lines += ["", "## Summary", ""]
    lines += _table(["Metric", "Value"], _summary_rows(report.aggregate, report.config.k_values))
    lines += ["", "## By question type", ""]
    lines += _table(
        [
            "Type",
            "n",
            f"Recall@{k}",
            f"Hit@{k}",
            "MRR",
            "Cit. valid",
            "Cit. page acc.",
            "Correctness",
            "Abstained",
        ],
        _type_rows(report.by_type, k),
    )
    lines += ["", "## Per question", ""]
    lines += _table(
        ["id", "type", f"hit@{k}", "citations valid", "correctness", "abstained", "notes"],
        _question_rows(report.questions, k),
    )
    lines += ["", "## Ingestion", ""]
    lines += _table(["File", "Outcome", "Pages", "Chunks", "ms"], _ingestion_rows(report.config))
    lines += [
        "",
        "Scores are uncalibrated proxies (see `evaluation/README.md`): retrieval metrics are "
        "page based, correctness is keyword coverage and abstention scoring is binary.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: EvaluationReport, output_dir: Path, *, stem: str = "latest"
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` into ``output_dir``; returns both paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
