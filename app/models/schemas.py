"""Public API schemas (request/response bodies) for the ContextIQ HTTP API.

Everything the FastAPI layer accepts or returns is typed here. Internal pipeline objects live
in ``app.models.domain``; the two are deliberately separate so that internal refactors do not
silently change the public contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------------


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    """Metadata for one indexed (or failed) document."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str
    workspace_id: str
    filename: str
    sha256: str
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    status: DocumentStatus
    error: str | None = Field(default=None, description="Failure reason when status is 'failed'.")
    created_at: datetime


class UploadOutcome(StrEnum):
    INDEXED = "indexed"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class DocumentUploadResult(BaseModel):
    """Per-file result of a multi-file upload."""

    filename: str
    outcome: UploadOutcome
    message: str
    document: DocumentInfo | None = None
    duplicate_of: str | None = Field(
        default=None, description="document_id of the existing identical document, if duplicate."
    )
    processing_ms: float | None = None


class DocumentUploadResponse(BaseModel):
    request_id: str
    results: list[DocumentUploadResult]

    @property
    def indexed_count(self) -> int:
        return sum(1 for r in self.results if r.outcome == UploadOutcome.INDEXED)


class DocumentListResponse(BaseModel):
    workspace_id: str
    documents: list[DocumentInfo]
    total: int


class DeleteDocumentResponse(BaseModel):
    request_id: str
    document_id: str
    deleted: bool
    chunks_removed: int = Field(ge=0)


# --------------------------------------------------------------------------------------------
# Search (debug-oriented raw retrieval)
# --------------------------------------------------------------------------------------------


class RetrievalMode(StrEnum):
    DENSE = "dense"
    HYBRID = "hybrid"


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its ranking signals. Scores are for debugging only."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    score: float = Field(description="Final ranking score. Not a calibrated probability.")
    dense_score: float | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    text: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents (must be in workspace)."
    )
    mode: RetrievalMode | None = None

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class Timings(BaseModel):
    retrieval_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0, default=0.0)
    total_ms: float = Field(ge=0)


class SearchResponse(BaseModel):
    request_id: str
    workspace_id: str
    query: str
    mode: RetrievalMode
    results: list[RetrievedChunk]
    timings: Timings


# --------------------------------------------------------------------------------------------
# Query (grounded answer)
# --------------------------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents (must be in workspace)."
    )
    mode: RetrievalMode | None = None
    include_debug: bool = Field(
        default=False, description="Include all retrieved chunks and scores in the response."
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class Citation(BaseModel):
    """A verified citation: it always maps to a chunk that was actually retrieved."""

    citation_id: str = Field(description="Short label used in the answer text, e.g. 'S1'.")
    chunk_id: str
    document_id: str
    filename: str
    page_number: int = Field(ge=1)
    excerpt: str = Field(description="Source excerpt (truncated) from the cited chunk.")
    score: float | None = Field(default=None, description="Retrieval score for debugging.")


class AnswerMode(StrEnum):
    LLM = "llm"
    EXTRACTIVE = "extractive"


class QueryResponse(BaseModel):
    request_id: str
    workspace_id: str
    question: str
    answer: str
    abstained: bool = Field(
        description="True when the system declined to answer for lack of supporting evidence."
    )
    answer_mode: AnswerMode
    citations: list[Citation]
    invalid_citation_ids: list[str] = Field(
        default_factory=list,
        description="Citation labels produced by the model that did not match any retrieved chunk "
        "and were therefore dropped.",
    )
    retrieved: list[RetrievedChunk] | None = Field(
        default=None, description="All retrieved chunks (only when include_debug=true)."
    )
    model: str | None = Field(default=None, description="LLM identifier, if an LLM was used.")
    timings: Timings


# --------------------------------------------------------------------------------------------
# Health / errors
# --------------------------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    embedding_model: str
    embedding_dimension: int | None = None
    llm_provider: str
    llm_model: str | None
    retrieval_mode: RetrievalMode
    reranker_enabled: bool
    auth_mode: str
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal problems, e.g. a workspace index that could not be loaded. "
        "Non-empty when status is 'degraded'.",
    )


class ErrorResponse(BaseModel):
    request_id: str | None = None
    error: str
    detail: str | None = None
