"""Internal domain models shared across the ingestion, retrieval and generation layers.

These are *not* the public API schemas (see ``app.models.schemas``). They describe the
objects that flow between pipeline stages: parsed pages, chunks with provenance metadata,
and scored retrieval results.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PageText(BaseModel):
    """Normalized text extracted from a single PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1, description="1-based page number as shown in a PDF viewer.")
    text: str = Field(description="Normalized page text. May be empty for image-only pages.")


class ParsedDocument(BaseModel):
    """Result of parsing one uploaded PDF."""

    model_config = ConfigDict(frozen=True)

    filename: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    pages: list[PageText]
    title: str | None = None

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def has_text(self) -> bool:
        return any(p.text.strip() for p in self.pages)


class Chunk(BaseModel):
    """A retrievable unit of text with full provenance.

    ``chunk_id`` is the stable source identifier used for citations. It is derived from the
    document ID, page number and chunk index, so it can always be mapped back to the exact
    passage that was retrieved.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    workspace_id: str
    filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0, description="0-based index of the chunk within the document.")
    text: str
    char_start: int = Field(ge=0, description="Start offset within the normalized page text.")
    char_end: int = Field(
        ge=0, description="End offset (exclusive) within the normalized page text."
    )

    @staticmethod
    def make_id(document_id: str, page_number: int, chunk_index: int) -> str:
        return f"{document_id}:p{page_number}:c{chunk_index}"


class ScoredChunk(BaseModel):
    """A chunk returned by retrieval together with its ranking signals.

    ``score`` is the final ranking score used to order results. Depending on the retrieval
    mode it is a cosine similarity (dense), a reciprocal-rank-fusion score (hybrid) or a
    cross-encoder score (rerank). None of these are calibrated probabilities.
    """

    chunk: Chunk
    vector_id: int
    score: float
    dense_score: float | None = Field(
        default=None, description="Cosine similarity from the dense index, if retrieved densely."
    )
    sparse_rank: int | None = Field(
        default=None, description="1-based rank from BM25 keyword search, if retrieved sparsely."
    )
    rerank_score: float | None = Field(
        default=None, description="Cross-encoder relevance score, if reranking was applied."
    )
