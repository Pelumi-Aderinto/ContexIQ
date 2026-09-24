"""Ingestion pipeline: uploaded PDF bytes -> parsed pages -> chunks -> embeddings -> SQLite + FAISS.

The pipeline is the single writer for a workspace's documents. It keeps the metadata store and
the vector index consistent: chunks are written to SQLite first (which assigns the FAISS ids),
then to FAISS, and a failure in the second step rolls the first one back. A document that fails
at any stage is kept as a ``failed`` row with a short, code-prefixed reason so the user can see
why, and never owns chunks.

Duplicate detection is per workspace and based on the SHA-256 of the raw upload. Re-uploading
bytes whose earlier ingestion *failed* replaces the failed row instead of reporting a duplicate,
so a transient failure can simply be retried.

Per design invariant 6 only identifiers, sizes, counts, durations and error codes are logged;
filenames and document text never are.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
import uuid
from datetime import UTC, datetime

from app.core.config import WORKSPACE_ID_PATTERN, Settings
from app.core.logging import get_logger
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import PdfParseError, is_pdf, parse_pdf
from app.models.domain import Chunk, ParsedDocument
from app.models.schemas import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentStatus,
    DocumentUploadResult,
    UploadOutcome,
)
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore
from app.storage.metadata_store import MetadataStore

log = get_logger(__name__)

MAX_ERROR_CHARS = 200
_BYTES_PER_MB = 1024 * 1024
_NO_TEXT_CODE = "no_text"
INTERRUPTED_ERROR = (
    "interrupted: indexing did not finish before the service stopped; upload the file again"
)


def _validate_workspace_id(workspace_id: str) -> str:
    """Reject ids that do not match ``WORKSPACE_ID_PATTERN`` before any row or file is touched."""
    if not isinstance(workspace_id, str) or not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
        raise ValueError("invalid workspace_id")
    return workspace_id


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _truncate(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe_failure(exc: BaseException) -> tuple[str, str]:
    """Map an exception to ``(code, reason)``: the parser's code, or the exception class name."""
    if isinstance(exc, PdfParseError):
        return exc.code, exc.message
    return type(exc).__name__, str(exc) or "no details"


def _format_error(code: str, reason: str) -> str:
    """Short ``"<code>: <reason>"`` stored on the failed document (at most 200 characters)."""
    return _truncate(f"{code}: {reason}")


def _failure_message(code: str, reason: str) -> str:
    return f"Ingestion failed ({code}): {_truncate(reason)}"


class IngestionPipeline:
    """Turn uploaded PDFs into indexed, searchable chunks for one workspace at a time."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: MetadataStore,
        vector_store: VectorStore,
        embedder: Embedder,
    ) -> None:
        self._settings = settings
        self._store = store
        self._vector_store = vector_store
        self._embedder = embedder

    # ---- public API ------------------------------------------------------------------

    def ingest(self, data: bytes, filename: str, workspace_id: str) -> DocumentUploadResult:
        """Index ``data`` into ``workspace_id`` and report the per-file outcome.

        Never raises for problems with the file itself: those come back as ``outcome=failed``
        (with the failed :class:`DocumentInfo` when a row was created). Raises ``ValueError``
        only for an invalid ``workspace_id``, which is a caller bug rather than a bad upload.
        """
        started = time.perf_counter()
        _validate_workspace_id(workspace_id)

        rejection = self._precheck(data)
        if rejection is not None:
            code, reason = rejection
            log.warning(
                "ingest.rejected",
                workspace_id=workspace_id,
                error_code=code,
                size_bytes=len(data),
            )
            return DocumentUploadResult(
                filename=filename,
                outcome=UploadOutcome.FAILED,
                message=_failure_message(code, reason),
                processing_ms=_elapsed_ms(started),
            )

        sha256 = hashlib.sha256(data).hexdigest()
        existing = self._find_duplicate(workspace_id, sha256)
        if existing is not None:
            log.info(
                "ingest.duplicate",
                workspace_id=workspace_id,
                duplicate_of=existing.document_id,
                duration_ms=_elapsed_ms(started),
            )
            return DocumentUploadResult(
                filename=filename,
                outcome=UploadOutcome.DUPLICATE,
                message=(
                    "Identical content is already in this workspace as document "
                    f"{existing.document_id}."
                ),
                document=existing,
                duplicate_of=existing.document_id,
                processing_ms=_elapsed_ms(started),
            )

        doc = self._create_document(workspace_id, filename, sha256, len(data))
        log.info(
            "ingest.started",
            document_id=doc.document_id,
            workspace_id=workspace_id,
            size_bytes=doc.size_bytes,
        )
        try:
            parsed, chunks = self._process(doc, data)
        except Exception as exc:  # any stage failure becomes a failed document, never a crash
            return self._fail(doc, exc, started)

        indexed = self._mark_indexed(doc, page_count=parsed.page_count, chunk_count=len(chunks))
        log.info(
            "ingest.completed",
            document_id=doc.document_id,
            workspace_id=workspace_id,
            page_count=indexed.page_count,
            chunk_count=indexed.chunk_count,
            duration_ms=_elapsed_ms(started),
        )
        return DocumentUploadResult(
            filename=filename,
            outcome=UploadOutcome.INDEXED,
            message=f"Indexed {indexed.chunk_count} chunks from {indexed.page_count} pages.",
            document=indexed,
            processing_ms=_elapsed_ms(started),
        )

    def reconcile_interrupted(self) -> int:
        """Repair documents left in ``processing`` by a crash or kill during ingestion.

        Ingestion is synchronous, so at startup no document should be ``processing``. Any that
        is was interrupted: its chunks may exist in SQLite without vectors (or vice versa), so
        they are removed and the row is marked ``failed`` with a retry hint. Duplicate detection
        then treats a re-upload as a retry rather than a duplicate. Returns the count repaired.
        """
        repaired = 0
        for workspace_id in self._store.list_workspaces():
            for doc in self._store.list_documents(workspace_id):
                if doc.status != DocumentStatus.PROCESSING:
                    continue
                vector_ids = self._store.delete_document(workspace_id, doc.document_id)
                with contextlib.suppress(Exception):
                    self._vector_store.remove(workspace_id, vector_ids)
                self._store.create_document(
                    doc.model_copy(
                        update={
                            "status": DocumentStatus.FAILED,
                            "error": INTERRUPTED_ERROR,
                            "chunk_count": 0,
                        }
                    )
                )
                log.warning(
                    "ingest.reconciled_interrupted",
                    document_id=doc.document_id,
                    workspace_id=workspace_id,
                    chunks_removed=len(vector_ids),
                )
                repaired += 1
        return repaired

    def delete_document(self, workspace_id: str, document_id: str) -> DeleteDocumentResponse | None:
        """Remove a document, its chunks and its vectors. ``None`` if it is not in the workspace.

        ``request_id`` is left empty for the API route to fill in.
        """
        _validate_workspace_id(workspace_id)
        if self._store.get_document(workspace_id, document_id) is None:
            return None
        vector_ids = self._store.delete_document(workspace_id, document_id)
        removed = self._vector_store.remove(workspace_id, vector_ids) if vector_ids else 0
        if removed != len(vector_ids):
            log.warning(
                "document.delete_index_mismatch",
                document_id=document_id,
                workspace_id=workspace_id,
                expected=len(vector_ids),
                removed=removed,
            )
        log.info(
            "document.deleted",
            document_id=document_id,
            workspace_id=workspace_id,
            chunks_removed=len(vector_ids),
        )
        return DeleteDocumentResponse(
            request_id="",
            document_id=document_id,
            deleted=True,
            chunks_removed=len(vector_ids),
        )

    # ---- stages ----------------------------------------------------------------------

    def _precheck(self, data: bytes) -> tuple[str, str] | None:
        """Cheap rejections that happen before anything is stored: size limit and PDF magic."""
        limit = self._settings.max_upload_bytes
        if len(data) > limit:
            size_mb = len(data) / _BYTES_PER_MB
            return (
                "too_large",
                f"The file is {size_mb:.1f} MB; the maximum upload size is "
                f"{self._settings.max_upload_mb} MB.",
            )
        if not is_pdf(data):
            return "not_pdf", "The file is not a PDF (missing %PDF- header)."
        return None

    def _find_duplicate(self, workspace_id: str, sha256: str) -> DocumentInfo | None:
        """Existing document with the same content, or ``None``.

        A previous *failed* attempt is not a duplicate: its row is removed so the new upload
        acts as a retry and the workspace never accumulates stale failures for one file.
        """
        existing = self._store.find_by_sha256(workspace_id, sha256)
        if existing is None:
            return None
        if existing.status is DocumentStatus.FAILED:
            self._store.delete_document(workspace_id, existing.document_id)
            log.info(
                "ingest.retrying_failed_document",
                workspace_id=workspace_id,
                document_id=existing.document_id,
            )
            return None
        return existing

    def _create_document(
        self, workspace_id: str, filename: str, sha256: str, size_bytes: int
    ) -> DocumentInfo:
        doc = DocumentInfo(
            document_id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
            page_count=0,
            chunk_count=0,
            status=DocumentStatus.PROCESSING,
            created_at=datetime.now(UTC),
        )
        self._store.create_document(doc)
        return doc

    def _process(self, doc: DocumentInfo, data: bytes) -> tuple[ParsedDocument, list[Chunk]]:
        """Parse, chunk, embed and persist. Raises on any failure (the caller marks the doc)."""
        parsed = parse_pdf(data, doc.filename, max_pages=self._settings.max_pages)
        chunks = chunk_document(
            parsed,
            document_id=doc.document_id,
            workspace_id=doc.workspace_id,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise PdfParseError("The PDF produced no indexable text.", code=_NO_TEXT_CODE)
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._store_and_index(doc, chunks, vectors)
        return parsed, chunks

    def _store_and_index(self, doc: DocumentInfo, chunks: list[Chunk], vectors: object) -> None:
        """Write chunks to SQLite, then their vectors to FAISS; undo SQLite if FAISS fails."""
        vector_ids = self._store.add_chunks(chunks)
        try:
            self._vector_store.add(doc.workspace_id, vector_ids, vectors)  # type: ignore[arg-type]
        except Exception:
            try:
                self._rollback_chunks(doc, vector_ids)
            except Exception as rollback_exc:
                log.error(
                    "ingest.rollback_failed",
                    document_id=doc.document_id,
                    workspace_id=doc.workspace_id,
                    error_type=type(rollback_exc).__name__,
                )
            raise

    def _rollback_chunks(self, doc: DocumentInfo, vector_ids: list[int]) -> None:
        """Remove a half-written document's chunks so SQLite and FAISS never diverge.

        Any vectors that did make it into the index are removed too, then the chunks and the
        document row are deleted and the row is re-created as it was before chunking so the
        caller can mark it failed.
        """
        with contextlib.suppress(Exception):
            self._vector_store.remove(doc.workspace_id, vector_ids)
        self._store.delete_document(doc.workspace_id, doc.document_id)
        self._store.create_document(doc)
        log.warning(
            "ingest.rolled_back",
            document_id=doc.document_id,
            workspace_id=doc.workspace_id,
            chunk_count=len(vector_ids),
        )

    def _mark_indexed(
        self, doc: DocumentInfo, *, page_count: int, chunk_count: int
    ) -> DocumentInfo:
        self._store.update_document(
            doc.workspace_id,
            doc.document_id,
            status=DocumentStatus.INDEXED,
            chunk_count=chunk_count,
            page_count=page_count,
        )
        stored = self._store.get_document(doc.workspace_id, doc.document_id)
        return stored or doc.model_copy(
            update={
                "status": DocumentStatus.INDEXED,
                "chunk_count": chunk_count,
                "page_count": page_count,
            }
        )

    def _fail(self, doc: DocumentInfo, exc: BaseException, started: float) -> DocumentUploadResult:
        """Record the failure on the document row and build the ``failed`` upload result."""
        code, reason = _describe_failure(exc)
        error = _format_error(code, reason)
        self._store.update_document(
            doc.workspace_id, doc.document_id, status=DocumentStatus.FAILED, error=error
        )
        failed = self._store.get_document(doc.workspace_id, doc.document_id) or doc.model_copy(
            update={"status": DocumentStatus.FAILED, "error": error}
        )
        log.warning(
            "ingest.failed",
            document_id=doc.document_id,
            workspace_id=doc.workspace_id,
            error_code=code,
            duration_ms=_elapsed_ms(started),
            exc_info=not isinstance(exc, PdfParseError),
        )
        return DocumentUploadResult(
            filename=doc.filename,
            outcome=UploadOutcome.FAILED,
            message=_failure_message(code, reason),
            document=failed,
            processing_ms=_elapsed_ms(started),
        )
