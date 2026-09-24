"""Start-up reconciliation of documents whose ingestion was interrupted by a crash."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.ingestion.pipeline import INTERRUPTED_ERROR
from app.models.domain import Chunk
from app.models.schemas import DocumentInfo, DocumentStatus, RetrievalMode, UploadOutcome

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import RagStack

WS = "alpha"


def _processing_doc(*, sha256: str = "ab" * 32, document_id: str = "deadbeef" * 4) -> DocumentInfo:
    return DocumentInfo(
        document_id=document_id,
        workspace_id=WS,
        filename="crash.pdf",
        sha256=sha256,
        size_bytes=10,
        page_count=0,
        chunk_count=0,
        status=DocumentStatus.PROCESSING,
        created_at=datetime.now(UTC),
    )


def test_interrupted_document_is_failed_and_its_orphan_chunks_removed(
    make_rag_stack: Callable[..., RagStack], make_pdf: Callable[..., bytes]
) -> None:
    stack = make_rag_stack()
    healthy = stack.pipeline.ingest(
        make_pdf(["A healthy document about rooftop solar panels and inverters."]), "ok.pdf", WS
    )
    assert healthy.outcome == UploadOutcome.INDEXED and healthy.document is not None
    doc = _processing_doc()
    stack.store.create_document(doc)
    orphan = Chunk(
        chunk_id=Chunk.make_id(doc.document_id, 1, 0),
        document_id=doc.document_id,
        workspace_id=WS,
        filename=doc.filename,
        page_number=1,
        chunk_index=0,
        text="orphaned zebra passage written just before the crash",
        char_start=0,
        char_end=52,
    )
    stack.store.add_chunks([orphan])  # simulated crash: chunks in SQLite, no vectors in FAISS
    chunks_before = stack.store.count_chunks(WS)

    restarted = make_rag_stack()
    repaired = restarted.pipeline.reconcile_interrupted()

    assert repaired == 1
    stored = restarted.store.get_document(WS, doc.document_id)
    assert stored is not None
    assert stored.status == DocumentStatus.FAILED
    assert stored.error == INTERRUPTED_ERROR
    assert stored.chunk_count == 0
    assert restarted.store.count_chunks(WS) == chunks_before - 1
    assert restarted.store.get_chunk_by_id(WS, orphan.chunk_id) is None
    hits = restarted.retrieval.retrieve(WS, "orphaned zebra passage", mode=RetrievalMode.HYBRID)
    assert all(sc.chunk.document_id != doc.document_id for sc in hits)
    untouched = restarted.store.get_document(WS, healthy.document.document_id)
    assert untouched is not None and untouched.status == DocumentStatus.INDEXED


def test_reconcile_is_a_no_op_when_nothing_was_interrupted(
    make_rag_stack: Callable[..., RagStack], make_pdf: Callable[..., bytes]
) -> None:
    stack = make_rag_stack()
    stack.pipeline.ingest(make_pdf(["Some perfectly ordinary text about invoices."]), "a.pdf", WS)
    before = stack.store.list_documents(WS)

    assert stack.pipeline.reconcile_interrupted() == 0
    assert stack.store.list_documents(WS) == before


def test_reupload_after_reconcile_is_a_retry_not_a_duplicate(
    make_rag_stack: Callable[..., RagStack], make_pdf: Callable[..., bytes]
) -> None:
    data = make_pdf(["Retry me please, this page has enough text to be chunked and indexed."])
    stack = make_rag_stack()
    stack.store.create_document(_processing_doc(sha256=hashlib.sha256(data).hexdigest()))

    restarted = make_rag_stack()
    assert restarted.pipeline.reconcile_interrupted() == 1
    result = restarted.pipeline.ingest(data, "crash.pdf", WS)

    assert result.outcome == UploadOutcome.INDEXED
    assert result.document is not None and result.document.status == DocumentStatus.INDEXED
