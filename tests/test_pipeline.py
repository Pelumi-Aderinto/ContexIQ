"""Integration tests for the ingestion pipeline: parse -> chunk -> embed -> SQLite + FAISS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import structlog

from app.ingestion.pipeline import IngestionPipeline
from app.models.domain import Chunk
from app.models.schemas import DocumentStatus, RetrievalMode, UploadOutcome

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import MakePdf, RagStack

WS = "alpha"
OTHER_WS = "beta"
CORRUPT_PDF = b"%PDF-1.7 garbage"

PAGES = [
    "Page one is about the zephyrite mineral. Zephyrite forms in ancient riverbeds and glows "
    "faintly blue under ultraviolet light, which is how collectors identify zephyrite.",
    "Page two describes the AX2-7731 pressure valve. The AX2-7731 valve tolerates 400 bar and "
    "must be inspected every twelve months by a certified technician.",
    "Page three covers the quarterly logistics report. Shipments rose in the second quarter "
    "thanks to the new warehouse and a shorter customs clearance process.",
]


class ExplodingEmbedder:
    """Embedder stand-in whose ``embed_documents`` always fails."""

    model_name = "exploding"
    dimension = 64

    def __init__(self, message: str = "model unavailable") -> None:
        self.message = message

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise ValueError(self.message)

    def embed_query(self, text: str) -> np.ndarray:
        raise ValueError(self.message)


@pytest.fixture
def pdf_bytes(make_pdf: MakePdf) -> bytes:
    return make_pdf(PAGES, title="Three page test document")


def _ingest(stack: RagStack, data: bytes, filename: str = "three_pages.pdf", ws: str = WS):
    return stack.pipeline.ingest(data, filename, ws)


def _boom(*args: object, **kwargs: object) -> None:
    raise RuntimeError("disk full")


# ---- happy path ------------------------------------------------------------------------------


def test_ingest_indexes_a_three_page_pdf(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    result = _ingest(rag_stack, pdf_bytes)

    assert result.outcome is UploadOutcome.INDEXED
    assert result.filename == "three_pages.pdf"
    assert result.duplicate_of is None
    assert result.processing_ms is not None and result.processing_ms >= 0
    doc = result.document
    assert doc is not None
    assert doc.status is DocumentStatus.INDEXED
    assert doc.error is None
    assert doc.page_count == 3
    assert doc.chunk_count > 0
    assert doc.workspace_id == WS
    assert doc.filename == "three_pages.pdf"
    assert doc.size_bytes == len(pdf_bytes)
    assert len(doc.sha256) == 64
    assert f"{doc.chunk_count} chunks" in result.message and "3 pages" in result.message

    assert rag_stack.store.count_chunks(WS) == doc.chunk_count
    assert rag_stack.vector_store.count(WS) == doc.chunk_count
    assert rag_stack.store.get_document(WS, doc.document_id) == doc
    assert rag_stack.store.list_documents(WS) == [doc]
    assert rag_stack.vector_store.workspaces() == [WS]


def test_chunk_metadata_matches_the_document(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    doc = _ingest(rag_stack, pdf_bytes).document
    assert doc is not None

    rows = list(rag_stack.store.iter_chunks(WS))
    chunks = [chunk for _, chunk in rows]
    assert len(chunks) == doc.chunk_count
    assert {c.page_number for c in chunks} == {1, 2, 3}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.document_id == doc.document_id
        assert chunk.workspace_id == WS
        assert chunk.filename == "three_pages.pdf"
        assert chunk.chunk_id == Chunk.make_id(
            doc.document_id, chunk.page_number, chunk.chunk_index
        )
        assert 0 <= chunk.char_start < chunk.char_end
    assert any("AX2-7731" in c.text for c in chunks if c.page_number == 2)
    assert not any("AX2-7731" in c.text for c in chunks if c.page_number != 2)

    # Every SQLite vector_id is in FAISS and vice versa.
    probe = rag_stack.embedder.embed_query("probe")
    faiss_ids = {vid for vid, _ in rag_stack.vector_store.search(WS, probe, k=1000)}
    assert faiss_ids == {vid for vid, _ in rows} == set(rag_stack.store.list_vector_ids(WS))


def test_ingested_content_is_retrievable_by_page(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    _ingest(rag_stack, pdf_bytes)
    results = rag_stack.retrieval.retrieve(WS, "AX2-7731", mode=RetrievalMode.HYBRID)
    assert results and results[0].chunk.page_number == 2
    assert "AX2-7731" in results[0].chunk.text


def test_logs_carry_ids_and_counts_but_never_names_or_text(
    rag_stack: RagStack, pdf_bytes: bytes
) -> None:
    with structlog.testing.capture_logs() as logs:
        result = _ingest(rag_stack, pdf_bytes, filename="confidential-merger-plan.pdf")

    assert result.document is not None
    events = [entry["event"] for entry in logs]
    assert "ingest.started" in events and "ingest.completed" in events
    completed = next(entry for entry in logs if entry["event"] == "ingest.completed")
    assert completed["document_id"] == result.document.document_id
    assert completed["workspace_id"] == WS
    assert completed["page_count"] == 3
    assert completed["chunk_count"] == result.document.chunk_count
    assert completed["duration_ms"] >= 0
    rendered = repr(logs).lower()
    assert "confidential-merger-plan" not in rendered
    assert "zephyrite" not in rendered


# ---- duplicates ------------------------------------------------------------------------------


def test_duplicate_upload_is_reported_not_reindexed(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    first = _ingest(rag_stack, pdf_bytes)
    assert first.document is not None

    second = _ingest(rag_stack, pdf_bytes, filename="renamed-copy.pdf")

    assert second.outcome is UploadOutcome.DUPLICATE
    assert second.filename == "renamed-copy.pdf"
    assert second.duplicate_of == first.document.document_id
    assert second.document == first.document
    assert first.document.document_id in second.message
    assert second.processing_ms is not None
    assert len(rag_stack.store.list_documents(WS)) == 1
    assert rag_stack.store.count_chunks(WS) == first.document.chunk_count
    assert rag_stack.vector_store.count(WS) == first.document.chunk_count


def test_same_bytes_in_another_workspace_are_indexed(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    alpha = _ingest(rag_stack, pdf_bytes)
    beta = _ingest(rag_stack, pdf_bytes, ws=OTHER_WS)

    assert alpha.outcome is UploadOutcome.INDEXED and beta.outcome is UploadOutcome.INDEXED
    assert alpha.document is not None and beta.document is not None
    assert alpha.document.document_id != beta.document.document_id
    assert alpha.document.sha256 == beta.document.sha256
    assert beta.document.workspace_id == OTHER_WS
    assert rag_stack.vector_store.workspaces() == [WS, OTHER_WS]
    assert rag_stack.store.count_chunks(OTHER_WS) == beta.document.chunk_count
    assert rag_stack.vector_store.count(OTHER_WS) == beta.document.chunk_count


# ---- rejected and failed uploads -------------------------------------------------------------


def test_non_pdf_is_rejected_without_storing_anything(rag_stack: RagStack) -> None:
    result = _ingest(rag_stack, b"hello, this is plain text", filename="notes.txt")

    assert result.outcome is UploadOutcome.FAILED
    assert "not_pdf" in result.message
    assert result.document is None
    assert result.duplicate_of is None
    assert rag_stack.store.list_documents(WS) == []
    assert rag_stack.store.count_chunks(WS) == 0
    assert rag_stack.vector_store.workspaces() == []


def test_oversize_upload_is_rejected_without_storing_anything(
    make_rag_stack: Callable[..., RagStack], make_pdf: MakePdf
) -> None:
    stack = make_rag_stack(max_upload_mb=1)
    data = make_pdf(["small page"]) + b"\n%" + b"0" * (1024 * 1024)
    assert len(data) > stack.settings.max_upload_bytes

    result = _ingest(stack, data, filename="huge.pdf")

    assert result.outcome is UploadOutcome.FAILED
    assert "too_large" in result.message and "1 MB" in result.message
    assert result.document is None
    assert stack.store.list_documents(WS) == []
    assert stack.store.count_chunks(WS) == 0
    assert stack.vector_store.workspaces() == []


def test_corrupt_pdf_is_recorded_as_failed(rag_stack: RagStack) -> None:
    result = _ingest(rag_stack, CORRUPT_PDF, filename="broken.pdf")

    assert result.outcome is UploadOutcome.FAILED
    assert "corrupt" in result.message
    doc = result.document
    assert doc is not None
    assert doc.status is DocumentStatus.FAILED
    assert doc.error is not None and doc.error.startswith("corrupt:")
    assert doc.chunk_count == 0
    assert rag_stack.store.list_documents(WS) == [doc]
    assert rag_stack.store.count_chunks(WS) == 0
    assert rag_stack.vector_store.count(WS) == 0


def test_pdf_without_text_fails_with_no_text(rag_stack: RagStack, make_pdf: MakePdf) -> None:
    result = _ingest(rag_stack, make_pdf(["", ""]), filename="scanned.pdf")

    assert result.outcome is UploadOutcome.FAILED
    assert result.document is not None
    assert result.document.status is DocumentStatus.FAILED
    assert result.document.error is not None and result.document.error.startswith("no_text:")
    assert result.document.chunk_count == 0
    assert rag_stack.store.count_chunks(WS) == 0


def test_encrypted_pdf_fails_with_encrypted_code(rag_stack: RagStack, make_pdf: MakePdf) -> None:
    import pymupdf

    doc = pymupdf.open(stream=make_pdf(["secret text on one page"]), filetype="pdf")
    try:
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    finally:
        doc.close()

    result = _ingest(rag_stack, data, filename="locked.pdf")

    assert result.outcome is UploadOutcome.FAILED
    assert result.document is not None
    assert result.document.error is not None and result.document.error.startswith("encrypted:")
    assert rag_stack.store.count_chunks(WS) == 0


def test_too_many_pages_fails(make_rag_stack: Callable[..., RagStack], pdf_bytes: bytes) -> None:
    stack = make_rag_stack(max_pages=2, chunk_size=200, chunk_overlap=20)
    result = _ingest(stack, pdf_bytes)

    assert result.outcome is UploadOutcome.FAILED
    assert result.document is not None
    assert result.document.error is not None
    assert result.document.error.startswith("too_many_pages:")
    assert stack.store.count_chunks(WS) == 0


def test_vector_store_failure_rolls_back_sqlite_chunks(
    rag_stack: RagStack, pdf_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_stack.vector_store, "add", _boom)

    result = _ingest(rag_stack, pdf_bytes)

    assert result.outcome is UploadOutcome.FAILED
    assert "RuntimeError" in result.message and "disk full" in result.message
    doc = result.document
    assert doc is not None
    assert doc.status is DocumentStatus.FAILED
    assert doc.error == "RuntimeError: disk full"
    assert doc.chunk_count == 0
    # SQLite and FAISS agree: no chunks anywhere, but the failed row is kept for the user.
    assert rag_stack.store.count_chunks(WS) == 0
    assert rag_stack.store.list_vector_ids(WS) == []
    assert rag_stack.vector_store.count(WS) == 0
    assert rag_stack.store.list_documents(WS) == [doc]


def test_embedder_failure_marks_the_document_failed(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    pipeline = IngestionPipeline(
        settings=rag_stack.settings,
        store=rag_stack.store,
        vector_store=rag_stack.vector_store,
        embedder=ExplodingEmbedder(),
    )

    result = pipeline.ingest(pdf_bytes, "three_pages.pdf", WS)

    assert result.outcome is UploadOutcome.FAILED
    assert result.document is not None
    assert result.document.status is DocumentStatus.FAILED
    assert result.document.error == "ValueError: model unavailable"
    assert rag_stack.store.count_chunks(WS) == 0
    assert rag_stack.vector_store.count(WS) == 0


def test_failure_reason_is_truncated(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    pipeline = IngestionPipeline(
        settings=rag_stack.settings,
        store=rag_stack.store,
        vector_store=rag_stack.vector_store,
        embedder=ExplodingEmbedder("x" * 1000),
    )
    result = pipeline.ingest(pdf_bytes, "three_pages.pdf", WS)

    assert result.document is not None and result.document.error is not None
    assert len(result.document.error) == 200
    assert result.document.error.startswith("ValueError: xxx")
    assert result.document.error.endswith("...")


def test_failed_document_can_be_retried(
    rag_stack: RagStack, pdf_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patched:
        patched.setattr(rag_stack.vector_store, "add", _boom)
        failed = _ingest(rag_stack, pdf_bytes)
    assert failed.outcome is UploadOutcome.FAILED and failed.document is not None

    retried = _ingest(rag_stack, pdf_bytes)

    assert retried.outcome is UploadOutcome.INDEXED
    assert retried.document is not None
    assert retried.duplicate_of is None
    remaining = rag_stack.store.list_documents(WS)
    assert [d.document_id for d in remaining] == [retried.document.document_id]
    assert rag_stack.store.get_document(WS, failed.document.document_id) is None
    assert rag_stack.store.count_chunks(WS) == retried.document.chunk_count
    assert rag_stack.vector_store.count(WS) == retried.document.chunk_count


def test_invalid_workspace_id_raises(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        _ingest(rag_stack, pdf_bytes, ws="../escape")
    with pytest.raises(ValueError, match="workspace_id"):
        rag_stack.pipeline.delete_document("Bad Workspace", "doc")
    assert rag_stack.store.list_workspaces() == []


# ---- deletion --------------------------------------------------------------------------------


def test_delete_removes_chunks_from_sqlite_and_faiss(rag_stack: RagStack, pdf_bytes: bytes) -> None:
    doc = _ingest(rag_stack, pdf_bytes).document
    assert doc is not None
    assert rag_stack.retrieval.retrieve(WS, "zephyrite mineral")

    response = rag_stack.pipeline.delete_document(WS, doc.document_id)

    assert response is not None
    assert response.deleted is True
    assert response.request_id == ""
    assert response.document_id == doc.document_id
    assert response.chunks_removed == doc.chunk_count > 0
    assert rag_stack.store.get_document(WS, doc.document_id) is None
    assert rag_stack.store.count_chunks(WS) == 0
    assert rag_stack.vector_store.count(WS) == 0
    assert rag_stack.retrieval.retrieve(WS, "zephyrite mineral") == []
    assert rag_stack.retrieval.retrieve(WS, "AX2-7731", mode=RetrievalMode.HYBRID) == []


def test_delete_keeps_other_documents_intact(
    rag_stack: RagStack, pdf_bytes: bytes, make_pdf: MakePdf
) -> None:
    first = _ingest(rag_stack, pdf_bytes).document
    second = _ingest(
        rag_stack, make_pdf(["A separate document about sourdough starters and baking."]), "b.pdf"
    ).document
    assert first is not None and second is not None

    response = rag_stack.pipeline.delete_document(WS, first.document_id)

    assert response is not None and response.chunks_removed == first.chunk_count
    assert rag_stack.store.list_documents(WS) == [second]
    assert rag_stack.store.count_chunks(WS) == second.chunk_count
    assert rag_stack.vector_store.count(WS) == second.chunk_count
    results = rag_stack.retrieval.retrieve(WS, "sourdough starters", mode=RetrievalMode.HYBRID)
    assert results and {sc.chunk.document_id for sc in results} == {second.document_id}


def test_delete_of_unknown_document_returns_none(rag_stack: RagStack) -> None:
    assert rag_stack.pipeline.delete_document(WS, "does-not-exist") is None


def test_delete_from_the_wrong_workspace_returns_none_and_keeps_data(
    rag_stack: RagStack, pdf_bytes: bytes
) -> None:
    doc = _ingest(rag_stack, pdf_bytes).document
    assert doc is not None

    assert rag_stack.pipeline.delete_document(OTHER_WS, doc.document_id) is None

    assert rag_stack.store.get_document(WS, doc.document_id) == doc
    assert rag_stack.store.count_chunks(WS) == doc.chunk_count
    assert rag_stack.vector_store.count(WS) == doc.chunk_count


def test_delete_of_failed_document_removes_its_row(rag_stack: RagStack) -> None:
    failed = _ingest(rag_stack, CORRUPT_PDF, filename="broken.pdf").document
    assert failed is not None

    response = rag_stack.pipeline.delete_document(WS, failed.document_id)

    assert response is not None
    assert response.chunks_removed == 0
    assert rag_stack.store.list_documents(WS) == []
