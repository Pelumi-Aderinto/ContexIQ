"""Document endpoints: multi-file upload with per-file outcomes, list, inspect and delete.

Every handler is scoped to the caller's workspace through ``PrincipalDep``; a document in
another workspace is indistinguishable from a missing one (404). Uploads are read in 1 MiB
pieces and abandoned as soon as they pass ``max_upload_bytes`` so an oversized file never fills
memory. The multipart content type is ignored on purpose: the ingestion pipeline checks the PDF
magic bytes, which cannot be spoofed by a header.
"""

from __future__ import annotations

import time
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Path, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import PrincipalDep, RequestIdDep, Services, ServicesDep, error_responses
from app.core.logging import get_logger
from app.models.schemas import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentUploadResult,
    UploadOutcome,
)

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"], responses=error_responses(401))

READ_CHUNK_BYTES = 1024 * 1024
DEFAULT_FILENAME = "upload.pdf"
MAX_FILENAME_CHARS = 255
NOT_FOUND_DETAIL = "Document not found"

DocumentId = Annotated[
    str, Path(min_length=1, max_length=128, description="Document id as returned by the upload.")
]


# ---- helpers -------------------------------------------------------------------------------


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _safe_filename(raw: str | None) -> str:
    """Base name of the client-supplied filename, bounded in length, or a default."""
    name = PurePosixPath(raw.replace("\\", "/")).name.strip() if raw else ""
    if not name or name in {".", ".."}:
        name = DEFAULT_FILENAME
    return name[:MAX_FILENAME_CHARS]


async def _read_limited(upload: UploadFile, limit: int) -> bytes | None:
    """Read ``upload`` in pieces; ``None`` as soon as it is known to exceed ``limit`` bytes."""
    buffer = bytearray()
    while len(buffer) <= limit:
        piece = await upload.read(min(READ_CHUNK_BYTES, limit + 1 - len(buffer)))
        if not piece:
            return bytes(buffer)
        buffer.extend(piece)
    return None


async def _ingest_upload(
    upload: UploadFile, workspace_id: str, services: Services
) -> DocumentUploadResult:
    """Read one uploaded file and run it through the pipeline; never raises for a bad file."""
    started = time.perf_counter()
    settings = services.settings
    filename = _safe_filename(upload.filename)
    data = await _read_limited(upload, settings.max_upload_bytes)
    if data is None:
        log.warning(
            "documents.upload_too_large",
            workspace_id=workspace_id,
            limit_bytes=settings.max_upload_bytes,
        )
        return DocumentUploadResult(
            filename=filename,
            outcome=UploadOutcome.FAILED,
            message=f"File exceeds {settings.max_upload_mb} MB limit",
            processing_ms=_elapsed_ms(started),
        )
    return await run_in_threadpool(services.ingestion.ingest, data, filename, workspace_id)


def _upload_status_code(results: list[DocumentUploadResult]) -> int:
    """201 if anything was indexed, 200 if only duplicates, 422 if every file failed."""
    outcomes = {result.outcome for result in results}
    if UploadOutcome.INDEXED in outcomes:
        return status.HTTP_201_CREATED
    if UploadOutcome.DUPLICATE in outcomes:
        return status.HTTP_200_OK
    return status.HTTP_422_UNPROCESSABLE_CONTENT


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_FOUND_DETAIL)


# ---- routes --------------------------------------------------------------------------------


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index PDF files",
    responses={
        status.HTTP_200_OK: {
            "model": DocumentUploadResponse,
            "description": "Nothing new was indexed: every file was a duplicate.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Every file failed (DocumentUploadResponse with per-file reasons), "
            "or the request itself was invalid (ErrorResponse).",
        },
    },
)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="One or more PDF files.")],
    response: Response,
    principal: PrincipalDep,
    services: ServicesDep,
    request_id: RequestIdDep,
) -> DocumentUploadResponse:
    """Index each file into the caller's workspace and report a per-file outcome."""
    limit = services.settings.max_files_per_upload
    if len(files) > limit:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Too many files: {len(files)} uploaded, at most {limit} allowed per request",
        )
    results = [await _ingest_upload(upload, principal.workspace_id, services) for upload in files]
    response.status_code = _upload_status_code(results)
    log.info(
        "documents.upload_completed",
        workspace_id=principal.workspace_id,
        file_count=len(results),
        indexed=sum(r.outcome is UploadOutcome.INDEXED for r in results),
        duplicates=sum(r.outcome is UploadOutcome.DUPLICATE for r in results),
        failed=sum(r.outcome is UploadOutcome.FAILED for r in results),
        status_code=response.status_code,
    )
    return DocumentUploadResponse(request_id=request_id, results=results)


@router.get("", response_model=DocumentListResponse, summary="List the workspace's documents")
def list_documents(principal: PrincipalDep, services: ServicesDep) -> DocumentListResponse:
    """Every document in the caller's workspace, newest first, including failed ones."""
    documents = services.store.list_documents(principal.workspace_id)
    return DocumentListResponse(
        workspace_id=principal.workspace_id, documents=documents, total=len(documents)
    )


@router.get(
    "/{document_id}",
    response_model=DocumentInfo,
    summary="Inspect one document",
    responses=error_responses(404),
)
def get_document(
    document_id: DocumentId, principal: PrincipalDep, services: ServicesDep
) -> DocumentInfo:
    """Metadata for one document; 404 unless it belongs to the caller's workspace."""
    document = services.store.get_document(principal.workspace_id, document_id)
    if document is None:
        raise _not_found()
    return document


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document and its chunks",
    responses=error_responses(404),
)
def delete_document(
    document_id: DocumentId,
    principal: PrincipalDep,
    services: ServicesDep,
    request_id: RequestIdDep,
) -> DeleteDocumentResponse:
    """Remove the document, its chunks and its vectors; 404 unless it is in the workspace."""
    result = services.ingestion.delete_document(principal.workspace_id, document_id)
    if result is None:
        raise _not_found()
    return result.model_copy(update={"request_id": request_id})
