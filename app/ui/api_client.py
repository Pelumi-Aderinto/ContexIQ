"""Thin HTTP client for the ContextIQ API.

The Streamlit UI talks to the backend exclusively through this module; it never imports the
ingestion, retrieval or generation pipeline. Responses are parsed into the public schemas from
``app.models.schemas`` so the UI works with typed objects. The API key is sent as the
``X-API-Key`` header and is never logged or included in ``repr``.
"""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.models.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

API_KEY_HEADER = "X-API-Key"
CONNECT_TIMEOUT_SECONDS = 5.0
MAX_ERROR_TEXT_CHARS = 300


class ApiError(Exception):
    """Raised when the API returns a non-2xx response or cannot be reached.

    ``status_code`` is ``None`` for transport-level failures (connection refused, timeout) and
    for client-side request validation failures.
    """

    def __init__(
        self, status_code: int | None, message: str, request_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.request_id = request_id

    def __str__(self) -> str:
        prefix = f"HTTP {self.status_code}: " if self.status_code is not None else ""
        suffix = f" (request id {self.request_id})" if self.request_id else ""
        return f"{prefix}{self.message}{suffix}"


class ContextIQClient:
    """Synchronous client for the ContextIQ HTTP API built on ``httpx.Client``."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {API_KEY_HEADER: api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"ContextIQClient(base_url={self.base_url!r}, api_key=***)"

    def __enter__(self) -> ContextIQClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    # ---- endpoints ---------------------------------------------------------------------

    def health(self) -> HealthResponse:
        """``GET /health`` (no auth required)."""
        return self._request("GET", "/health", HealthResponse)

    def list_documents(self) -> DocumentListResponse:
        """``GET /documents`` for the caller's workspace."""
        return self._request("GET", "/documents", DocumentListResponse)

    def upload(self, files: list[tuple[str, bytes]]) -> DocumentUploadResponse:
        """``POST /documents`` with one multipart ``files`` part per ``(filename, bytes)``."""
        if not files:
            raise ValueError("files must not be empty")
        parts = [("files", (name, data, "application/pdf")) for name, data in files]
        return self._request("POST", "/documents", DocumentUploadResponse, files=parts)

    def delete_document(self, document_id: str) -> DeleteDocumentResponse:
        """``DELETE /documents/{document_id}``."""
        return self._request("DELETE", f"/documents/{document_id}", DeleteDocumentResponse)

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        mode: str | None = None,
        include_debug: bool = False,
    ) -> QueryResponse:
        """``POST /query``: grounded answer with verified citations."""
        body = _validated_body(
            QueryRequest,
            question=question,
            top_k=top_k,
            document_ids=document_ids or None,
            mode=mode,
            include_debug=include_debug,
        )
        return self._request("POST", "/query", QueryResponse, json=body)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        mode: str | None = None,
    ) -> SearchResponse:
        """``POST /search``: raw retrieval results for debugging."""
        body = _validated_body(
            SearchRequest, query=query, top_k=top_k, document_ids=document_ids or None, mode=mode
        )
        return self._request("POST", "/search", SearchResponse, json=body)

    # ---- internals ---------------------------------------------------------------------

    def _request(self, method: str, path: str, model: type[ModelT], **kwargs: Any) -> ModelT:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            message = (
                f"Could not reach the ContextIQ API at {self.base_url} ({type(exc).__name__})."
            )
            raise ApiError(None, message) from exc
        if response.is_error:
            raise _error_from_response(response)
        return _parse_body(response, model)


def _validated_body(model: type[BaseModel], **fields: Any) -> dict[str, Any]:
    """Validate request fields client-side and serialize them for the JSON body."""
    try:
        request = model(**fields)
    except ValidationError as exc:
        raise ApiError(None, _first_validation_message(exc)) from exc
    return request.model_dump(mode="json", exclude_none=True)


def _first_validation_message(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ())) or "request"
    message = str(error.get("msg", "invalid value")).removeprefix("Value error, ")
    return f"{location}: {message}"


def _error_from_response(response: httpx.Response) -> ApiError:
    """Build an ``ApiError`` from an ``ErrorResponse`` body, a FastAPI ``detail`` or raw text."""
    request_id = response.headers.get("X-Request-ID")
    payload = _safe_json(response)
    if isinstance(payload, dict):
        try:
            error = ErrorResponse.model_validate(payload)
        except ValidationError:
            detail = payload.get("detail")
            if detail:
                return ApiError(response.status_code, _stringify_detail(detail), request_id)
        else:
            message = f"{error.error}: {error.detail}" if error.detail else error.error
            return ApiError(response.status_code, message, error.request_id or request_id)
    text = response.text.strip() or response.reason_phrase or "Unknown error"
    return ApiError(response.status_code, text[:MAX_ERROR_TEXT_CHARS], request_id)


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _stringify_detail(detail: Any) -> str:
    """Flatten FastAPI's ``detail`` (a string or a list of validation errors)."""
    if isinstance(detail, list):
        parts = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail
        ]
        return "; ".join(parts)
    return str(detail)


def _parse_body(response: httpx.Response, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(response.content)
    except ValidationError as exc:
        message = (
            f"Unexpected response body from {response.url.path}: "
            f"{exc.error_count()} validation error(s)."
        )
        raise ApiError(response.status_code, message) from exc
