"""FastAPI dependencies shared by the route modules.

``Services`` is the container the application lifespan stores on ``app.state.services``;
routes obtain it through :func:`get_services` instead of touching global state, so tests can
build an app around fakes. The ``*Dep`` aliases keep handler signatures short and uniform.

``Services`` lives here rather than in ``app.api.main`` because ``main`` imports the routes
and the routes need this type; ``main`` re-exports it under the contract name.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings
from app.core.logging import new_request_id
from app.core.security import Principal, get_principal
from app.generation.chain import AnswerService
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import ErrorResponse
from app.retrieval.embeddings import Embedder
from app.retrieval.retriever import RetrievalService
from app.retrieval.vector_store import VectorStore
from app.storage.metadata_store import MetadataStore

SERVICES_UNAVAILABLE_DETAIL = "Application services are not initialised"


@dataclass(slots=True)
class Services:
    """Everything a request handler needs, built once per process by the lifespan."""

    settings: Settings
    store: MetadataStore
    vector_store: VectorStore
    embedder: Embedder
    retrieval: RetrievalService
    ingestion: IngestionPipeline
    answer: AnswerService
    llm_model_name: str | None = None


def get_services(request: Request) -> Services:
    """Return the process-wide :class:`Services`; 503 while the app is not (yet) started."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SERVICES_UNAVAILABLE_DETAIL
        )
    return services


def get_request_id(request: Request) -> str:
    """The request id bound by ``RequestContextMiddleware`` (a fresh one if it is not running)."""
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = new_request_id()
        request.state.request_id = request_id
    return request_id


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries declaring an ``ErrorResponse`` body for each status code."""
    return {
        code: {"model": ErrorResponse, "description": HTTPStatus(code).phrase}
        for code in status_codes
    }


ServicesDep = Annotated[Services, Depends(get_services)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
