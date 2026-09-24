"""Question answering and raw retrieval over the caller's workspace.

Both handlers are plain ``def`` so FastAPI runs them in its thread pool: embedding a query and
calling the LLM block for hundreds of milliseconds and must not stall the event loop.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import PrincipalDep, RequestIdDep, ServicesDep, error_responses
from app.models.schemas import QueryRequest, QueryResponse, SearchRequest, SearchResponse

router = APIRouter(tags=["query"], responses=error_responses(401, 422))


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question with verified citations",
    responses=error_responses(502),
)
def query(
    payload: QueryRequest, principal: PrincipalDep, services: ServicesDep, request_id: RequestIdDep
) -> QueryResponse:
    """Retrieve supporting chunks, generate a grounded answer and verify every citation."""
    return services.answer.answer(principal.workspace_id, payload, request_id=request_id)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve chunks without generating an answer",
)
def search(
    payload: SearchRequest, principal: PrincipalDep, services: ServicesDep, request_id: RequestIdDep
) -> SearchResponse:
    """Return the ranked chunks retrieval would hand to the LLM, with their debug scores."""
    return services.answer.search(principal.workspace_id, payload, request_id=request_id)
