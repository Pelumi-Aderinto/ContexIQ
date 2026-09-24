"""Liveness and configuration summary. No authentication: it reveals nothing workspace-specific."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.deps import Services, ServicesDep
from app.models.schemas import HealthResponse, RetrievalMode

router = APIRouter(tags=["health"])

_EXTRACTIVE = "extractive"
_CUSTOM_PROVIDER = "custom"


def _llm_provider(services: Services) -> str:
    """Resolved provider name, or ``custom`` for an injected model that settings do not describe."""
    if services.llm_model_name is None:
        return _EXTRACTIVE
    provider = services.settings.resolved_llm_provider()
    return provider if provider != _EXTRACTIVE else _CUSTOM_PROVIDER


def _warnings(services: Services) -> list[str]:
    """Operator-facing warnings; an index that failed to load makes the service degraded."""
    return [
        f"Index for workspace '{workspace}' was not loaded ({reason}); its documents are listed "
        "but not searchable until scripts/rebuild_index.py is run."
        for workspace, reason in sorted(services.vector_store.load_errors.items())
    ]


@router.get("/health", response_model=HealthResponse, summary="Service health and configuration")
async def health(services: ServicesDep) -> HealthResponse:
    """Report the running configuration and whether every persisted index could be loaded."""
    settings = services.settings
    warnings = _warnings(services)
    return HealthResponse(
        status="degraded" if warnings else "ok",
        warnings=warnings,
        version=__version__,
        embedding_model=services.embedder.model_name,
        embedding_dimension=services.embedder.dimension,
        llm_provider=_llm_provider(services),
        llm_model=services.llm_model_name,
        retrieval_mode=RetrievalMode(settings.retrieval_mode),
        reranker_enabled=services.retrieval.reranker is not None,
        auth_mode=settings.auth_mode,
    )
