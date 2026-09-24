"""FastAPI application factory and process entrypoint for the ContextIQ API.

``create_app`` wires the HTTP layer: request-id middleware, CORS, the routers and the exception
handlers that turn domain errors into ``ErrorResponse`` bodies. The heavy resources (embedding
model, SQLite store, FAISS indexes, chat model) are opened in the lifespan by ``build_services``
so importing this module stays cheap and tests can inject fakes for the embedder and the LLM.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from http import HTTPStatus
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.deps import Services
from app.api.routes import documents, health, query
from app.core.config import Settings, get_settings
from app.core.logging import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    configure_logging,
    get_logger,
    new_request_id,
)
from app.core.security import AuthError
from app.generation.chain import AnswerService, GenerationError
from app.generation.llm import ConfigurationError, create_chat_model
from app.ingestion.parser import PdfParseError
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import Embedder, create_embedder
from app.retrieval.retriever import CrossEncoderReranker, Reranker, RetrievalService
from app.retrieval.vector_store import VectorStore
from app.storage.metadata_store import MetadataStore

__all__ = ["Services", "app", "build_services", "create_app", "run"]

log = get_logger(__name__)

LLMOption = BaseChatModel | None | Literal["auto"]
"""``build_services`` accepts a chat model, ``None`` (extractive mode) or ``"auto"``."""

API_TITLE = "ContextIQ API"
API_DESCRIPTION = (
    "Document-grounded question answering over PDFs with verified citations. "
    "Authenticate with the `X-API-Key` header: every key is scoped to exactly one workspace "
    "and nothing in a request body can select another one."
)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
UNAUTHORIZED_DETAIL = "Invalid or missing API key"
GENERATION_FAILED_DETAIL = "LLM generation failed"
INTERNAL_ERROR_DETAIL = "Internal server error"
_VALIDATION_ERROR = "Validation error"
_MAX_DETAIL_CHARS = 200


# --------------------------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------------------------


def _build_reranker(settings: Settings) -> Reranker | None:
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(settings.rerank_model, device=settings.embedding_device)


def _resolve_llm(settings: Settings, llm: LLMOption) -> BaseChatModel | None:
    """``"auto"`` builds the configured model; anything else is used as given."""
    if isinstance(llm, str):
        if llm != "auto":
            raise ValueError(f"llm must be a chat model, None or 'auto', not {llm!r}")
        return create_chat_model(settings)
    return llm


def _llm_model_name(llm: BaseChatModel | None, settings: Settings) -> str | None:
    """Display name for the active model: the configured one, else what the model reports."""
    if llm is None:
        return None
    configured = settings.resolved_llm_model()
    if configured:
        return configured
    for attribute in ("model_name", "model"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(llm).__name__


def build_services(
    settings: Settings, *, embedder: Embedder | None = None, llm: LLMOption = "auto"
) -> Services:
    """Open the stores and build the retrieval, ingestion and answer services.

    ``embedder`` and ``llm`` can be injected (tests, scripts); ``llm="auto"`` resolves the chat
    model from settings and ``None`` forces extractive mode. If anything after the store fails
    to build, the store is closed before the exception propagates.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store = MetadataStore(settings.db_path)
    try:
        active_embedder = embedder if embedder is not None else create_embedder(settings)
        vector_store = VectorStore(settings.index_dir, active_embedder.dimension)
        retrieval = RetrievalService(
            settings=settings,
            store=store,
            vector_store=vector_store,
            embedder=active_embedder,
            reranker=_build_reranker(settings),
        )
        ingestion = IngestionPipeline(
            settings=settings, store=store, vector_store=vector_store, embedder=active_embedder
        )
        repaired = ingestion.reconcile_interrupted()
        if repaired:
            log.warning("app.reconciled_interrupted_documents", count=repaired)
        chat_model = _resolve_llm(settings, llm)
        answer = AnswerService(settings=settings, retrieval=retrieval, llm=chat_model)
    except Exception:
        store.close()
        raise
    return Services(
        settings=settings,
        store=store,
        vector_store=vector_store,
        embedder=active_embedder,
        retrieval=retrieval,
        ingestion=ingestion,
        answer=answer,
        llm_model_name=_llm_model_name(chat_model, settings),
    )


# --------------------------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------------------------


def _load_api_key_map(settings: Settings) -> dict[str, str]:
    """Parse the configured keys up front so a malformed ``CONTEXTIQ_API_KEYS`` fails startup."""
    if settings.auth_mode != "api_key":
        return {}
    try:
        key_map = settings.api_key_map()
    except ValueError as exc:
        log.error("auth.misconfigured", error_type=type(exc).__name__)
        raise
    if not key_map:
        log.warning("auth.no_api_keys_configured")
    return key_map


def _log_started(services: Services) -> None:
    settings = services.settings
    workspaces = services.store.list_workspaces()
    load_errors = services.vector_store.load_errors
    if load_errors:
        log.error(
            "app.index_load_errors",
            workspaces=sorted(load_errors),
            hint="documents in these workspaces are listed but not searchable; "
            "run scripts/rebuild_index.py",
        )
    log.info(
        "app.started",
        version=__version__,
        workspaces=len(workspaces),
        vectors=sum(services.vector_store.count(workspace) for workspace in workspaces),
        embedding_model=services.embedder.model_name,
        embedding_dimension=services.embedder.dimension,
        llm_model=services.llm_model_name,
        retrieval_mode=settings.retrieval_mode,
        rerank_enabled=services.retrieval.reranker is not None,
        auth_mode=settings.auth_mode,
    )


def _start_services(settings: Settings, *, embedder: Embedder | None, llm: LLMOption) -> Services:
    try:
        services = build_services(settings, embedder=embedder, llm=llm)
    except Exception as exc:
        log.error("app.startup_failed", error_type=type(exc).__name__)
        raise
    _log_started(services)
    return services


def _make_lifespan(
    settings: Settings, embedder: Embedder | None, llm: LLMOption
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Lifespan that builds the services on startup and closes the store on shutdown."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings)
        application.state.api_key_map = _load_api_key_map(settings)
        services = _start_services(settings, embedder=embedder, llm=llm)
        application.state.services = services
        try:
            yield
        finally:
            application.state.services = None
            services.store.close()
            log.info("app.stopped")

    return lifespan


# --------------------------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------------------------


def _request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", None) or new_request_id()


def _phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


def _truncate(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _error_response(
    request: Request,
    status_code: int,
    *,
    detail: Any = None,
    error: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """An ``ErrorResponse``-shaped JSON body carrying the request id in body and header.

    ``detail`` is normally a string; validation errors pass FastAPI's list of error records.
    The header is set here as well because 500 responses are produced outside the request
    middleware and would otherwise lack it.
    """
    request_id = _request_id_of(request)
    content = {"request_id": request_id, "error": error or _phrase(status_code), "detail": detail}
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(content),
        headers={REQUEST_ID_HEADER: request_id, **(headers or {})},
    )


async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error_response(request, exc.status_code, detail=exc.detail, headers=exc.headers)


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(
        request, HTTPStatus.UNPROCESSABLE_CONTENT, error=_VALIDATION_ERROR, detail=exc.errors()
    )


async def _handle_auth_error(request: Request, exc: AuthError) -> JSONResponse:
    log.info("auth.rejected", reason=exc.reason)
    return _error_response(
        request,
        HTTPStatus.UNAUTHORIZED,
        detail=UNAUTHORIZED_DETAIL,
        headers={"WWW-Authenticate": "ApiKey"},
    )


async def _handle_pdf_parse_error(request: Request, exc: PdfParseError) -> JSONResponse:
    return _error_response(
        request, HTTPStatus.UNPROCESSABLE_CONTENT, detail=f"{exc.code}: {exc.message}"
    )


async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    log.warning("request.invalid_value", error_type=type(exc).__name__)
    return _error_response(
        request, HTTPStatus.UNPROCESSABLE_CONTENT, detail=_truncate(str(exc)) or None
    )


def _generation_detail(exc: GenerationError) -> str:
    """User-facing reason for a failed LLM call, with a hint for the most common causes."""
    detail = f"{GENERATION_FAILED_DETAIL} ({exc.error_type})"
    if exc.reason:
        detail += f": {exc.reason}"
    lowered = exc.reason.lower()
    if "model_not_found" in lowered or "does not exist" in lowered:
        detail += " Hint: set CONTEXTIQ_LLM_MODEL to a model your provider account can access."
    elif exc.error_type in {"AuthenticationError", "PermissionDeniedError"}:
        detail += " Hint: check the provider API key in .env."
    return detail


async def _handle_generation_error(request: Request, exc: GenerationError) -> JSONResponse:
    log.warning("request.generation_failed", error_type=exc.error_type)
    return _error_response(request, HTTPStatus.BAD_GATEWAY, detail=_generation_detail(exc))


async def _handle_configuration_error(request: Request, exc: ConfigurationError) -> JSONResponse:
    log.error("request.llm_misconfigured", error_type=type(exc).__name__)
    return _error_response(request, HTTPStatus.INTERNAL_SERVER_ERROR, detail=_truncate(str(exc)))


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("request.unhandled_error", error_type=type(exc).__name__, exc_info=exc)
    return _error_response(request, HTTPStatus.INTERNAL_SERVER_ERROR, detail=INTERNAL_ERROR_DETAIL)


def _register_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    application.add_exception_handler(RequestValidationError, _handle_validation_error)
    application.add_exception_handler(AuthError, _handle_auth_error)
    application.add_exception_handler(PdfParseError, _handle_pdf_parse_error)
    application.add_exception_handler(ValueError, _handle_value_error)
    application.add_exception_handler(GenerationError, _handle_generation_error)
    application.add_exception_handler(ConfigurationError, _handle_configuration_error)
    application.add_exception_handler(Exception, _handle_unexpected_error)


# --------------------------------------------------------------------------------------------
# Application factory and entrypoint
# --------------------------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMOption = "auto",
) -> FastAPI:
    """Build the FastAPI application; heavy resources are created when the lifespan starts.

    ``settings`` defaults to the environment-driven :func:`get_settings`. ``embedder`` and
    ``llm`` are forwarded to :func:`build_services` so tests can substitute fakes.
    """
    active_settings = settings if settings is not None else get_settings()
    application = FastAPI(
        title=API_TITLE,
        version=__version__,
        description=API_DESCRIPTION,
        lifespan=_make_lifespan(active_settings, embedder, llm),
    )
    application.state.settings = active_settings
    # Added first so RequestContextMiddleware (added last) wraps it and tags CORS replies too.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    application.add_middleware(RequestContextMiddleware)
    application.include_router(health.router)
    application.include_router(documents.router)
    application.include_router(query.router)
    _register_exception_handlers(application)
    return application


app = create_app()


def _host_from_env() -> str:
    return os.environ.get("CONTEXTIQ_HOST", "").strip() or DEFAULT_HOST


def _port_from_env() -> int:
    raw = os.environ.get("CONTEXTIQ_PORT", "").strip() or str(DEFAULT_PORT)
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"CONTEXTIQ_PORT must be an integer, got {raw!r}") from exc
    if not 0 < port < 65536:
        raise SystemExit(f"CONTEXTIQ_PORT must be between 1 and 65535, got {port}")
    return port


def run() -> None:
    """Serve the module-level ``app`` with uvicorn (the ``contextiq-api`` console script).

    Host and port come from ``CONTEXTIQ_HOST`` / ``CONTEXTIQ_PORT`` (default ``0.0.0.0:8000``).
    Uvicorn's own logging config is disabled so its records flow through ``configure_logging``.
    """
    configure_logging(get_settings())
    uvicorn.run(
        "app.api.main:app",
        host=_host_from_env(),
        port=_port_from_env(),
        reload=False,
        log_config=None,
    )
