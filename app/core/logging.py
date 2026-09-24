"""Structured logging with structlog and per-request log context.

``configure_logging`` wires structlog *and* the standard library (uvicorn, fastapi and any
third-party logger) through a single ``ProcessorFormatter`` so every line is rendered the same
way: JSON in production, a coloured console renderer for development.
``RequestContextMiddleware`` binds a request id, method and path into structlog's contextvars
so every line emitted while handling a request carries them.

Per design invariant 6 nothing here logs query strings, headers, bodies or document contents:
only identifiers, counts, durations and error types.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from structlog.typing import Processor

from app.core.config import Settings

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_MAX_LENGTH = 64
_REQUEST_ID_PATTERN = re.compile(rf"^[A-Za-z0-9_-]{{1,{REQUEST_ID_MAX_LENGTH}}}$")

_HANDLER_NAME = "contextiq"
# Loggers whose output is re-routed through our formatter so it is uniform with ours.
_ROUTED_STDLIB_LOGGERS = ("uvicorn", "uvicorn.error", "fastapi")
# uvicorn's access log prints the full request line including the query string, which we never
# log; ``RequestContextMiddleware`` emits the per-request line instead.
_SILENCED_STDLIB_LOGGERS = ("uvicorn.access",)
# HTTP client libraries log every request URL at INFO; keep only their warnings and errors.
_QUIET_STDLIB_LOGGERS = ("httpx", "httpx2", "httpcore")


class _StderrHandler(logging.StreamHandler):
    """Stream handler that resolves ``sys.stderr`` on every emit.

    Looking the stream up lazily keeps logging working when stderr is swapped at runtime
    (pytest's capture, some process supervisors) instead of writing to a stale stream.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        """Ignored on purpose: the stream is always the current ``sys.stderr``."""


def _shared_processors() -> list[Processor]:
    """Processors applied to structlog events and to foreign stdlib records alike."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def _console_exception_formatter() -> structlog.dev.ExceptionRenderer:
    """Pretty tracebacks without local variables, which could expose document text or keys."""
    try:
        import rich  # noqa: F401
    except ImportError:
        return structlog.dev.plain_traceback
    return structlog.dev.RichTracebackFormatter(show_locals=False)


def _renderers(settings: Settings) -> list[Processor]:
    """Final rendering step: JSON lines, or a console renderer that formats tracebacks itself."""
    if settings.log_json:
        return [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    return [
        structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(), exception_formatter=_console_exception_formatter()
        )
    ]


def _parse_level(name: str) -> int:
    """Translate a level name such as ``"debug"`` to its numeric value (unknown -> INFO)."""
    return logging.getLevelNamesMapping().get(name.strip().upper(), logging.INFO)


def _install_root_handler(handler: logging.Handler, level: int) -> None:
    """Replace any handler installed by a previous call, leaving foreign handlers alone."""
    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


def _route_stdlib_loggers() -> None:
    """Strip handlers uvicorn/fastapi may have installed so records reach the root handler."""
    for name in _ROUTED_STDLIB_LOGGERS:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers.clear()
        stdlib_logger.propagate = True
    for name in _SILENCED_STDLIB_LOGGERS:
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers.clear()
        stdlib_logger.propagate = False
    for name in _QUIET_STDLIB_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_logging(settings: Settings) -> None:
    """Configure structlog and route stdlib logging through it. Safe to call repeatedly."""
    shared = _shared_processors()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *_renderers(settings),
        ],
    )
    handler = _StderrHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(formatter)

    _install_root_handler(handler, _parse_level(settings.log_level))
    _route_stdlib_loggers()
    logging.captureWarnings(True)

    if settings.log_level.strip().upper() not in logging.getLevelNamesMapping():
        get_logger(__name__).warning("logging.unknown_level", configured=settings.log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger for ``name`` (normally the calling module's ``__name__``)."""
    return structlog.stdlib.get_logger(name)


def new_request_id() -> str:
    """Generate a fresh request id."""
    return uuid.uuid4().hex


def sanitize_request_id(raw: str | None) -> str:
    """Return ``raw`` if it is a safe id (1-64 chars of ``[A-Za-z0-9_-]``), else a new one."""
    if raw is not None and _REQUEST_ID_PATTERN.fullmatch(raw):
        return raw
    return new_request_id()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)


def _principal_fields(request: Request) -> dict[str, Any]:
    """Workspace of the authenticated caller, if the auth dependency recorded one."""
    principal = getattr(request.state, "principal", None)
    workspace_id = getattr(principal, "workspace_id", None)
    return {"workspace_id": workspace_id} if workspace_id else {}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id into the log context and log exactly one line per request.

    The id comes from the ``X-Request-ID`` header when it is well formed, otherwise a new
    ``uuid4().hex`` is generated. It is stored on ``request.state.request_id`` and echoed back
    in the ``X-Request-ID`` response header. Only method, path (never the query string),
    status code and duration are logged.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        structlog.contextvars.clear_contextvars()
        request_id = sanitize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(
            request_id=request_id, method=request.method, path=request.url.path
        )
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            _log.error(
                "request.failed",
                exc_type=type(exc).__name__,
                duration_ms=_elapsed_ms(started),
                **_principal_fields(request),
            )
            raise
        response.headers[REQUEST_ID_HEADER] = request_id
        _log.info(
            "request.completed",
            status_code=response.status_code,
            duration_ms=_elapsed_ms(started),
            **_principal_fields(request),
        )
        return response


@contextmanager
def log_timing(
    logger: structlog.stdlib.BoundLogger, event: str, **fields: Any
) -> Iterator[dict[str, Any]]:
    """Log ``event`` with ``duration_ms`` when the block exits.

    Yields a dict; keys added to it inside the block (counts, ids) are included in the final
    line. If the block raises, ``"<event>.failed"`` is logged with the exception type and the
    exception is re-raised.
    """
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    try:
        yield extra
    except Exception as exc:
        logger.error(
            f"{event}.failed",
            duration_ms=_elapsed_ms(started),
            exc_type=type(exc).__name__,
            **{**fields, **extra},
        )
        raise
    logger.info(event, duration_ms=_elapsed_ms(started), **{**fields, **extra})


_log = get_logger(__name__)
