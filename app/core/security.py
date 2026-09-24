"""API-key authentication and workspace scoping.

The workspace a caller may touch is derived *only* from the credential presented in the
``X-API-Key`` header (design invariant 2 in ``docs/CONTRACTS.md``). Nothing in a request body
can ever select a workspace, so cross-workspace access is impossible by construction.
"""

from __future__ import annotations

import secrets

import structlog
from fastapi import Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from app.core.config import WORKSPACE_ID_PATTERN, Settings, get_settings
from app.core.logging import get_logger

API_KEY_HEADER = "X-API-Key"
KEY_LABEL_PREFIX_CHARS = 4
_UNAUTHORIZED_DETAIL = "Invalid or missing API key"
_MISCONFIGURED_DETAIL = "Authentication is misconfigured"

log = get_logger(__name__)


class Principal(BaseModel):
    """The authenticated caller: a workspace plus a log-safe label of the key that was used."""

    model_config = ConfigDict(frozen=True)

    workspace_id: str
    key_label: str


class AuthError(Exception):
    """Authentication failed. ``reason`` is ``"missing"`` or ``"invalid"``; never the key."""

    def __init__(self, reason: str = "invalid") -> None:
        super().__init__(f"authentication failed: {reason}")
        self.reason = reason


def make_key_label(api_key: str) -> str:
    """Return a short label safe for logs: the first four characters followed by ``...``."""
    return f"{api_key[:KEY_LABEL_PREFIX_CHARS]}..."


def validate_workspace_id(workspace_id: str) -> str:
    """Return ``workspace_id`` if it matches ``WORKSPACE_ID_PATTERN``, else raise ``ValueError``."""
    if not isinstance(workspace_id, str) or not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
        raise ValueError(
            "workspace_id must be 1-64 characters of lowercase letters, digits, '_' or '-', "
            "starting with a letter or digit"
        )
    return workspace_id


def authenticate(api_key: str | None, key_map: dict[str, str]) -> Principal:
    """Map an API key to its workspace.

    Every configured key is compared with ``secrets.compare_digest`` and the loop never exits
    early, so the time taken does not reveal which (if any) key matched.
    """
    if not api_key:
        raise AuthError("missing")
    provided = api_key.encode("utf-8")
    matched_workspace: str | None = None
    for candidate, workspace_id in key_map.items():
        if secrets.compare_digest(provided, candidate.encode("utf-8")):
            matched_workspace = workspace_id
    if matched_workspace is None:
        raise AuthError("invalid")
    return Principal(workspace_id=matched_workspace, key_label=make_key_label(api_key))


async def get_principal(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> Principal:
    """FastAPI dependency resolving the caller's workspace from the ``X-API-Key`` header.

    With ``auth_mode="disabled"`` every request maps to ``settings.default_workspace_id``.
    Otherwise a failed lookup raises a 401 carrying ``WWW-Authenticate: ApiKey``; the response
    deliberately does not say whether the key was missing or wrong. On success the workspace
    and key label are bound into the structlog context and stored on ``request.state``.

    Declared ``async`` on purpose: FastAPI runs sync dependencies in a worker thread with a
    *copy* of the request context, so contextvars bound there would be lost. Nothing here
    blocks, so running on the event loop is free.
    """
    settings = _resolve_settings(request)
    try:
        if settings.auth_mode == "disabled":
            principal = Principal(
                workspace_id=validate_workspace_id(settings.default_workspace_id),
                key_label="disabled",
            )
        else:
            principal = authenticate(x_api_key, _resolve_key_map(request, settings))
    except AuthError as exc:
        log.info("auth.rejected", reason=exc.reason)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED_DETAIL,
            headers={"WWW-Authenticate": "ApiKey"},
        ) from exc
    except ValueError as exc:
        log.error("auth.misconfigured", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=_MISCONFIGURED_DETAIL
        ) from exc
    _bind_principal(request, principal)
    return principal


def _resolve_settings(request: Request) -> Settings:
    """Prefer the settings wired into ``app.state.services``; fall back to the global ones."""
    services = getattr(request.app.state, "services", None)
    settings = getattr(services, "settings", None)
    return settings if settings is not None else get_settings()


def _resolve_key_map(request: Request, settings: Settings) -> dict[str, str]:
    """Parse the configured keys once per app and cache them on ``app.state.api_key_map``."""
    state = request.app.state
    cached = getattr(state, "api_key_map", None)
    if cached is not None:
        return cached
    key_map = settings.api_key_map()
    state.api_key_map = key_map
    return key_map


def _bind_principal(request: Request, principal: Principal) -> None:
    """Expose the principal to later log lines (request middleware) and processors."""
    request.state.principal = principal
    structlog.contextvars.bind_contextvars(
        workspace_id=principal.workspace_id, key_label=principal.key_label
    )
