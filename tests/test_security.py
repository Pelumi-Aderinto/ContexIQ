"""Tests for ``app.core.security``: key -> workspace mapping and the FastAPI dependency."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import structlog
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core import security
from app.core.config import Settings
from app.core.security import (
    AuthError,
    Principal,
    authenticate,
    get_principal,
    make_key_label,
    validate_workspace_id,
)

ALPHA_KEY = "alpha-key-0001"
BETA_KEY = "beta-key-00002"
KEY_MAP = {ALPHA_KEY: "alpha", BETA_KEY: "beta"}


def make_settings(**overrides: Any) -> Settings:
    """Settings that ignore any local ``.env`` so tests are hermetic."""
    values: dict[str, Any] = {
        "auth_mode": "api_key",
        "api_keys": f"{ALPHA_KEY}:alpha,{BETA_KEY}:beta",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def build_app(settings: Settings | None) -> FastAPI:
    """Minimal app exercising ``get_principal`` the way the real API does."""
    app = FastAPI()
    if settings is not None:
        app.state.services = SimpleNamespace(settings=settings)

    @app.get("/whoami")
    def whoami(principal: Principal = Depends(get_principal)) -> dict[str, Any]:
        return {
            "workspace_id": principal.workspace_id,
            "key_label": principal.key_label,
            "context": structlog.contextvars.get_contextvars(),
        }

    return app


# ---- authenticate ---------------------------------------------------------------------------


def test_valid_key_maps_to_its_workspace() -> None:
    assert authenticate(ALPHA_KEY, KEY_MAP).workspace_id == "alpha"


def test_other_valid_key_maps_to_other_workspace() -> None:
    assert authenticate(BETA_KEY, KEY_MAP).workspace_id == "beta"


@pytest.mark.parametrize(
    "bad_key", ["wrong-key", ALPHA_KEY[:-1], ALPHA_KEY + "x", "Alpha-Key-0001"]
)
def test_wrong_key_raises_invalid(bad_key: str) -> None:
    with pytest.raises(AuthError) as excinfo:
        authenticate(bad_key, KEY_MAP)
    assert excinfo.value.reason == "invalid"


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_key_raises_missing(missing: str | None) -> None:
    with pytest.raises(AuthError) as excinfo:
        authenticate(missing, KEY_MAP)
    assert excinfo.value.reason == "missing"


def test_empty_key_map_rejects_everything() -> None:
    with pytest.raises(AuthError):
        authenticate(ALPHA_KEY, {})


def test_auth_error_message_never_contains_the_key() -> None:
    with pytest.raises(AuthError) as excinfo:
        authenticate("super-secret-key", KEY_MAP)
    assert "super-secret-key" not in str(excinfo.value)


# ---- key labels -----------------------------------------------------------------------------


def test_key_label_is_truncated() -> None:
    principal = authenticate(ALPHA_KEY, KEY_MAP)
    assert principal.key_label == "alph..."
    assert ALPHA_KEY not in principal.key_label


def test_make_key_label_short_key() -> None:
    assert make_key_label("ab") == "ab..."


def test_principal_is_immutable() -> None:
    principal = Principal(workspace_id="alpha", key_label="alph...")
    with pytest.raises(ValidationError):
        principal.workspace_id = "beta"  # type: ignore[misc]


# ---- validate_workspace_id ------------------------------------------------------------------


@pytest.mark.parametrize("valid", ["alpha", "a", "0abc", "ws-1_x", "a" * 64])
def test_validate_workspace_id_accepts(valid: str) -> None:
    assert validate_workspace_id(valid) == valid


@pytest.mark.parametrize(
    "invalid",
    ["", "-abc", "_abc", "Alpha", "a b", "a/b", "../etc", "a" * 65, "ws.1", "ünï"],
)
def test_validate_workspace_id_rejects(invalid: str) -> None:
    with pytest.raises(ValueError):
        validate_workspace_id(invalid)


def test_validate_workspace_id_rejects_non_strings() -> None:
    with pytest.raises(ValueError):
        validate_workspace_id(None)  # type: ignore[arg-type]


# ---- get_principal dependency ----------------------------------------------------------------


def test_dependency_returns_401_without_key() -> None:
    client = TestClient(build_app(make_settings()))
    response = client.get("/whoami")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    assert response.json()["detail"] == "Invalid or missing API key"


def test_dependency_returns_401_with_wrong_key() -> None:
    client = TestClient(build_app(make_settings()))
    response = client.get("/whoami", headers={"X-API-Key": "not-a-real-key"})
    assert response.status_code == 401


def test_dependency_returns_200_with_key_and_binds_context() -> None:
    client = TestClient(build_app(make_settings()))
    response = client.get("/whoami", headers={"X-API-Key": ALPHA_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["workspace_id"] == "alpha"
    assert body["key_label"] == "alph..."
    assert body["context"]["workspace_id"] == "alpha"
    assert body["context"]["key_label"] == "alph..."


def test_dependency_scopes_each_key_to_its_own_workspace() -> None:
    client = TestClient(build_app(make_settings()))
    beta = client.get("/whoami", headers={"X-API-Key": BETA_KEY}).json()
    assert beta["workspace_id"] == "beta"


def test_dependency_caches_key_map_on_app_state() -> None:
    app = build_app(make_settings())
    client = TestClient(app)
    assert client.get("/whoami", headers={"X-API-Key": ALPHA_KEY}).status_code == 200
    assert app.state.api_key_map == KEY_MAP


def test_disabled_auth_mode_returns_default_workspace() -> None:
    settings = make_settings(auth_mode="disabled", default_workspace_id="sandbox", api_keys="")
    client = TestClient(build_app(settings))
    response = client.get("/whoami")
    assert response.status_code == 200
    assert response.json() == {
        "workspace_id": "sandbox",
        "key_label": "disabled",
        "context": {"workspace_id": "sandbox", "key_label": "disabled"},
    }


def test_disabled_auth_mode_ignores_any_key() -> None:
    client = TestClient(build_app(make_settings(auth_mode="disabled", api_keys="")))
    response = client.get("/whoami", headers={"X-API-Key": "whatever"})
    assert response.json()["workspace_id"] == "default"


def test_falls_back_to_global_settings_without_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(security, "get_settings", lambda: make_settings())
    client = TestClient(build_app(settings=None))
    assert client.get("/whoami").status_code == 401
    assert client.get("/whoami", headers={"X-API-Key": BETA_KEY}).json()["workspace_id"] == "beta"


def test_malformed_key_config_returns_500_not_401() -> None:
    client = TestClient(build_app(make_settings(api_keys="short:alpha")))
    response = client.get("/whoami", headers={"X-API-Key": "short"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Authentication is misconfigured"
