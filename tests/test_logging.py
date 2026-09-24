"""Tests for ``app.core.logging``: idempotent configuration and the request-context middleware."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    configure_logging,
    get_logger,
    log_timing,
    sanitize_request_id,
)

HANDLER_NAME = "contextiq"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"log_json": True, "log_level": "INFO"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h.get_name() == HANDLER_NAME]


def build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    def ping(request: Request) -> dict[str, Any]:
        return {
            "request_id": request.state.request_id,
            "context": structlog.contextvars.get_contextvars(),
        }

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    return app


def json_events(captured: str) -> list[dict[str, Any]]:
    """Parse JSON log lines, skipping anything that is not one."""
    events = []
    for line in captured.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


@pytest.fixture(autouse=True)
def _reset_logging_state() -> Any:
    """Keep the global logging configuration from leaking between tests."""
    yield
    for handler in our_handlers():
        logging.getLogger().removeHandler(handler)
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


# ---- configure_logging ----------------------------------------------------------------------


def test_configure_twice_does_not_duplicate_handlers() -> None:
    configure_logging(make_settings())
    configure_logging(make_settings(log_json=False))
    assert len(our_handlers()) == 1


def test_configure_sets_level_and_routes_uvicorn() -> None:
    logging.getLogger("uvicorn.error").addHandler(logging.NullHandler())
    configure_logging(make_settings(log_level="debug"))
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("uvicorn.error").handlers == []
    assert logging.getLogger("uvicorn.error").propagate is True
    assert logging.getLogger("uvicorn.access").propagate is False


def test_unknown_level_falls_back_to_info() -> None:
    configure_logging(make_settings(log_level="loud"))
    assert logging.getLogger().level == logging.INFO


def test_stdlib_records_are_rendered_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    logging.getLogger("uvicorn.error").info("Application startup complete.")
    events = json_events(capsys.readouterr().err)
    assert any(
        e.get("event") == "Application startup complete." and e.get("logger") == "uvicorn.error"
        for e in events
    )


def test_console_renderer_emits_text(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings(log_json=False))
    get_logger("tests.console").info("console.event", answer=42)
    err = capsys.readouterr().err
    assert "console.event" in err
    assert "answer=42" in err


def test_exceptions_are_formatted_in_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    try:
        raise ValueError("kaboom")
    except ValueError:
        get_logger("tests.exc").exception("something.failed")
    events = json_events(capsys.readouterr().err)
    failed = next(e for e in events if e["event"] == "something.failed")
    assert failed["level"] == "error"
    assert "ValueError: kaboom" in failed["exception"]


# ---- request id sanitising -----------------------------------------------------------------


@pytest.mark.parametrize("good", ["abc", "trace-42_X", "a" * 64])
def test_sanitize_request_id_keeps_safe_ids(good: str) -> None:
    assert sanitize_request_id(good) == good


@pytest.mark.parametrize("bad", [None, "", "a" * 65, "bad id", "x\ny", "../etc", "ünïcode", "<b>"])
def test_sanitize_request_id_replaces_bad_ids(bad: str | None) -> None:
    generated = sanitize_request_id(bad)
    assert generated != bad
    assert len(generated) == 32
    int(generated, 16)  # uuid4().hex


# ---- middleware -----------------------------------------------------------------------------


def test_middleware_generates_request_id_when_missing() -> None:
    client = TestClient(build_app())
    response = client.get("/ping")
    request_id = response.headers[REQUEST_ID_HEADER]
    assert len(request_id) == 32
    body = response.json()
    assert body["request_id"] == request_id
    assert body["context"]["request_id"] == request_id
    assert body["context"]["method"] == "GET"
    assert body["context"]["path"] == "/ping"


def test_middleware_echoes_valid_provided_id() -> None:
    client = TestClient(build_app())
    response = client.get("/ping", headers={REQUEST_ID_HEADER: "trace-42_ok"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-42_ok"
    assert response.json()["request_id"] == "trace-42_ok"


def test_middleware_rejects_malicious_id_and_generates_new_one() -> None:
    client = TestClient(build_app())
    malicious = "evil\r\nX-Injected: 1"
    response = client.get("/ping", headers={REQUEST_ID_HEADER: "a" * 65})
    assert response.headers[REQUEST_ID_HEADER] != "a" * 65
    assert len(response.headers[REQUEST_ID_HEADER]) == 32
    assert sanitize_request_id(malicious) != malicious


def test_request_id_appears_in_completed_log_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    client = TestClient(build_app())
    response = client.get("/ping?token=do-not-log", headers={REQUEST_ID_HEADER: "trace-7"})
    assert response.status_code == 200

    err = capsys.readouterr().err
    events = json_events(err)
    completed = next(e for e in events if e["event"] == "request.completed")
    assert completed["request_id"] == "trace-7"
    assert completed["method"] == "GET"
    assert completed["path"] == "/ping"
    assert completed["status_code"] == 200
    assert completed["duration_ms"] >= 0
    assert completed["level"] == "info"
    assert completed["timestamp"].endswith("Z")
    ours = [e for e in events if e.get("logger") == "app.core.logging"]
    assert ours, "middleware log line missing"
    assert all("do-not-log" not in json.dumps(e) for e in ours)


def test_request_id_visible_via_capture_logs() -> None:
    configure_logging(make_settings())
    client = TestClient(build_app())
    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as entries:
        client.get("/ping", headers={REQUEST_ID_HEADER: "trace-8"})
    completed = next(e for e in entries if e["event"] == "request.completed")
    assert completed["request_id"] == "trace-8"
    assert completed["status_code"] == 200


def test_request_failed_is_logged_and_reraised(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(make_settings())
    client = TestClient(build_app(), raise_server_exceptions=False)
    response = client.get("/boom", headers={REQUEST_ID_HEADER: "trace-9"})
    assert response.status_code == 500

    events = json_events(capsys.readouterr().err)
    failed = next(e for e in events if e["event"] == "request.failed")
    assert failed["request_id"] == "trace-9"
    assert failed["exc_type"] == "RuntimeError"
    assert failed["level"] == "error"
    assert not any(e["event"] == "request.completed" for e in events)


def test_middleware_includes_workspace_when_principal_recorded() -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/scoped")
    def scoped(request: Request) -> dict[str, str]:
        request.state.principal = type("P", (), {"workspace_id": "alpha"})()
        return {"ok": "yes"}

    with structlog.testing.capture_logs() as entries:
        TestClient(app).get("/scoped")
    completed = next(e for e in entries if e["event"] == "request.completed")
    assert completed["workspace_id"] == "alpha"


# ---- log_timing -----------------------------------------------------------------------------


def test_log_timing_logs_duration_and_extra_fields() -> None:
    logger = get_logger("tests.timing")
    with structlog.testing.capture_logs() as entries, log_timing(logger, "step", n=1) as extra:
        extra["count"] = 3
    entry = next(e for e in entries if e["event"] == "step")
    assert entry["n"] == 1
    assert entry["count"] == 3
    assert entry["duration_ms"] >= 0


def test_log_timing_logs_failure_and_reraises() -> None:
    logger = get_logger("tests.timing")
    with (
        structlog.testing.capture_logs() as entries,
        pytest.raises(KeyError),
        log_timing(logger, "step"),
    ):
        raise KeyError("missing")
    entry = next(e for e in entries if e["event"] == "step.failed")
    assert entry["exc_type"] == "KeyError"
    assert entry["log_level"] == "error"
