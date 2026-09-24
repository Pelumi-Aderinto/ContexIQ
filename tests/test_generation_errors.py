"""Failed LLM calls surface a sanitised, actionable reason instead of a bare 502."""

from __future__ import annotations

from app.api.main import _generation_detail
from app.generation.chain import GenerationError, sanitize_reason


def test_sanitize_reason_redacts_keys_and_collapses_whitespace() -> None:
    message = "Error   401 for key gsk_abcdefghijklmnop1234\nBearer sk-ant-secret-key-value"
    exc = RuntimeError(message)

    reason = sanitize_reason(exc)

    assert "gsk_" not in reason
    assert "sk-ant" not in reason
    assert "<redacted>" in reason
    assert "\n" not in reason and "  " not in reason


def test_sanitize_reason_caps_length_and_handles_empty_messages() -> None:
    assert sanitize_reason(RuntimeError("")) == "no details"
    assert len(sanitize_reason(RuntimeError("x" * 1000))) == 300


def test_generation_detail_includes_type_reason_and_model_hint() -> None:
    exc = GenerationError(
        "NotFoundError",
        "Error code: 404 - The model `llama-old` does not exist or you do not have access to it.",
    )

    detail = _generation_detail(exc)

    assert detail.startswith("LLM generation failed (NotFoundError): Error code: 404")
    assert "CONTEXTIQ_LLM_MODEL" in detail


def test_generation_detail_hints_at_key_for_auth_errors() -> None:
    detail = _generation_detail(GenerationError("AuthenticationError", "Invalid API key"))

    assert "API key" in detail
    assert "CONTEXTIQ_LLM_MODEL" not in detail


def test_generation_detail_without_reason_is_still_informative() -> None:
    detail = _generation_detail(GenerationError("TimeoutError"))

    assert detail == "LLM generation failed (TimeoutError)"
