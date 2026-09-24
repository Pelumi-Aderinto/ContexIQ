"""Settings parsing, including the shipped .env.example, which must always start the service."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_env_example_loads_and_resolves_extractive_mode() -> None:
    settings = Settings(_env_file=REPO_ROOT / ".env.example")

    assert settings.auth_mode == "api_key"
    assert settings.api_key_map() == {"dev-key-alpha-0001": "alpha", "dev-key-beta-0002": "beta"}
    assert settings.cors_origins == ["http://localhost:8501"]
    assert settings.resolved_llm_provider() == "extractive"
    assert settings.resolved_llm_model() is None


def test_env_example_documents_every_public_setting() -> None:
    """Every CONTEXTIQ_* setting a user may tune is listed in .env.example (or is internal)."""
    text = (REPO_ROOT / ".env.example").read_text()
    internal = {"app_name", "environment", "default_workspace_id", "max_files_per_upload",
                "embedding_query_prefix", "embedding_cache_dir", "hybrid_candidate_multiplier",
                "rrf_k", "min_dense_score", "rerank_candidates", "llm_max_retries",
                "max_context_chars", "citation_excerpt_chars", "anthropic_api_key",
                "openai_api_key", "groq_api_key"}  # fmt: skip
    missing = [
        name
        for name in Settings.model_fields
        if name not in internal and f"CONTEXTIQ_{name.upper()}" not in text
    ]
    assert missing == []


def test_cors_origins_accepts_comma_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTIQ_CORS_ORIGINS", "http://a:1, http://b:2 ,")

    assert Settings(_env_file=None).cors_origins == ["http://a:1", "http://b:2"]


def test_cors_origins_accepts_json_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTIQ_CORS_ORIGINS", '["http://a:1", "http://b:2"]')

    assert Settings(_env_file=None).cors_origins == ["http://a:1", "http://b:2"]


def test_empty_provider_keys_mean_extractive(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("CONTEXTIQ_LLM_MODEL", "")

    settings = Settings(_env_file=None)

    assert settings.resolved_llm_provider() == "extractive"
    assert settings.resolved_llm_model() is None


def test_auto_provider_prefers_anthropic_then_openai_then_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert Settings(_env_file=None).resolved_llm_provider() == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert Settings(_env_file=None).resolved_llm_provider() == "anthropic"
    assert Settings(_env_file=None).resolved_llm_model() == "claude-opus-5"


def test_api_key_map_rejects_short_keys_and_bad_workspaces() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        Settings(_env_file=None, api_keys="short:alpha").api_key_map()
    with pytest.raises(ValueError, match="Invalid workspace"):
        Settings(_env_file=None, api_keys="long-enough-key:Bad Space").api_key_map()
    with pytest.raises(ValueError, match="key:workspace"):
        Settings(_env_file=None, api_keys="no-colon-here").api_key_map()
