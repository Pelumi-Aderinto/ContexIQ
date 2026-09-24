"""Chat model factory.

Builds a LangChain chat model from ``Settings`` via ``init_chat_model``. Returns ``None`` when
the resolved provider is ``extractive`` so the answer service can fall back to returning
passages instead of a synthesised answer.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.core.config import Settings

# Providers that accept sampling parameters. Anthropic Claude 5-family models reject them.
_SAMPLING_PROVIDERS = frozenset({"openai", "groq"})


class ConfigurationError(Exception):
    """Raised when the configured LLM provider or model cannot be initialised."""


def _api_key_for(settings: Settings, provider: str) -> SecretStr | None:
    keys: dict[str, SecretStr | None] = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "groq": settings.groq_api_key,
    }
    return keys.get(provider)


def _model_kwargs(settings: Settings, provider: str) -> dict[str, object]:
    """Keyword arguments for ``init_chat_model`` appropriate to ``provider``."""
    kwargs: dict[str, object] = {
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if provider in _SAMPLING_PROVIDERS:
        kwargs["temperature"] = settings.llm_temperature
    api_key = _api_key_for(settings, provider)
    if api_key is not None:
        kwargs["api_key"] = api_key.get_secret_value()
    return kwargs


def create_chat_model(settings: Settings) -> BaseChatModel | None:
    """Create the configured chat model, or ``None`` for extractive mode.

    Raises ``ConfigurationError`` when the provider package is missing or the provider/model
    combination is rejected. The error message never contains secrets.
    """
    provider = settings.resolved_llm_provider()
    if provider == "extractive":
        return None
    model = settings.resolved_llm_model()
    if not model:
        raise ConfigurationError(f"No LLM model configured for provider '{provider}'.")
    try:
        return init_chat_model(model, model_provider=provider, **_model_kwargs(settings, provider))
    except (ImportError, ValueError) as exc:
        raise ConfigurationError(
            f"Could not initialise LLM '{model}' for provider '{provider}' "
            f"({type(exc).__name__}). Check the provider, model name and API key settings."
        ) from exc
