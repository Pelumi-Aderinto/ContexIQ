"""Application settings.

All settings are read from environment variables prefixed with ``CONTEXTIQ_`` (or from a
``.env`` file). Provider API keys use their conventional unprefixed names
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GROQ_API_KEY``) so existing shells work unchanged.

Secrets are ``SecretStr`` and are never logged or returned by the API.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProvider = Literal["auto", "anthropic", "openai", "groq", "extractive"]
AuthMode = Literal["api_key", "disabled"]
RetrievalModeName = Literal["dense", "hybrid"]

WORKSPACE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5-mini",
    "groq": "openai/gpt-oss-120b",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONTEXTIQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- general -----------------------------------------------------------------------
    app_name: str = "ContextIQ"
    environment: Literal["development", "test", "production"] = "development"
    data_dir: Path = Field(
        default=Path("data"), description="Root for the SQLite DB and FAISS indexes."
    )
    log_level: str = "INFO"
    log_json: bool = Field(default=True, description="Emit JSON logs (False = console renderer).")
    # ``NoDecode``: pydantic-settings would otherwise JSON-decode the env value before the
    # validator below runs, so a plain "http://a,http://b" in .env would crash start-up.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8501"],
        description="Comma-separated origins allowed by CORS (a JSON list is accepted too).",
    )

    # ---- auth --------------------------------------------------------------------------
    auth_mode: AuthMode = "api_key"
    api_keys: str = Field(
        default="",
        description="Comma-separated 'api_key:workspace_id' pairs, e.g. 'k1:alpha,k2:beta'.",
    )
    default_workspace_id: str = Field(
        default="default", description="Workspace used when auth_mode='disabled'."
    )

    # ---- ingestion ---------------------------------------------------------------------
    max_upload_mb: int = Field(default=25, ge=1, le=500)
    max_pages: int = Field(default=500, ge=1)
    max_files_per_upload: int = Field(default=10, ge=1, le=100)
    chunk_size: int = Field(default=1000, ge=100, description="Chunk size in characters.")
    chunk_overlap: int = Field(default=150, ge=0)

    # ---- embeddings --------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = Field(default="cpu", description="'cpu', 'cuda', 'mps' or 'auto'.")
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="Instruction prepended to queries (BGE models benefit; set '' for others).",
    )
    embedding_cache_dir: Path | None = Field(
        default=None, description="Hugging Face cache folder. Defaults to HF's own default."
    )

    # ---- retrieval ---------------------------------------------------------------------
    retrieval_mode: RetrievalModeName = "hybrid"
    top_k: int = Field(default=5, ge=1)
    max_top_k: int = Field(default=20, ge=1)
    hybrid_candidate_multiplier: int = Field(
        default=3, ge=1, description="Each retriever fetches top_k * multiplier candidates."
    )
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal rank fusion constant.")
    min_dense_score: float = Field(
        default=0.0,
        description="Drop dense hits below this cosine similarity. 0 disables. Uncalibrated.",
    )
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1)

    # ---- generation --------------------------------------------------------------------
    llm_provider: LLMProvider = "auto"
    llm_model: str | None = None
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    max_context_chars: int = Field(
        default=12000, ge=1000, description="Upper bound on total context characters sent to LLM."
    )
    citation_excerpt_chars: int = Field(default=300, ge=50)

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")

    # ---- UI ----------------------------------------------------------------------------
    api_url: str = Field(default="http://localhost:8000", description="Backend URL used by the UI.")
    ui_api_key: SecretStr | None = Field(default=None, description="Default API key in the UI.")

    # ------------------------------------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            text = v.strip()
            if text.startswith("["):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            return [o.strip() for o in text.split(",") if o.strip()]
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info: object) -> int:
        data = getattr(info, "data", {})
        size = data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    # ---- derived paths -----------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "contextiq.db"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    # ---- auth helpers ------------------------------------------------------------------
    def api_key_map(self) -> dict[str, str]:
        """Parse ``api_keys`` into ``{api_key: workspace_id}``. Invalid entries raise."""
        mapping: dict[str, str] = {}
        for raw in self.api_keys.split(","):
            item = raw.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"CONTEXTIQ_API_KEYS entry must be 'key:workspace': {item[:8]}...")
            key, workspace = item.rsplit(":", 1)
            key, workspace = key.strip(), workspace.strip()
            if len(key) < 8:
                raise ValueError("API keys must be at least 8 characters long")
            if not WORKSPACE_ID_PATTERN.match(workspace):
                raise ValueError(f"Invalid workspace id: {workspace!r}")
            mapping[key] = workspace
        return mapping

    # ---- LLM resolution ----------------------------------------------------------------
    def resolved_llm_provider(self) -> str:
        """Pick a provider. ``auto`` chooses the first provider with a key, else extractive."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        return "extractive"

    def resolved_llm_model(self) -> str | None:
        provider = self.resolved_llm_provider()
        if provider == "extractive":
            return None
        return self.llm_model or DEFAULT_MODELS.get(provider)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
