"""Embedding models behind a small ``Embedder`` protocol.

``SentenceTransformerEmbedder`` is the production implementation, a thin wrapper around
``langchain_huggingface.HuggingFaceEmbeddings``. ``HashingEmbedder`` is a deterministic,
dependency-free stand-in so the rest of the pipeline can be tested without loading a model.
Both return L2-normalised ``float32`` arrays, so cosine similarity is a plain inner product
and the FAISS ``IndexFlatIP`` index yields exact cosine scores.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

log = get_logger(__name__)

HASHING_MODEL_NAME = "hashing-test-embedder"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DIMENSION_PROBE = "dimension probe"
# Newest sentence-transformers name first; the older one is kept for compatibility.
_DIMENSION_GETTERS = ("get_embedding_dimension", "get_sentence_embedding_dimension")
_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


class Embedder(Protocol):
    """Anything that turns text into L2-normalised ``float32`` vectors."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages; returns shape ``(len(texts), dimension)``."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query; returns shape ``(dimension,)``."""
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return ``vectors`` (1-D or 2-D) as C-contiguous ``float32`` rows of unit length.

    Zero rows are left as zeros rather than turned into NaNs.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        return l2_normalize(array[np.newaxis, :])[0]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available torch device; other names pass through."""
    if device != "auto":
        return device
    import torch  # heavy import, only needed for auto-detection

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _hf_offline() -> bool:
    """True when the environment asks Hugging Face libraries to stay offline."""
    return any(
        os.environ.get(name, "").lower() in {"1", "true", "yes"} for name in _OFFLINE_ENV_VARS
    )


def _quiet_hf_progress_bars() -> None:
    """Keep tqdm download/load progress bars out of the structured logs."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with contextlib.suppress(Exception):
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    with contextlib.suppress(Exception):
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()


class SentenceTransformerEmbedder:
    """Sentence-Transformers model via ``langchain_huggingface.HuggingFaceEmbeddings``.

    The model is loaded eagerly so startup fails fast if it is unavailable; the embedding
    dimension is discovered lazily the first time it is needed. ``query_prefix`` is prepended
    to queries only (BGE-style instruction), never to passages.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
        cache_dir: Path | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self._dimension: int | None = None
        self._model = self._load(cache_dir)

    def _load(self, cache_dir: Path | None) -> HuggingFaceEmbeddings:
        _quiet_hf_progress_bars()
        from langchain_huggingface import HuggingFaceEmbeddings

        model_kwargs: dict[str, Any] = {"device": self.device}
        if _hf_offline():
            # Honour offline mode even if huggingface_hub was imported before the env was set.
            model_kwargs["local_files_only"] = True
        started = time.perf_counter()
        model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            cache_folder=str(cache_dir) if cache_dir else None,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True, "batch_size": self.batch_size},
        )
        log.info(
            "embedder.loaded",
            model_name=self.model_name,
            device=self.device,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )
        return model

    @property
    def dimension(self) -> int:
        """Embedding size, read from the model on first access."""
        if self._dimension is None:
            self._dimension = self._discover_dimension()
        return self._dimension

    def _discover_dimension(self) -> int:
        """Ask the underlying SentenceTransformer; fall back to embedding a probe string."""
        client = getattr(self._model, "_client", None) or getattr(self._model, "client", None)
        for getter_name in _DIMENSION_GETTERS:
            getter = getattr(client, getter_name, None)
            reported = getter() if callable(getter) else None
            if isinstance(reported, int) and reported > 0:
                return reported
        return int(self._embed_texts([_DIMENSION_PROBE]).shape[1])

    def _record_dimension(self, observed: int) -> None:
        if self._dimension is None:
            self._dimension = observed
        elif observed != self._dimension:
            raise RuntimeError(
                f"embedding dimension changed: expected {self._dimension}, got {observed}"
            )

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        raw = self._model.embed_documents(texts)
        vectors = l2_normalize(np.asarray(raw, dtype=np.float32).reshape(len(texts), -1))
        self._record_dimension(int(vectors.shape[1]))
        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages into an ``(n, dimension)`` float32 array; ``[]`` -> ``(0, dimension)``."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._embed_texts(list(texts))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query (``query_prefix`` prepended) into a ``(dimension,)`` float32 vector."""
        raw = self._model.embed_query(f"{self.query_prefix}{text}")
        vector = l2_normalize(np.asarray(raw, dtype=np.float32).reshape(-1))
        self._record_dimension(int(vector.shape[0]))
        return vector


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for tests: no model, no downloads.

    Tokens are lowercase alphanumeric runs. Each is hashed with BLAKE2b into one of
    ``dimension`` buckets with a hash-derived sign, the counts are accumulated and the vector
    is L2-normalised. Texts that share tokens therefore score higher than unrelated texts,
    which is enough to exercise retrieval end to end. Not suitable for real semantic search.
    """

    model_name: str = HASHING_MODEL_NAME

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.model_name = HASHING_MODEL_NAME
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages into an ``(n, dimension)`` float32 array; ``[]`` -> ``(0, dimension)``."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vectorize(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query into a ``(dimension,)`` float32 vector."""
        return self._vectorize(text)

    def _hash_token(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        bucket = int.from_bytes(digest[:8], "big") % self.dimension
        sign = -1.0 if digest[8] & 1 else 1.0
        return bucket, sign

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in _TOKEN_PATTERN.findall(text.lower()):
            bucket, sign = self._hash_token(token)
            vector[bucket] += sign
        if not vector.any():
            vector[0] = 1.0  # no tokens (or perfect cancellation): a fixed unit vector
            return vector
        return l2_normalize(vector)


def create_embedder(settings: Settings) -> Embedder:
    """Build the production embedder from application settings."""
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        query_prefix=settings.embedding_query_prefix,
        cache_dir=settings.embedding_cache_dir,
    )
