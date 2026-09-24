"""Index files that cannot be loaded are reported, and /health turns degraded instead of silent."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.config import Settings
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.vector_store import VectorStore

DIM = 64


def _write_index(path: Path, dimension: int, count: int = 3) -> None:
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    vectors = np.random.default_rng(0).random((count, dimension), dtype=np.float32)
    index.add_with_ids(vectors, np.arange(count, dtype=np.int64))
    faiss.write_index(index, str(path))


def test_dimension_mismatch_is_recorded_and_workspace_is_unsearchable(tmp_path: Path) -> None:
    _write_index(tmp_path / "alpha.faiss", 8)
    _write_index(tmp_path / "beta.faiss", DIM)

    store = VectorStore(tmp_path, DIM)

    assert set(store.load_errors) == {"alpha"}
    assert "dimension mismatch" in store.load_errors["alpha"]
    assert store.workspaces() == ["beta"]
    assert store.search("alpha", np.ones(DIM, dtype=np.float32), 3) == []
    assert store.count("beta") == 3


def test_corrupt_index_file_is_recorded(tmp_path: Path) -> None:
    (tmp_path / "gamma.faiss").write_bytes(b"definitely not a faiss index")

    store = VectorStore(tmp_path, DIM)

    assert "gamma" in store.load_errors
    assert store.workspaces() == []


def test_clean_start_has_no_load_errors(tmp_path: Path) -> None:
    assert VectorStore(tmp_path, DIM).load_errors == {}


def test_health_is_degraded_when_an_index_cannot_be_loaded(settings: Settings) -> None:
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    _write_index(settings.index_dir / "alpha.faiss", 8)
    app = create_app(settings, embedder=HashingEmbedder(DIM), llm=None)

    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert len(body["warnings"]) == 1
    assert "alpha" in body["warnings"][0]
    assert "rebuild_index" in body["warnings"][0]


def test_health_is_ok_with_no_warnings_by_default(settings: Settings) -> None:
    app = create_app(settings, embedder=HashingEmbedder(DIM), llm=None)

    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["warnings"] == []
