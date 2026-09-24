"""Tests for the per-workspace FAISS vector store."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import faiss
import numpy as np
import pytest

from app.retrieval.vector_store import VectorStore

DIM = 16
WS = "alpha"


def unit_vectors(n: int, seed: int = 0, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "indexes", DIM)


def test_search_identical_vector_has_cosine_one(store: VectorStore) -> None:
    vecs = unit_vectors(5)
    store.add(WS, [10, 11, 12, 13, 14], vecs)

    results = store.search(WS, vecs[2], k=3)
    assert len(results) == 3
    assert results[0][0] == 12
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert all(a[1] >= b[1] for a, b in pairwise(results))
    assert store.count(WS) == 5
    assert store.workspaces() == [WS]


def test_search_accepts_row_vector_and_clamps_k(store: VectorStore) -> None:
    vecs = unit_vectors(3)
    store.add(WS, [1, 2, 3], vecs)
    results = store.search(WS, vecs[0:1], k=50)
    assert len(results) == 3
    assert results[0][0] == 1
    assert store.search(WS, vecs[0], k=0) == []


def test_unknown_workspace_returns_empty(store: VectorStore) -> None:
    assert store.search("nowhere", unit_vectors(1)[0], k=5) == []
    assert store.count("nowhere") == 0
    assert store.remove("nowhere", [1, 2]) == 0
    assert store.workspaces() == []


def test_persistence_across_instances(tmp_path: Path) -> None:
    index_dir = tmp_path / "indexes"
    vecs = unit_vectors(4)
    first = VectorStore(index_dir, DIM)
    first.add(WS, [1, 2, 3, 4], vecs)
    first.add("beta", [7], vecs[:1])
    assert (index_dir / f"{WS}.faiss").exists()
    assert not list(index_dir.glob("*.tmp"))

    second = VectorStore(index_dir, DIM)
    assert second.workspaces() == [WS, "beta"]
    assert second.count(WS) == 4
    results = second.search(WS, vecs[3], k=1)
    assert results[0][0] == 4
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_remove_reduces_count_and_hides_vector(store: VectorStore) -> None:
    vecs = unit_vectors(4)
    store.add(WS, [1, 2, 3, 4], vecs)

    removed = store.remove(WS, [2, 999])
    assert removed == 1
    assert store.count(WS) == 3
    ids = {vid for vid, _ in store.search(WS, vecs[1], k=10)}
    assert 2 not in ids and ids == {1, 3, 4}
    assert store.remove(WS, []) == 0

    reloaded = VectorStore(store._index_dir, DIM)  # removal was persisted
    assert reloaded.count(WS) == 3


def test_allowed_vector_ids_filter(store: VectorStore) -> None:
    vecs = unit_vectors(6)
    store.add(WS, [1, 2, 3, 4, 5, 6], vecs)

    results = store.search(WS, vecs[0], k=10, allowed_vector_ids={2, 5})
    assert {vid for vid, _ in results} == {2, 5}
    assert store.search(WS, vecs[0], k=10, allowed_vector_ids=set()) == []
    assert store.search(WS, vecs[0], k=10, allowed_vector_ids={999}) == []

    # An allowed id that matches exactly still scores ~1.0 and ranks first.
    results = store.search(WS, vecs[3], k=2, allowed_vector_ids={4, 1})
    assert results[0][0] == 4
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dimension_mismatch_raises(store: VectorStore) -> None:
    wrong = unit_vectors(2, dim=DIM + 1)
    with pytest.raises(ValueError, match="dimension"):
        store.add(WS, [1, 2], wrong)
    with pytest.raises(ValueError, match="dimension"):
        store.rebuild(WS, [1, 2], wrong)
    store.add(WS, [1], unit_vectors(1))
    with pytest.raises(ValueError, match="dimension"):
        store.search(WS, wrong[0], k=1)


def test_invalid_inputs_raise(store: VectorStore) -> None:
    vecs = unit_vectors(2)
    with pytest.raises(ValueError, match="counts must match"):
        store.add(WS, [1], vecs)
    with pytest.raises(ValueError, match="duplicates"):
        store.add(WS, [1, 1], vecs)
    with pytest.raises(ValueError, match="non-negative"):
        store.add(WS, [-1, 2], vecs)
    with pytest.raises(ValueError, match="2-D"):
        store.add(WS, [1], vecs[0])
    with pytest.raises(ValueError, match="NaN"):
        store.add(WS, [1, 2], np.full((2, DIM), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="workspace_id"):
        store.add("../escape", [1, 2], vecs)
    with pytest.raises(ValueError):
        VectorStore(store._index_dir, 0)
    assert store.count(WS) == 0


def test_adding_existing_id_raises(store: VectorStore) -> None:
    vecs = unit_vectors(3)
    store.add(WS, [1, 2], vecs[:2])
    with pytest.raises(ValueError, match="already present"):
        store.add(WS, [2, 3], vecs[1:])
    assert store.count(WS) == 2  # nothing from the rejected batch was added


def test_add_converts_dtype_and_layout(store: VectorStore) -> None:
    vecs = unit_vectors(3).astype(np.float64)
    non_contiguous = np.asfortranarray(vecs)
    store.add(WS, [1, 2, 3], non_contiguous)
    results = store.search(WS, vecs[1], k=1)
    assert results[0][0] == 2
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_workspaces_are_isolated(store: VectorStore) -> None:
    vecs = unit_vectors(2)
    store.add("alpha", [1], vecs[:1])
    store.add("beta", [2], vecs[1:])
    assert [vid for vid, _ in store.search("alpha", vecs[1], k=5)] == [1]
    assert [vid for vid, _ in store.search("beta", vecs[0], k=5)] == [2]


def test_drop_workspace_removes_file_and_index(store: VectorStore) -> None:
    store.add(WS, [1], unit_vectors(1))
    path = store._index_dir / f"{WS}.faiss"
    assert path.exists()
    store.drop_workspace(WS)
    assert not path.exists()
    assert store.workspaces() == []
    assert store.count(WS) == 0
    store.drop_workspace(WS)  # idempotent


def test_rebuild_replaces_index(store: VectorStore) -> None:
    old = unit_vectors(3, seed=1)
    store.add(WS, [1, 2, 3], old)
    new = unit_vectors(2, seed=2)
    store.rebuild(WS, [10, 11], new)
    assert store.count(WS) == 2
    assert {vid for vid, _ in store.search(WS, new[0], k=5)} == {10, 11}

    reloaded = VectorStore(store._index_dir, DIM)
    assert reloaded.count(WS) == 2
    store.rebuild(WS, [], np.empty((0, DIM), dtype=np.float32))
    assert store.count(WS) == 0


def test_save_unknown_workspace_raises(store: VectorStore) -> None:
    with pytest.raises(KeyError):
        store.save("ghost")


def test_load_skips_dimension_mismatch_and_corrupt_files(tmp_path: Path) -> None:
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    other = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM + 4))
    other.add_with_ids(unit_vectors(1, dim=DIM + 4), np.array([1], dtype=np.int64))
    faiss.write_index(other, str(index_dir / "wrongdim.faiss"))
    (index_dir / "corrupt.faiss").write_bytes(b"not a faiss index")
    (index_dir / "Bad Name.faiss").write_bytes(b"ignored")
    good = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
    good.add_with_ids(unit_vectors(2), np.array([5, 6], dtype=np.int64))
    faiss.write_index(good, str(index_dir / "good.faiss"))

    store = VectorStore(index_dir, DIM)
    assert store.workspaces() == ["good"]
    assert store.count("good") == 2


def test_concurrent_adds_do_not_corrupt(store: VectorStore) -> None:
    threads = 8
    per_thread = 25
    vecs = unit_vectors(threads * per_thread)
    barrier = threading.Barrier(threads)

    def worker(t: int) -> None:
        barrier.wait()
        start = t * per_thread
        ids = list(range(start, start + per_thread))
        store.add(WS, ids, vecs[start : start + per_thread])

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(worker, range(threads)))

    assert store.count(WS) == threads * per_thread
    probe = vecs[threads * per_thread - 1]
    assert store.search(WS, probe, k=1)[0][0] == threads * per_thread - 1
    reloaded = VectorStore(store._index_dir, DIM)
    assert reloaded.count(WS) == threads * per_thread
    assert not list(store._index_dir.glob("*.tmp"))
