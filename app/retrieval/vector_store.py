"""FAISS-backed dense vector store: one exact inner-product index per workspace.

Workspace isolation is physical. Each workspace owns a ``faiss.IndexIDMap2(IndexFlatIP(d))``
persisted to ``{index_dir}/{workspace_id}.faiss``. Vectors are expected to be L2-normalised
so inner product equals cosine similarity. External ids are the SQLite ``vector_id`` values.

Filtered search passes ``faiss.SearchParameters(sel=IDSelectorBatch(...))`` straight to the
``IndexIDMap2``; FAISS translates the selector so it operates on *external* ids (verified
against faiss-cpu 1.15), so no Python-side over-fetching is needed.

Concurrency: a registry lock guards the workspace dictionaries and a per-workspace
``RLock`` serialises every mutation and search of that workspace's index. Lock order is
always workspace lock -> registry lock, never the reverse.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
import structlog

from app.core.config import WORKSPACE_ID_PATTERN

logger = structlog.get_logger(__name__)

INDEX_SUFFIX = ".faiss"


class IndexLoadError(Exception):
    """An index file on disk could not be loaded for this store's dimension."""


def _validate_workspace_id(workspace_id: str) -> str:
    """Reject ids that could escape ``index_dir`` or collide with temp files."""
    if not isinstance(workspace_id, str) or not WORKSPACE_ID_PATTERN.match(workspace_id):
        raise ValueError("invalid workspace_id")
    return workspace_id


class VectorStore:
    """Per-workspace FAISS indexes with atomic persistence and thread-safe access."""

    def __init__(self, index_dir: Path, dimension: int) -> None:
        """Create the store rooted at ``index_dir`` and load every existing ``*.faiss`` index."""
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._index_dir = Path(index_dir)
        self._dimension = int(dimension)
        self._indexes: dict[str, faiss.IndexIDMap2] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._registry_lock = threading.RLock()
        self._load_errors: dict[str, str] = {}
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def load_errors(self) -> dict[str, str]:
        """Index files found at start-up that could not be loaded: ``{workspace_id: reason}``.

        A non-empty mapping means those workspaces have documents that are listed but not
        searchable (typically after changing the embedding model); ``/health`` reports it and
        ``scripts/rebuild_index.py`` repairs it.
        """
        return dict(self._load_errors)

    # ---- public API ------------------------------------------------------------------

    def add(self, workspace_id: str, vector_ids: list[int], vectors: np.ndarray) -> None:
        """Add vectors under the given ids and persist the workspace index.

        Raises ``ValueError`` on shape/dtype problems, duplicate ids within the batch, or ids
        that already exist in the index (remove them first).
        """
        ids = self._coerce_ids(vector_ids)
        matrix = self._coerce_vectors(vectors)
        self._check_batch(ids, matrix)
        if ids.size == 0:
            return
        with self._lock_for(workspace_id):
            index = self._get_or_create(workspace_id)
            existing = faiss.vector_to_array(index.id_map)
            if existing.size and bool(np.isin(ids, existing).any()):
                raise ValueError("vector_ids already present in the index")
            index.add_with_ids(matrix, ids)
            self._save_locked(workspace_id, index)
        logger.info("vector_store.added", workspace_id=workspace_id, count=int(ids.size))

    def remove(self, workspace_id: str, vector_ids: list[int]) -> int:
        """Remove ids from the workspace index; persist and return how many were removed."""
        ids = np.unique(self._coerce_ids(vector_ids))
        if ids.size == 0:
            return 0
        with self._lock_for(workspace_id):
            index = self._get_index(workspace_id)
            if index is None:
                return 0
            removed = int(index.remove_ids(faiss.IDSelectorBatch(ids)))
            if removed:
                self._save_locked(workspace_id, index)
        logger.info("vector_store.removed", workspace_id=workspace_id, count=removed)
        return removed

    def search(
        self,
        workspace_id: str,
        query: np.ndarray,
        k: int,
        allowed_vector_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(vector_id, cosine)`` pairs, best first.

        Unknown or empty workspaces yield ``[]``. When ``allowed_vector_ids`` is given only
        those ids are candidates (an empty set matches nothing).
        """
        if allowed_vector_ids is not None and not allowed_vector_ids:
            return []
        vector = self._coerce_query(query)
        with self._lock_for(workspace_id):
            index = self._get_index(workspace_id)
            if index is None:
                return []
            k_eff = min(int(k), index.ntotal)
            if k_eff <= 0:
                return []
            if allowed_vector_ids is None:
                scores, labels = index.search(vector, k_eff)
            else:
                # Keep the selector referenced until the search returns (FAISS holds a raw pointer).
                selector = faiss.IDSelectorBatch(
                    np.fromiter(allowed_vector_ids, dtype=np.int64, count=len(allowed_vector_ids))
                )
                params = faiss.SearchParameters(sel=selector)
                scores, labels = index.search(vector, k_eff, params=params)
        return [
            (int(label), float(score))
            for label, score in zip(labels[0], scores[0], strict=True)
            if label != -1
        ]

    def count(self, workspace_id: str) -> int:
        index = self._get_index(workspace_id)
        return int(index.ntotal) if index is not None else 0

    def workspaces(self) -> list[str]:
        with self._registry_lock:
            return sorted(self._indexes)

    def drop_workspace(self, workspace_id: str) -> None:
        """Forget the workspace index and delete its file. No-op if it does not exist."""
        path = self._index_path(workspace_id)
        with self._lock_for(workspace_id):
            with self._registry_lock:
                self._indexes.pop(workspace_id, None)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
        logger.info("vector_store.workspace_dropped", workspace_id=workspace_id)

    def save(self, workspace_id: str) -> None:
        """Persist the workspace index atomically. Raises ``KeyError`` for unknown workspaces."""
        with self._lock_for(workspace_id):
            index = self._get_index(workspace_id)
            if index is None:
                raise KeyError(workspace_id)
            self._save_locked(workspace_id, index)

    def rebuild(self, workspace_id: str, vector_ids: list[int], vectors: np.ndarray) -> None:
        """Replace the workspace index with one built from scratch and persist it."""
        ids = self._coerce_ids(vector_ids)
        matrix = self._coerce_vectors(vectors)
        self._check_batch(ids, matrix)
        fresh = self._new_index()
        if ids.size:
            fresh.add_with_ids(matrix, ids)
        with self._lock_for(workspace_id):
            with self._registry_lock:
                self._indexes[workspace_id] = fresh
            self._save_locked(workspace_id, fresh)
        logger.info("vector_store.rebuilt", workspace_id=workspace_id, count=int(ids.size))

    # ---- internals -------------------------------------------------------------------

    def _new_index(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))

    def _index_path(self, workspace_id: str) -> Path:
        return self._index_dir / f"{_validate_workspace_id(workspace_id)}{INDEX_SUFFIX}"

    def _lock_for(self, workspace_id: str) -> threading.RLock:
        with self._registry_lock:
            lock = self._locks.get(workspace_id)
            if lock is None:
                lock = self._locks[workspace_id] = threading.RLock()
            return lock

    def _get_index(self, workspace_id: str) -> faiss.IndexIDMap2 | None:
        with self._registry_lock:
            return self._indexes.get(workspace_id)

    def _get_or_create(self, workspace_id: str) -> faiss.IndexIDMap2:
        """Return the workspace index, creating an empty one if needed (caller holds its lock)."""
        _validate_workspace_id(workspace_id)
        with self._registry_lock:
            index = self._indexes.get(workspace_id)
            if index is None:
                index = self._indexes[workspace_id] = self._new_index()
            return index

    def _save_locked(self, workspace_id: str, index: faiss.IndexIDMap2) -> None:
        """Write to a temp file in the same directory, then ``os.replace`` (caller holds lock)."""
        final_path = self._index_path(workspace_id)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._index_dir, prefix=f".{workspace_id}.", suffix=".tmp"
        )
        os.close(fd)
        try:
            faiss.write_index(index, tmp_path)
            os.replace(tmp_path, final_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _load_existing(self) -> None:
        """Load every ``*.faiss`` file whose name is a valid workspace id and dimension matches."""
        for path in sorted(self._index_dir.glob(f"*{INDEX_SUFFIX}")):
            workspace_id = path.name[: -len(INDEX_SUFFIX)]
            if not WORKSPACE_ID_PATTERN.match(workspace_id):
                logger.warning("vector_store.skip_invalid_filename", filename=path.name)
                self._load_errors[path.name] = "invalid workspace id in filename"
                continue
            try:
                index = self._read_index(path)
            except IndexLoadError as exc:
                self._load_errors[workspace_id] = str(exc)
                continue
            self._indexes[workspace_id] = index
            logger.info("vector_store.loaded", workspace_id=workspace_id, count=int(index.ntotal))

    def _read_index(self, path: Path) -> faiss.IndexIDMap2:
        """Read one index file; raise ``IndexLoadError`` (after logging) if it is unusable."""
        try:
            index = faiss.read_index(str(path))
        except (RuntimeError, OSError) as exc:
            logger.error(
                "vector_store.load_failed", filename=path.name, error_type=type(exc).__name__
            )
            raise IndexLoadError(f"unreadable index file ({type(exc).__name__})") from exc
        if not isinstance(index, faiss.IndexIDMap2):
            logger.error("vector_store.skip_unexpected_index_type", filename=path.name)
            raise IndexLoadError("unexpected index type")
        if index.d != self._dimension:
            logger.error(
                "vector_store.skip_dimension_mismatch",
                filename=path.name,
                expected=self._dimension,
                found=int(index.d),
            )
            raise IndexLoadError(
                f"dimension mismatch: index has {int(index.d)} dimensions, "
                f"embedding model produces {self._dimension}"
            )
        return index

    def _coerce_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Validate a ``(n, dimension)`` matrix and return it as C-contiguous float32."""
        try:
            matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("vectors must be a numeric array") from exc
        if matrix.ndim != 2:
            raise ValueError(f"vectors must be 2-D (n, {self._dimension}); got {matrix.shape}")
        if matrix.shape[1] != self._dimension:
            raise ValueError(
                f"vector dimension mismatch: expected {self._dimension}, got {matrix.shape[1]}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("vectors must not contain NaN or infinity")
        return matrix

    def _coerce_query(self, query: np.ndarray) -> np.ndarray:
        """Accept a ``(dimension,)`` or ``(1, dimension)`` query and return ``(1, dimension)``."""
        try:
            vector = np.ascontiguousarray(query, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("query must be a numeric array") from exc
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError(f"query must have shape ({self._dimension},); got {vector.shape}")
        return self._coerce_vectors(vector)

    @staticmethod
    def _coerce_ids(vector_ids: Iterable[int]) -> np.ndarray:
        """Return ids as a 1-D int64 array; ids must be non-negative (-1 is FAISS's sentinel)."""
        try:
            ids = np.asarray(list(vector_ids), dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("vector_ids must be integers") from exc
        if ids.ndim != 1:
            raise ValueError("vector_ids must be a flat list of integers")
        if ids.size and bool((ids < 0).any()):
            raise ValueError("vector_ids must be non-negative")
        return ids

    @staticmethod
    def _check_batch(ids: np.ndarray, matrix: np.ndarray) -> None:
        if ids.size != matrix.shape[0]:
            raise ValueError(
                f"got {ids.size} vector_ids for {matrix.shape[0]} vectors; counts must match"
            )
        if np.unique(ids).size != ids.size:
            raise ValueError("vector_ids contain duplicates")
