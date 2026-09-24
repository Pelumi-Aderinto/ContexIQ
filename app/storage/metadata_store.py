"""SQLite-backed metadata store for documents and chunks, with FTS5 (BM25) keyword search.

One ``sqlite3`` connection is shared across threads (``check_same_thread=False``) and every
operation is serialised through a re-entrant lock. Every chunk query filters by
``workspace_id`` as defence in depth: workspace isolation is a design invariant and nothing
here ever returns chunks from another workspace.

Storage conventions:

* ``created_at`` is stored as ISO-8601 text in UTC with microsecond precision so that
  lexicographic order equals chronological order.
* ``DocumentStatus`` is stored as its string value.
* ``chunks_fts`` is an FTS5 external-content table over ``chunks`` kept in sync by triggers.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from app.models.domain import Chunk
from app.models.schemas import DocumentInfo, DocumentStatus

logger = structlog.get_logger(__name__)

MEMORY_DB = ":memory:"

_ITER_BATCH_SIZE = 500
_MAX_QUERY_TOKENS = 64
# Runs of Unicode letters/digits (``\w`` minus underscore). Everything else is a separator.
_TOKEN_RE = re.compile(r"[^\W_]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT PRIMARY KEY,
    workspace_id TEXT    NOT NULL,
    filename     TEXT    NOT NULL,
    sha256       TEXT    NOT NULL,
    size_bytes   INTEGER NOT NULL,
    page_count   INTEGER NOT NULL DEFAULT 0,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL,
    error        TEXT,
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_workspace_sha256 ON documents(workspace_id, sha256);

CREATE TABLE IF NOT EXISTS chunks (
    vector_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id     TEXT    NOT NULL UNIQUE,
    document_id  TEXT    NOT NULL,
    workspace_id TEXT    NOT NULL,
    filename     TEXT    NOT NULL,
    page_number  INTEGER NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT    NOT NULL,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace_document ON chunks(workspace_id, document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, content='chunks', content_rowid='vector_id');

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.vector_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.vector_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.vector_id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.vector_id, new.text);
END;
"""

_DOCUMENT_COLUMNS = (
    "document_id, workspace_id, filename, sha256, size_bytes, page_count, chunk_count, "
    "status, error, created_at"
)
_CHUNK_COLUMNS = (
    "vector_id, chunk_id, document_id, workspace_id, filename, page_number, chunk_index, "
    "text, char_start, char_end"
)


def build_match_expression(query: str) -> str | None:
    """Turn free text into a safe FTS5 ``MATCH`` expression.

    The query is lower-cased and split on non-alphanumerics; each distinct token is quoted as
    a phrase and the phrases are joined with ``OR``. FTS5 operators, parentheses and quotes in
    the input therefore never reach the parser. Returns ``None`` when no token survives.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(query.lower()):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= _MAX_QUERY_TOKENS:
            break
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def _to_iso(value: datetime) -> str:
    """Serialise a datetime as UTC ISO-8601 text; naive values are assumed to be UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _from_iso(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _json_list(values: Sequence[Any]) -> str:
    """Encode a list for ``IN (SELECT value FROM json_each(?))`` (no SQL variable limit)."""
    return json.dumps(list(values))


def _row_to_document(row: sqlite3.Row) -> DocumentInfo:
    return DocumentInfo(
        document_id=row["document_id"],
        workspace_id=row["workspace_id"],
        filename=row["filename"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        status=DocumentStatus(row["status"]),
        error=row["error"],
        created_at=_from_iso(row["created_at"]),
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        workspace_id=row["workspace_id"],
        filename=row["filename"],
        page_number=row["page_number"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        char_start=row["char_start"],
        char_end=row["char_end"],
    )


class MetadataStore:
    """Thread-safe SQLite store for document metadata, chunk provenance and BM25 search."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the database at ``db_path`` and ensure the schema exists.

        ``Path(":memory:")`` opens a private in-memory database (handy for tests).
        """
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._closed = False
        is_memory = str(self._db_path) == MEMORY_DB
        if not is_memory:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure(is_memory=is_memory)
        self._conn.executescript(_SCHEMA)
        logger.info("metadata_store.opened", in_memory=is_memory)

    def _configure(self, *, is_memory: bool) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not is_memory:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True
        logger.info("metadata_store.closed")

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- documents -------------------------------------------------------------------

    def create_document(self, doc: DocumentInfo) -> None:
        """Insert a new document row. Raises ``ValueError`` if ``document_id`` already exists."""
        sql = f"INSERT INTO documents ({_DOCUMENT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        params = (
            doc.document_id,
            doc.workspace_id,
            doc.filename,
            doc.sha256,
            doc.size_bytes,
            doc.page_count,
            doc.chunk_count,
            doc.status.value,
            doc.error,
            _to_iso(doc.created_at),
        )
        try:
            with self._lock, self._conn:
                self._conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"document already exists: {doc.document_id}") from exc
        logger.debug(
            "metadata_store.document_created",
            document_id=doc.document_id,
            workspace_id=doc.workspace_id,
        )

    def get_document(self, workspace_id: str, document_id: str) -> DocumentInfo | None:
        sql = (
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE workspace_id = ? AND document_id = ?"
        )
        with self._lock:
            row = self._conn.execute(sql, (workspace_id, document_id)).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self, workspace_id: str) -> list[DocumentInfo]:
        """All documents in the workspace, newest first."""
        sql = (
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE workspace_id = ? "
            "ORDER BY created_at DESC, rowid DESC"
        )
        with self._lock:
            rows = self._conn.execute(sql, (workspace_id,)).fetchall()
        return [_row_to_document(row) for row in rows]

    def find_by_sha256(self, workspace_id: str, sha256: str) -> DocumentInfo | None:
        """Find an existing document with identical content within the workspace."""
        sql = (
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE workspace_id = ? AND sha256 = ? "
            "ORDER BY created_at ASC, rowid ASC LIMIT 1"
        )
        with self._lock:
            row = self._conn.execute(sql, (workspace_id, sha256)).fetchone()
        return _row_to_document(row) if row else None

    def update_document(
        self,
        workspace_id: str,
        document_id: str,
        *,
        status: DocumentStatus,
        error: str | None = None,
        chunk_count: int | None = None,
        page_count: int | None = None,
    ) -> None:
        """Update ``status`` and only those optional fields that were provided (not ``None``)."""
        assignments = ["status = ?"]
        params: list[Any] = [status.value]
        for column, value in (
            ("error", error),
            ("chunk_count", chunk_count),
            ("page_count", page_count),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        params.extend([workspace_id, document_id])
        sql = (
            f"UPDATE documents SET {', '.join(assignments)} "
            "WHERE workspace_id = ? AND document_id = ?"
        )
        with self._lock, self._conn:
            updated = self._conn.execute(sql, params).rowcount
        if updated == 0:
            logger.warning(
                "metadata_store.update_missing_document",
                document_id=document_id,
                workspace_id=workspace_id,
            )

    def delete_document(self, workspace_id: str, document_id: str) -> list[int]:
        """Delete a document and all of its chunks; return the removed ``vector_id`` values.

        Returns ``[]`` (and changes nothing) when the document is not in the workspace.
        The caller is responsible for removing the returned ids from the vector index.
        """
        with self._lock, self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM documents WHERE workspace_id = ? AND document_id = ?",
                (workspace_id, document_id),
            ).fetchone()
            if exists is None:
                return []
            rows = self._conn.execute(
                "SELECT vector_id FROM chunks WHERE workspace_id = ? AND document_id = ? "
                "ORDER BY vector_id",
                (workspace_id, document_id),
            ).fetchall()
            self._conn.execute(
                "DELETE FROM chunks WHERE workspace_id = ? AND document_id = ?",
                (workspace_id, document_id),
            )
            self._conn.execute(
                "DELETE FROM documents WHERE workspace_id = ? AND document_id = ?",
                (workspace_id, document_id),
            )
        vector_ids = [int(row["vector_id"]) for row in rows]
        logger.info(
            "metadata_store.document_deleted",
            document_id=document_id,
            workspace_id=workspace_id,
            chunks_removed=len(vector_ids),
        )
        return vector_ids

    # ---- chunks ----------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> list[int]:
        """Insert chunks in one transaction; return their new ``vector_id`` values in input order.

        Raises ``ValueError`` if any ``chunk_id`` already exists (the transaction is rolled back).
        """
        if not chunks:
            return []
        sql = (
            "INSERT INTO chunks (chunk_id, document_id, workspace_id, filename, page_number, "
            "chunk_index, text, char_start, char_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        vector_ids: list[int] = []
        try:
            with self._lock, self._conn:
                for chunk in chunks:
                    cursor = self._conn.execute(
                        sql,
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.workspace_id,
                            chunk.filename,
                            chunk.page_number,
                            chunk.chunk_index,
                            chunk.text,
                            chunk.char_start,
                            chunk.char_end,
                        ),
                    )
                    if cursor.lastrowid is None:  # pragma: no cover - defensive
                        raise RuntimeError("SQLite did not return a rowid for inserted chunk")
                    vector_ids.append(int(cursor.lastrowid))
        except sqlite3.IntegrityError as exc:
            raise ValueError("one or more chunk_ids already exist") from exc
        logger.debug("metadata_store.chunks_added", count=len(vector_ids))
        return vector_ids

    def get_chunks(self, workspace_id: str, vector_ids: list[int]) -> dict[int, Chunk]:
        """Fetch chunks by ``vector_id`` within the workspace. Unknown ids are simply absent."""
        if not vector_ids:
            return {}
        sql = (
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE workspace_id = ? "
            "AND vector_id IN (SELECT value FROM json_each(?))"
        )
        with self._lock:
            rows = self._conn.execute(sql, (workspace_id, _json_list(vector_ids))).fetchall()
        return {int(row["vector_id"]): _row_to_chunk(row) for row in rows}

    def get_chunk_by_id(self, workspace_id: str, chunk_id: str) -> Chunk | None:
        sql = f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE workspace_id = ? AND chunk_id = ?"
        with self._lock:
            row = self._conn.execute(sql, (workspace_id, chunk_id)).fetchone()
        return _row_to_chunk(row) if row else None

    def list_vector_ids(
        self, workspace_id: str, document_ids: list[str] | None = None
    ) -> list[int]:
        """Ascending ``vector_id`` values in the workspace, optionally restricted to documents.

        ``document_ids=None`` means no restriction; an empty list matches nothing.
        """
        sql = "SELECT vector_id FROM chunks WHERE workspace_id = ?"
        params: list[Any] = [workspace_id]
        if document_ids is not None:
            if not document_ids:
                return []
            sql += " AND document_id IN (SELECT value FROM json_each(?))"
            params.append(_json_list(document_ids))
        sql += " ORDER BY vector_id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [int(row["vector_id"]) for row in rows]

    def keyword_search(
        self,
        workspace_id: str,
        query: str,
        k: int,
        document_ids: list[str] | None = None,
    ) -> list[tuple[int, float]]:
        """BM25 keyword search over the workspace's chunks.

        Returns ``(vector_id, score)`` pairs, best first, where ``score = -bm25`` so that
        higher is better. The query is sanitised into an OR-of-phrases expression, so odd or
        hostile input never raises: it yields ``[]`` instead.
        """
        match = build_match_expression(query)
        if match is None or k <= 0 or (document_ids is not None and not document_ids):
            return []
        sql = (
            "SELECT c.vector_id AS vector_id, bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks AS c ON c.vector_id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND c.workspace_id = ?"
        )
        params: list[Any] = [match, workspace_id]
        if document_ids is not None:
            sql += " AND c.document_id IN (SELECT value FROM json_each(?))"
            params.append(_json_list(document_ids))
        sql += " ORDER BY rank LIMIT ?"
        params.append(k)
        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("metadata_store.keyword_search_failed", error_type=type(exc).__name__)
            return []
        return [(int(row["vector_id"]), -float(row["rank"])) for row in rows]

    def iter_chunks(self, workspace_id: str) -> Iterator[tuple[int, Chunk]]:
        """Yield ``(vector_id, chunk)`` for every chunk in the workspace, ascending by id.

        Rows are fetched in batches (keyset pagination) so the lock is not held while the
        consumer processes a batch, e.g. during a full index rebuild.
        """
        sql = (
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE workspace_id = ? AND vector_id > ? "
            "ORDER BY vector_id LIMIT ?"
        )
        last_id = -1
        while True:
            with self._lock:
                rows = self._conn.execute(sql, (workspace_id, last_id, _ITER_BATCH_SIZE)).fetchall()
            if not rows:
                return
            for row in rows:
                last_id = int(row["vector_id"])
                yield last_id, _row_to_chunk(row)

    def count_chunks(self, workspace_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE workspace_id = ?", (workspace_id,)
            ).fetchone()
        return int(row["n"])

    def list_workspaces(self) -> list[str]:
        """Sorted workspace ids that own at least one document or chunk."""
        sql = (
            "SELECT workspace_id FROM documents UNION SELECT workspace_id FROM chunks "
            "ORDER BY workspace_id"
        )
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [row["workspace_id"] for row in rows]

    # ---- diagnostics -----------------------------------------------------------------

    def _check_fts_integrity(self) -> bool:
        """Verify the FTS index matches the ``chunks`` table (used by tests and diagnostics)."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO chunks_fts(chunks_fts, rank) VALUES ('integrity-check', 1)"
                )
        except sqlite3.DatabaseError as exc:
            logger.error("metadata_store.fts_integrity_failed", error_type=type(exc).__name__)
            return False
        return True
