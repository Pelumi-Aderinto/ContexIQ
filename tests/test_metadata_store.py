"""Tests for the SQLite metadata store (documents, chunks, FTS5 keyword search)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from app.models.domain import Chunk
from app.models.schemas import DocumentInfo, DocumentStatus
from app.storage.metadata_store import MetadataStore, build_match_expression

WS_A = "alpha"
WS_B = "beta"
BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def make_doc(
    workspace_id: str = WS_A,
    document_id: str = "doc1",
    *,
    sha256: str = "a" * 64,
    created_at: datetime = BASE_TIME,
    status: DocumentStatus = DocumentStatus.PROCESSING,
) -> DocumentInfo:
    return DocumentInfo(
        document_id=document_id,
        workspace_id=workspace_id,
        filename=f"{document_id}.pdf",
        sha256=sha256,
        size_bytes=1234,
        page_count=0,
        chunk_count=0,
        status=status,
        created_at=created_at,
    )


def make_chunks(
    texts: list[str], *, workspace_id: str = WS_A, document_id: str = "doc1"
) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=Chunk.make_id(document_id, 1, i),
            document_id=document_id,
            workspace_id=workspace_id,
            filename=f"{document_id}.pdf",
            page_number=1,
            chunk_index=i,
            text=text,
            char_start=i * 10,
            char_end=i * 10 + len(text),
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def store(tmp_path: Path):
    s = MetadataStore(tmp_path / "meta.db")
    yield s
    s.close()


# ---- schema ----------------------------------------------------------------------------


def test_schema_created_and_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "meta.db"
    first = MetadataStore(db_path)
    first.close()
    second = MetadataStore(db_path)  # re-opening must not fail on existing objects
    try:
        conn = sqlite3.connect(db_path)
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
            )
        }
        conn.close()
    finally:
        second.close()
    assert {"documents", "chunks", "chunks_fts"} <= names
    assert {"chunks_fts_ai", "chunks_fts_ad", "chunks_fts_au"} <= names


def test_memory_database_works() -> None:
    store = MetadataStore(Path(":memory:"))
    try:
        store.create_document(make_doc())
        assert store.get_document(WS_A, "doc1") is not None
    finally:
        store.close()
        store.close()  # idempotent


# ---- documents -------------------------------------------------------------------------


def test_document_crud_round_trip(store: MetadataStore) -> None:
    doc = make_doc()
    store.create_document(doc)

    fetched = store.get_document(WS_A, "doc1")
    assert fetched == doc
    assert fetched is not None and fetched.created_at.tzinfo is not None

    store.update_document(WS_A, "doc1", status=DocumentStatus.INDEXED, chunk_count=7, page_count=3)
    updated = store.get_document(WS_A, "doc1")
    assert updated is not None
    assert updated.status is DocumentStatus.INDEXED
    assert updated.chunk_count == 7
    assert updated.page_count == 3
    assert updated.error is None

    # Only provided fields are touched.
    store.update_document(WS_A, "doc1", status=DocumentStatus.FAILED, error="boom")
    failed = store.get_document(WS_A, "doc1")
    assert failed is not None
    assert failed.status is DocumentStatus.FAILED
    assert failed.error == "boom"
    assert failed.chunk_count == 7
    assert failed.page_count == 3

    assert store.delete_document(WS_A, "doc1") == []
    assert store.get_document(WS_A, "doc1") is None


def test_document_is_scoped_to_workspace(store: MetadataStore) -> None:
    store.create_document(make_doc(WS_A))
    assert store.get_document(WS_B, "doc1") is None
    assert store.list_documents(WS_B) == []
    store.update_document(WS_B, "doc1", status=DocumentStatus.FAILED, error="x")
    doc = store.get_document(WS_A, "doc1")
    assert doc is not None and doc.status is DocumentStatus.PROCESSING
    assert store.delete_document(WS_B, "doc1") == []
    assert store.get_document(WS_A, "doc1") is not None


def test_create_duplicate_document_id_raises(store: MetadataStore) -> None:
    store.create_document(make_doc())
    with pytest.raises(ValueError):
        store.create_document(make_doc())


def test_naive_created_at_is_treated_as_utc(store: MetadataStore) -> None:
    store.create_document(make_doc(created_at=datetime(2026, 1, 1, 12, 0, 0)))
    doc = store.get_document(WS_A, "doc1")
    assert doc is not None
    assert doc.created_at == BASE_TIME


def test_find_by_sha256_is_per_workspace(store: MetadataStore) -> None:
    sha = "f" * 64
    store.create_document(make_doc(WS_A, "doc1", sha256=sha))
    found = store.find_by_sha256(WS_A, sha)
    assert found is not None and found.document_id == "doc1"
    assert store.find_by_sha256(WS_B, sha) is None
    assert store.find_by_sha256(WS_A, "0" * 64) is None


def test_list_documents_newest_first(store: MetadataStore) -> None:
    for i in range(3):
        store.create_document(
            make_doc(
                WS_A, f"doc{i}", sha256=f"{i}" * 64, created_at=BASE_TIME + timedelta(minutes=i)
            )
        )
    store.create_document(make_doc(WS_B, "other", sha256="b" * 64))
    listed = store.list_documents(WS_A)
    assert [d.document_id for d in listed] == ["doc2", "doc1", "doc0"]


# ---- chunks ----------------------------------------------------------------------------


def test_add_chunks_returns_increasing_ids_and_round_trips(store: MetadataStore) -> None:
    store.create_document(make_doc())
    chunks = make_chunks(["first chunk", "second chunk", "third chunk"])

    ids = store.add_chunks(chunks)
    assert len(ids) == 3
    assert ids == sorted(ids) and len(set(ids)) == 3

    fetched = store.get_chunks(WS_A, ids)
    assert set(fetched) == set(ids)
    for vector_id, chunk in zip(ids, chunks, strict=True):
        assert fetched[vector_id] == chunk

    assert store.get_chunk_by_id(WS_A, chunks[1].chunk_id) == chunks[1]
    assert store.count_chunks(WS_A) == 3
    assert store.list_vector_ids(WS_A) == ids
    assert [vid for vid, _ in store.iter_chunks(WS_A)] == ids
    assert store.add_chunks([]) == []


def test_add_chunks_continue_incrementing_across_batches(store: MetadataStore) -> None:
    first = store.add_chunks(make_chunks(["a", "b"], document_id="d1"))
    second = store.add_chunks(make_chunks(["c"], document_id="d2"))
    assert second[0] > max(first)


def test_duplicate_chunk_id_rolls_back_whole_batch(store: MetadataStore) -> None:
    chunks = make_chunks(["a", "b"])
    store.add_chunks(chunks)
    with pytest.raises(ValueError):
        store.add_chunks([*make_chunks(["c"], document_id="d2"), chunks[0]])
    assert store.count_chunks(WS_A) == 2  # the partial batch was rolled back


def test_chunks_invisible_from_other_workspace(store: MetadataStore) -> None:
    ids = store.add_chunks(make_chunks(["secret text", "more secret"]))
    assert store.get_chunks(WS_B, ids) == {}
    assert store.get_chunk_by_id(WS_B, Chunk.make_id("doc1", 1, 0)) is None
    assert store.list_vector_ids(WS_B) == []
    assert store.count_chunks(WS_B) == 0
    assert list(store.iter_chunks(WS_B)) == []
    assert store.keyword_search(WS_B, "secret", k=5) == []


def test_list_vector_ids_filters_by_document(store: MetadataStore) -> None:
    ids1 = store.add_chunks(make_chunks(["a", "b"], document_id="d1"))
    ids2 = store.add_chunks(make_chunks(["c"], document_id="d2"))
    assert store.list_vector_ids(WS_A, ["d1"]) == ids1
    assert store.list_vector_ids(WS_A, ["d2", "unknown"]) == ids2
    assert store.list_vector_ids(WS_A, []) == []
    assert store.list_vector_ids(WS_A) == ids1 + ids2


def test_list_workspaces(store: MetadataStore) -> None:
    assert store.list_workspaces() == []
    store.create_document(make_doc(WS_B, "docb", sha256="b" * 64))
    store.add_chunks(make_chunks(["x"], workspace_id=WS_A))
    assert store.list_workspaces() == [WS_A, WS_B]


# ---- keyword search --------------------------------------------------------------------


def test_build_match_expression_sanitizes() -> None:
    assert build_match_expression("Hello, World!") == '"hello" OR "world"'
    assert build_match_expression('"; DROP TABLE chunks; --') == '"drop" OR "table" OR "chunks"'
    assert build_match_expression("NEAR(foo bar) ^baz*") == '"near" OR "foo" OR "bar" OR "baz"'
    assert build_match_expression("repeat repeat REPEAT") == '"repeat"'
    assert build_match_expression("") is None
    assert build_match_expression("   ---  ") is None
    assert build_match_expression("snake_case") == '"snake" OR "case"'


def test_keyword_search_finds_rare_term_first(store: MetadataStore) -> None:
    ids = store.add_chunks(
        make_chunks(
            [
                "the quick brown fox jumps over the lazy dog",
                "a totally unrelated sentence about zyxwvut widgets",
                "another common sentence about dogs and foxes",
            ]
        )
    )
    results = store.keyword_search(WS_A, "zyxwvut", k=5)
    assert results and results[0][0] == ids[1]
    assert results[0][1] > 0  # score is -bm25, higher is better
    assert all(a[1] >= b[1] for a, b in pairwise(results))


def test_keyword_search_respects_k_and_document_ids(store: MetadataStore) -> None:
    ids1 = store.add_chunks(make_chunks(["gamma ray burst", "gamma decay"], document_id="d1"))
    ids2 = store.add_chunks(make_chunks(["gamma function"], document_id="d2"))

    all_hits = {vid for vid, _ in store.keyword_search(WS_A, "gamma", k=10)}
    assert all_hits == set(ids1) | set(ids2)
    assert len(store.keyword_search(WS_A, "gamma", k=1)) == 1

    only_d2 = store.keyword_search(WS_A, "gamma", k=10, document_ids=["d2"])
    assert [vid for vid, _ in only_d2] == ids2
    assert store.keyword_search(WS_A, "gamma", k=10, document_ids=[]) == []
    assert store.keyword_search(WS_A, "gamma", k=10, document_ids=["missing"]) == []
    assert store.keyword_search(WS_A, "gamma", k=0) == []


@pytest.mark.parametrize(
    "query",
    [
        '"; DROP TABLE chunks; --',
        "",
        "   ",
        "日本語 テキスト",
        "NEAR(foo bar)",
        "((((",
        '"unbalanced',
        "foo* OR ^bar NOT baz",
        "émigré café",
        "\x00\x01control",
    ],
)
def test_keyword_search_never_raises_on_odd_input(store: MetadataStore, query: str) -> None:
    store.add_chunks(make_chunks(["some indexed text here"]))
    result = store.keyword_search(WS_A, query, k=5)
    assert isinstance(result, list)
    assert store.count_chunks(WS_A) == 1  # tables are intact


def test_keyword_search_unicode_term(store: MetadataStore) -> None:
    ids = store.add_chunks(make_chunks(["plain english", "résumé of émigré café culture"]))
    hits = store.keyword_search(WS_A, "café", k=5)
    assert [vid for vid, _ in hits] == [ids[1]]


# ---- deletion / FTS sync ---------------------------------------------------------------


def test_delete_document_cascades_and_returns_vector_ids(store: MetadataStore) -> None:
    store.create_document(make_doc(WS_A, "d1", sha256="1" * 64))
    store.create_document(make_doc(WS_A, "d2", sha256="2" * 64))
    ids1 = store.add_chunks(make_chunks(["unique quokka text", "more quokka"], document_id="d1"))
    ids2 = store.add_chunks(make_chunks(["wombat text"], document_id="d2"))

    removed = store.delete_document(WS_A, "d1")
    assert sorted(removed) == sorted(ids1)
    assert store.get_document(WS_A, "d1") is None
    assert store.get_chunks(WS_A, ids1) == {}
    assert store.count_chunks(WS_A) == 1
    assert store.list_vector_ids(WS_A) == ids2
    assert store.delete_document(WS_A, "d1") == []
    assert store.delete_document(WS_A, "nope") == []


def test_fts_stays_in_sync_after_delete(store: MetadataStore) -> None:
    store.create_document(make_doc(WS_A, "d1", sha256="1" * 64))
    store.create_document(make_doc(WS_A, "d2", sha256="2" * 64))
    store.add_chunks(make_chunks(["unique quokka text"], document_id="d1"))
    ids2 = store.add_chunks(make_chunks(["quokka elsewhere"], document_id="d2"))
    assert len(store.keyword_search(WS_A, "quokka", k=10)) == 2

    store.delete_document(WS_A, "d1")
    assert [vid for vid, _ in store.keyword_search(WS_A, "quokka", k=10)] == ids2
    assert store._check_fts_integrity()  # FTS index matches the content table


def test_operations_after_close_raise(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "meta.db")
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.count_chunks(WS_A)
