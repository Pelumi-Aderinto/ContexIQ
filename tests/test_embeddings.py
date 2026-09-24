"""Tests for ``app.retrieval.embeddings``."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.core.config import Settings
from app.retrieval import embeddings
from app.retrieval.embeddings import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    l2_normalize,
    resolve_device,
)

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---- HashingEmbedder ------------------------------------------------------------------------


def test_hashing_is_deterministic_across_instances() -> None:
    texts = ["The quick brown fox", "jumps over the lazy dog"]
    first = HashingEmbedder().embed_documents(texts)
    second = HashingEmbedder().embed_documents(texts)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        HashingEmbedder().embed_query("fox"), HashingEmbedder().embed_query("fox")
    )


def test_hashing_shapes_dtype_and_metadata() -> None:
    embedder = HashingEmbedder()
    docs = embedder.embed_documents(["a", "b", "c"])
    assert docs.shape == (3, 64)
    assert docs.dtype == np.float32
    assert docs.flags["C_CONTIGUOUS"]
    query = embedder.embed_query("a")
    assert query.shape == (64,)
    assert query.dtype == np.float32
    assert embedder.dimension == 64
    assert embedder.model_name == "hashing-test-embedder"


def test_hashing_vectors_are_normalized() -> None:
    embedder = HashingEmbedder()
    docs = embedder.embed_documents(["one two three", "four", "five five five five"])
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-6)
    assert abs(np.linalg.norm(embedder.embed_query("six seven")) - 1.0) < 1e-6


def test_hashing_empty_input_returns_zero_rows() -> None:
    empty = HashingEmbedder().embed_documents([])
    assert empty.shape == (0, 64)
    assert empty.dtype == np.float32


def test_hashing_similar_texts_score_higher_than_dissimilar() -> None:
    embedder = HashingEmbedder()
    query = embedder.embed_query("how do I reset my password")
    related, unrelated = embedder.embed_documents(
        [
            "To reset your password open the account settings page.",
            "Quarterly revenue grew twelve percent year over year.",
        ]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_hashing_is_case_and_punctuation_insensitive() -> None:
    embedder = HashingEmbedder()
    np.testing.assert_allclose(
        embedder.embed_query("Hello, World!"), embedder.embed_query("hello world"), atol=1e-7
    )


def test_hashing_text_without_tokens_gives_fixed_unit_vector() -> None:
    vector = HashingEmbedder().embed_query("!!! ??? ...")
    assert vector[0] == 1.0
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-6
    np.testing.assert_array_equal(vector, HashingEmbedder().embed_query(""))


def test_hashing_custom_dimension_and_validation() -> None:
    assert HashingEmbedder(dimension=16).embed_documents(["x"]).shape == (1, 16)
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=0)


def test_hashing_satisfies_embedder_protocol_shape() -> None:
    embedder: embeddings.Embedder = HashingEmbedder()
    assert isinstance(embedder.model_name, str)
    assert isinstance(embedder.dimension, int)


# ---- helpers --------------------------------------------------------------------------------


def test_l2_normalize_rows_and_zero_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_array_equal(out[1], [0.0, 0.0])
    assert out.dtype == np.float32
    assert not np.isnan(out).any()


def test_l2_normalize_one_dimensional() -> None:
    out = l2_normalize(np.array([0.0, 2.0, 0.0]))
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0])


def test_resolve_device_passes_explicit_names_through() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"), (True, True, "cuda")],
)
def test_resolve_device_auto(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("auto") == expected


def test_create_embedder_passes_settings_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeEmbedder:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            captured["model_name"] = model_name
            captured.update(kwargs)

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", FakeEmbedder)
    settings = Settings(
        _env_file=None,
        embedding_model="some/model",
        embedding_device="auto",
        embedding_batch_size=7,
        embedding_query_prefix="q: ",
        embedding_cache_dir="/tmp/hf-cache",
    )
    create_embedder(settings)
    assert captured == {
        "model_name": "some/model",
        "device": "auto",
        "batch_size": 7,
        "query_prefix": "q: ",
        "cache_dir": settings.embedding_cache_dir,
    }


def test_sentence_transformer_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder(BGE_MODEL, batch_size=0)


# ---- real model (slow) ----------------------------------------------------------------------


@pytest.mark.slow
class TestSentenceTransformerEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self) -> Iterator[SentenceTransformerEmbedder]:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("HF_HUB_OFFLINE", "1")  # load from the local cache only, never the network
            yield SentenceTransformerEmbedder(
                BGE_MODEL, device="cpu", batch_size=8, query_prefix=BGE_QUERY_PREFIX
            )

    def test_dimension_is_384(self, embedder: SentenceTransformerEmbedder) -> None:
        assert embedder.dimension == 384
        assert embedder.model_name == BGE_MODEL
        assert embedder.device == "cpu"

    def test_documents_are_normalized_float32(self, embedder: SentenceTransformerEmbedder) -> None:
        vectors = embedder.embed_documents(["first passage", "second passage", "third"])
        assert vectors.shape == (3, 384)
        assert vectors.dtype == np.float32
        assert vectors.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_empty_documents(self, embedder: SentenceTransformerEmbedder) -> None:
        assert embedder.embed_documents([]).shape == (0, 384)

    def test_query_is_normalized_vector(self, embedder: SentenceTransformerEmbedder) -> None:
        vector = embedder.embed_query("what is the refund policy?")
        assert vector.shape == (384,)
        assert vector.dtype == np.float32
        assert abs(np.linalg.norm(vector) - 1.0) < 1e-5

    def test_query_prefix_changes_the_embedding(
        self, embedder: SentenceTransformerEmbedder
    ) -> None:
        with_prefix = embedder.embed_query("refund policy")
        as_document = embedder.embed_documents(["refund policy"])[0]
        assert cosine(with_prefix, as_document) < 0.9999

    def test_relevant_passage_scores_higher(self, embedder: SentenceTransformerEmbedder) -> None:
        query = embedder.embed_query("How do I reset my password?")
        relevant, unrelated = embedder.embed_documents(
            [
                "To reset your password, open Settings > Account and choose 'Forgot password'.",
                "The quarterly revenue grew by 12 percent compared to the previous year.",
            ]
        )
        assert cosine(query, relevant) > cosine(query, unrelated)
