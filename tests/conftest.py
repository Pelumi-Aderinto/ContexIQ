"""Shared pytest fixtures for ContextIQ.

Keep this module light: nothing heavy (torch, faiss, sentence-transformers) is imported at
module import time so that fast unit tests stay fast.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.embeddings import HashingEmbedder
    from app.retrieval.retriever import RetrievalService
    from app.retrieval.vector_store import VectorStore
    from app.storage.metadata_store import MetadataStore

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"

# US Letter pages; taller pages are tried when a test passes a very long page text.
_PAGE_WIDTH = 612.0
_PAGE_HEIGHTS = (792.0, 1584.0, 3168.0, 6336.0, 12672.0)
_MARGIN = 54.0

MakePdf = Callable[..., bytes]


def _insert_page(doc: object, text: str) -> None:
    """Add one page holding ``text`` (wrapped), growing the page height until it fits."""
    import pymupdf

    for height in _PAGE_HEIGHTS:
        page = doc.new_page(width=_PAGE_WIDTH, height=height)  # type: ignore[attr-defined]
        if not text:
            return
        rect = pymupdf.Rect(_MARGIN, _MARGIN, _PAGE_WIDTH - _MARGIN, height - _MARGIN)
        remaining = page.insert_textbox(rect, text, fontsize=10, fontname="helv")
        if remaining >= 0:
            return
        doc.delete_page(-1)  # type: ignore[attr-defined]
    raise ValueError("page text is too long for the test PDF builder")


@pytest.fixture
def make_pdf() -> MakePdf:
    """Factory: ``make_pdf(pages, *, title=None) -> bytes`` builds a PDF in memory."""

    def _make(pages: list[str], *, title: str | None = None) -> bytes:
        import pymupdf

        doc = pymupdf.open()
        try:
            for text in pages:
                _insert_page(doc, text)
            if title:
                doc.set_metadata({"title": title})
            return doc.tobytes()
        finally:
            doc.close()

    return _make


@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    return SAMPLE_DATA_DIR


@pytest.fixture(scope="session")
def sample_pdf_paths(sample_data_dir: Path) -> list[Path]:
    paths = sorted(sample_data_dir.glob("*.pdf"))
    if not paths:
        pytest.skip("sample_data/*.pdf missing; run scripts/generate_sample_data.py")
    return paths


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Isolated test settings: extractive LLM, API-key auth with two workspaces, tmp data dir."""
    from app.core.config import Settings

    for name in list(os.environ):
        if name.startswith("CONTEXTIQ_"):
            monkeypatch.delenv(name, raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        _env_file=None,
        data_dir=data_dir,
        auth_mode="api_key",
        api_keys="test-key-alpha-000:alpha,test-key-beta-0000:beta",
        llm_provider="extractive",
        environment="test",
    )


# --------------------------------------------------------------------------------------------
# Ingestion + retrieval stack (pipeline, retriever and isolation tests)
# --------------------------------------------------------------------------------------------

HASHING_DIMENSION = 64


@dataclass
class RagStack:
    """Everything needed to ingest and retrieve in tests, built on the ``HashingEmbedder``."""

    settings: Settings
    store: MetadataStore
    vector_store: VectorStore
    embedder: HashingEmbedder
    pipeline: IngestionPipeline
    retrieval: RetrievalService

    def close(self) -> None:
        self.store.close()


@pytest.fixture
def make_rag_stack(settings: Settings) -> Iterator[Callable[..., RagStack]]:
    """Factory: ``make_rag_stack(**settings_overrides) -> RagStack`` over the test data dir.

    Calling it again builds a second, independent stack over the same SQLite file and FAISS
    directory, which simulates an application restart. Every stack is closed at teardown.
    """
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.embeddings import HashingEmbedder
    from app.retrieval.retriever import RetrievalService
    from app.retrieval.vector_store import VectorStore
    from app.storage.metadata_store import MetadataStore

    stacks: list[RagStack] = []

    def _make(**overrides: object) -> RagStack:
        stack_settings = settings.model_copy(update=overrides) if overrides else settings
        embedder = HashingEmbedder(dimension=HASHING_DIMENSION)
        store = MetadataStore(stack_settings.db_path)
        vector_store = VectorStore(stack_settings.index_dir, embedder.dimension)
        stack = RagStack(
            settings=stack_settings,
            store=store,
            vector_store=vector_store,
            embedder=embedder,
            pipeline=IngestionPipeline(
                settings=stack_settings, store=store, vector_store=vector_store, embedder=embedder
            ),
            retrieval=RetrievalService(
                settings=stack_settings, store=store, vector_store=vector_store, embedder=embedder
            ),
        )
        stacks.append(stack)
        return stack

    yield _make
    for stack in stacks:
        stack.close()


@pytest.fixture
def rag_stack(make_rag_stack: Callable[..., RagStack]) -> RagStack:
    """A ready-to-use stack with small chunks so short test PDFs still yield several chunks."""
    return make_rag_stack(chunk_size=200, chunk_overlap=20)
