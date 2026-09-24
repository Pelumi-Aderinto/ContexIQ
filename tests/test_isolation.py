"""Workspace isolation: retrieval in one workspace must never surface another workspace's data.

Two workspaces (``alpha`` and ``beta``) are populated with different documents; ``beta`` holds a
made-up term that exists nowhere in ``alpha``. Every test then attacks the boundary from a
different angle: plain queries, the beta-only term, beta document ids, deletions and a
simulated restart over the same data directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from app.models.schemas import DocumentInfo, RetrievalMode, UploadOutcome

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import MakePdf, RagStack

ALPHA = "alpha"
BETA = "beta"
MODES = (RetrievalMode.DENSE, RetrievalMode.HYBRID)

# Deliberately overlapping vocabulary ("inspection", "valve", "blue") so that similarity alone
# would happily cross workspaces if the boundary were not enforced.
ALPHA_DOCS: dict[str, list[str]] = {
    "alpha_minerals.pdf": [
        "Zephyrite is a rare blue mineral found in ancient riverbeds. Collectors prize "
        "zephyrite for its faint blue glow and inspect every zephyrite sample under UV light.",
        "Granite quarries in the northern hills supply durable stone blocks. Each quarry runs "
        "a valve-controlled water system to keep dust down during cutting and inspection.",
    ],
    "alpha_birds.pdf": [
        "Kestrel migration routes cross the plains every autumn. Kestrels nest in cliff "
        "hollows and abandoned barns, and volunteers inspect the nests each spring.",
        "The blue heron colony on the estuary is counted every June by boat. Counts are "
        "logged with the date, weather and the name of the inspection volunteer.",
    ],
}
BETA_ONLY_TERM = "quokkafern"
BETA_DOCS: dict[str, list[str]] = {
    "beta_bulletin.pdf": [
        "Maintenance bulletin. Replace the pressure valve AX2-7731 every twelve months and "
        f"record the {BETA_ONLY_TERM} coating batch number in the inspection log.",
        f"The {BETA_ONLY_TERM} coating protects valve seats from corrosion. {BETA_ONLY_TERM} "
        "must be reapplied after every inspection of the blue-tagged valves.",
    ],
    "beta_recipes.pdf": [
        "Sourdough starter needs flour, water and patience. Feed the starter daily and keep "
        "it at room temperature away from the blue kitchen window.",
        "Inspection of the crust tells you when the loaf is done: tap it and listen for a "
        "hollow sound, then cool the loaf on a wire rack.",
    ],
}
QUERIES = (
    "zephyrite mineral",
    f"{BETA_ONLY_TERM} coating",
    "AX2-7731 valve",
    "kestrel migration",
    "sourdough starter",
    "granite quarry",
    "inspection",
    "blue",
    "valve seats corrosion",
    "counted every June",
)


@dataclass
class TwoWorkspaces:
    stack: RagStack
    alpha: dict[str, DocumentInfo]
    beta: dict[str, DocumentInfo]

    def docs(self, workspace_id: str) -> dict[str, DocumentInfo]:
        return self.alpha if workspace_id == ALPHA else self.beta


def _ingest_all(stack: RagStack, workspace_id: str, docs: dict[str, list[str]], make_pdf: MakePdf):
    indexed: dict[str, DocumentInfo] = {}
    for filename, pages in docs.items():
        result = stack.pipeline.ingest(make_pdf(pages), filename, workspace_id)
        assert result.outcome is UploadOutcome.INDEXED, result.message
        assert result.document is not None
        indexed[filename] = result.document
    return indexed


@pytest.fixture
def two_workspaces(rag_stack: RagStack, make_pdf: MakePdf) -> TwoWorkspaces:
    return TwoWorkspaces(
        stack=rag_stack,
        alpha=_ingest_all(rag_stack, ALPHA, ALPHA_DOCS, make_pdf),
        beta=_ingest_all(rag_stack, BETA, BETA_DOCS, make_pdf),
    )


def _assert_results_belong_to(world: TwoWorkspaces, workspace_id: str, results) -> None:
    own_docs = world.docs(workspace_id)
    own_ids = {doc.document_id for doc in own_docs.values()}
    for sc in results:
        assert sc.chunk.workspace_id == workspace_id
        assert sc.chunk.document_id in own_ids
        assert sc.chunk.filename in own_docs


# ---- queries ---------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("workspace_id", [ALPHA, BETA])
def test_every_result_belongs_to_the_queried_workspace(
    two_workspaces: TwoWorkspaces, workspace_id: str, mode: RetrievalMode
) -> None:
    total = 0
    for query in QUERIES:
        results = two_workspaces.stack.retrieval.retrieve(workspace_id, query, k=20, mode=mode)
        _assert_results_belong_to(two_workspaces, workspace_id, results)
        total += len(results)
    assert total > 0


@pytest.mark.parametrize("mode", MODES)
def test_beta_only_term_never_appears_in_alpha(
    two_workspaces: TwoWorkspaces, mode: RetrievalMode
) -> None:
    retrieval = two_workspaces.stack.retrieval
    alpha_results = retrieval.retrieve(ALPHA, BETA_ONLY_TERM, k=20, mode=mode)
    _assert_results_belong_to(two_workspaces, ALPHA, alpha_results)
    assert all(BETA_ONLY_TERM not in sc.chunk.text.lower() for sc in alpha_results)

    # Sanity check that the term is retrievable where it actually lives.
    beta_results = retrieval.retrieve(BETA, BETA_ONLY_TERM, k=5, mode=mode)
    assert beta_results
    assert BETA_ONLY_TERM in beta_results[0].chunk.text.lower()
    assert beta_results[0].chunk.filename == "beta_bulletin.pdf"


def test_keyword_search_is_workspace_scoped(two_workspaces: TwoWorkspaces) -> None:
    store = two_workspaces.stack.store
    assert store.keyword_search(ALPHA, BETA_ONLY_TERM, 10) == []
    assert store.keyword_search(BETA, BETA_ONLY_TERM, 10)


@pytest.mark.parametrize("mode", MODES)
def test_beta_document_ids_yield_nothing_in_alpha(
    two_workspaces: TwoWorkspaces, mode: RetrievalMode
) -> None:
    beta_ids = [doc.document_id for doc in two_workspaces.beta.values()]
    retrieval = two_workspaces.stack.retrieval
    query = f"{BETA_ONLY_TERM} AX2-7731 valve"
    assert retrieval.retrieve(ALPHA, query, document_ids=beta_ids, mode=mode) == []
    # Mixing one alpha id with beta ids only ever yields alpha chunks.
    alpha_id = two_workspaces.alpha["alpha_minerals.pdf"].document_id
    mixed = retrieval.retrieve(ALPHA, "zephyrite", document_ids=[*beta_ids, alpha_id], mode=mode)
    assert mixed
    assert {sc.chunk.document_id for sc in mixed} == {alpha_id}


def test_langchain_retriever_honours_workspace_boundary(two_workspaces: TwoWorkspaces) -> None:
    retriever = two_workspaces.stack.retrieval.as_langchain_retriever(ALPHA, k=10)
    docs = retriever.invoke(f"{BETA_ONLY_TERM} coating for valve seats")
    assert docs
    alpha_ids = {doc.document_id for doc in two_workspaces.alpha.values()}
    assert all(doc.metadata["document_id"] in alpha_ids for doc in docs)
    assert all(BETA_ONLY_TERM not in doc.page_content.lower() for doc in docs)


# ---- mutations ------------------------------------------------------------------------------


def test_deleting_a_beta_document_leaves_alpha_untouched(two_workspaces: TwoWorkspaces) -> None:
    stack = two_workspaces.stack
    alpha_chunks_before = stack.store.count_chunks(ALPHA)
    alpha_vectors_before = stack.vector_store.count(ALPHA)
    alpha_hits_before = [
        (sc.vector_id, sc.score) for sc in stack.retrieval.retrieve(ALPHA, "zephyrite", k=5)
    ]
    bulletin = two_workspaces.beta["beta_bulletin.pdf"]
    beta_chunks_before = stack.store.count_chunks(BETA)

    response = stack.pipeline.delete_document(BETA, bulletin.document_id)

    assert response is not None
    assert response.chunks_removed == bulletin.chunk_count > 0
    assert stack.store.count_chunks(BETA) == beta_chunks_before - bulletin.chunk_count
    assert stack.vector_store.count(BETA) == stack.store.count_chunks(BETA)
    assert stack.store.count_chunks(ALPHA) == alpha_chunks_before
    assert stack.vector_store.count(ALPHA) == alpha_vectors_before
    alpha_hits_after = [
        (sc.vector_id, sc.score) for sc in stack.retrieval.retrieve(ALPHA, "zephyrite", k=5)
    ]
    assert alpha_hits_after == alpha_hits_before
    for mode in MODES:
        remaining = stack.retrieval.retrieve(BETA, BETA_ONLY_TERM, k=20, mode=mode)
        _assert_results_belong_to(two_workspaces, BETA, remaining)
        assert all(BETA_ONLY_TERM not in sc.chunk.text.lower() for sc in remaining)


def test_alpha_cannot_delete_a_beta_document(two_workspaces: TwoWorkspaces) -> None:
    stack = two_workspaces.stack
    bulletin = two_workspaces.beta["beta_bulletin.pdf"]
    assert stack.pipeline.delete_document(ALPHA, bulletin.document_id) is None
    assert stack.store.get_document(BETA, bulletin.document_id) == bulletin
    assert stack.store.count_chunks(BETA) == sum(
        d.chunk_count for d in two_workspaces.beta.values()
    )


# ---- storage layout and restart --------------------------------------------------------------


def test_each_workspace_has_its_own_index_file(two_workspaces: TwoWorkspaces) -> None:
    stack = two_workspaces.stack
    index_files = sorted(p.name for p in stack.settings.index_dir.glob("*.faiss"))
    assert index_files == [f"{ALPHA}.faiss", f"{BETA}.faiss"]
    assert stack.vector_store.workspaces() == [ALPHA, BETA]
    assert stack.store.list_workspaces() == [ALPHA, BETA]
    for workspace_id in (ALPHA, BETA):
        probe = stack.embedder.embed_query("probe")
        indexed_ids = {vid for vid, _ in stack.vector_store.search(workspace_id, probe, k=1000)}
        assert indexed_ids == set(stack.store.list_vector_ids(workspace_id))
        assert len(indexed_ids) == stack.store.count_chunks(workspace_id)


def test_restart_preserves_both_workspaces_and_isolation(
    two_workspaces: TwoWorkspaces, make_rag_stack: Callable[..., RagStack]
) -> None:
    first = two_workspaces.stack
    before = {
        ws: [(sc.vector_id, sc.chunk.chunk_id) for sc in first.retrieval.retrieve(ws, q, k=5)]
        for ws in (ALPHA, BETA)
        for q in ("zephyrite", BETA_ONLY_TERM)
    }

    # A fresh store + index over the same data directory stands in for a process restart.
    restarted = make_rag_stack(chunk_size=200, chunk_overlap=20)

    assert restarted.vector_store.workspaces() == [ALPHA, BETA]
    for ws in (ALPHA, BETA):
        assert restarted.store.count_chunks(ws) == first.store.count_chunks(ws) > 0
        assert restarted.vector_store.count(ws) == first.vector_store.count(ws)
        assert {d.document_id for d in restarted.store.list_documents(ws)} == {
            d.document_id for d in two_workspaces.docs(ws).values()
        }
    after = {
        ws: [(sc.vector_id, sc.chunk.chunk_id) for sc in restarted.retrieval.retrieve(ws, q, k=5)]
        for ws in (ALPHA, BETA)
        for q in ("zephyrite", BETA_ONLY_TERM)
    }
    assert after == before
    for mode in MODES:
        results = restarted.retrieval.retrieve(ALPHA, BETA_ONLY_TERM, k=20, mode=mode)
        _assert_results_belong_to(two_workspaces, ALPHA, results)
        assert all(BETA_ONLY_TERM not in sc.chunk.text.lower() for sc in results)
