"""Question-focused citation excerpts show the supporting sentence, not the chunk start."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.config import Settings
from app.generation.chain import AnswerService
from app.generation.citations import build_citations, make_excerpt
from app.models.domain import Chunk, ScoredChunk
from app.models.schemas import AnswerMode, QueryRequest

PASSAGE = (
    "Unused sick days do not carry over and are not paid out. A medical certificate is required "
    "after three consecutive days. Birthing parents receive 16 weeks of fully paid leave and "
    "non-birthing parents receive 8 weeks. Leave must start within 12 months of the birth."
)
QUESTION = "How many weeks of parental leave do birthing parents receive?"


def _scored(text: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=Chunk.make_id("doc1", 2, 0),
        document_id="doc1",
        workspace_id="ws",
        filename="handbook.pdf",
        page_number=2,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )
    return ScoredChunk(chunk=chunk, vector_id=1, score=0.5)


class _StubRetrieval:
    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, workspace_id: str, query: str, **_: object) -> list[ScoredChunk]:
        return self._chunks


# ---- make_excerpt ------------------------------------------------------------------------------


def test_focus_starts_at_best_matching_sentence() -> None:
    excerpt = make_excerpt(PASSAGE, 110, focus=QUESTION)
    assert excerpt.startswith("... Birthing parents receive 16 weeks")
    assert len(excerpt) <= 110


def test_focus_without_any_match_falls_back_to_start() -> None:
    assert make_excerpt(PASSAGE, 60, focus="zebra quantum") == make_excerpt(PASSAGE, 60)


def test_focus_matching_first_sentence_has_no_leading_ellipsis() -> None:
    excerpt = make_excerpt(PASSAGE, 60, focus="Do unused sick days carry over?")
    assert excerpt.startswith("Unused sick days")
    assert not excerpt.startswith("...")


def test_focus_is_ignored_when_text_fits() -> None:
    assert make_excerpt("Short. Text here.", 100, focus="text") == "Short. Text here."


def test_focus_counts_numeric_terms_like_error_codes() -> None:
    text = (
        "Error E-101 means the battery is below 15 percent. Error E-455 means propeller imbalance; "
        "check the rotation direction. Error E-999 means the firmware image is unsigned."
    )
    excerpt = make_excerpt(text, 75, focus="What does error code E-455 mean?")
    assert excerpt.startswith("... Error E-455 means propeller imbalance")


def test_focus_stopwords_do_not_pull_the_excerpt() -> None:
    text = "The the the the the the the the the the the the the the. Payment is due in 30 days."
    excerpt = make_excerpt(text, 40, focus="When is the payment due?")
    assert excerpt.startswith("... Payment is due in 30 days")


def test_build_citations_passes_focus_through() -> None:
    label_map = {"S1": _scored(PASSAGE)}
    citations, invalid = build_citations(["S1"], label_map, excerpt_chars=110, focus=QUESTION)
    assert invalid == []
    assert citations[0].excerpt.startswith("... Birthing parents receive 16 weeks")


# ---- AnswerService -----------------------------------------------------------------------------


def test_extractive_answer_uses_question_focused_excerpts() -> None:
    settings = Settings(_env_file=None, llm_provider="extractive", citation_excerpt_chars=110)
    service = AnswerService(
        settings=settings, retrieval=_StubRetrieval([_scored(PASSAGE)]), llm=None
    )

    response = service.answer("ws", QueryRequest(question=QUESTION), request_id="r1")

    assert response.answer_mode == AnswerMode.EXTRACTIVE
    assert response.citations[0].excerpt.startswith("... Birthing parents receive 16 weeks")
    assert "16 weeks" in response.answer


def test_llm_answer_citations_use_question_focused_excerpts() -> None:
    reply = '{"answer": "16 weeks [S1]", "citations": ["S1"], "insufficient_evidence": false}'
    llm = FakeListChatModel(responses=[reply])
    settings = Settings(_env_file=None, llm_provider="openai", citation_excerpt_chars=110)
    service = AnswerService(
        settings=settings, retrieval=_StubRetrieval([_scored(PASSAGE)]), llm=llm
    )

    response = service.answer("ws", QueryRequest(question=QUESTION), request_id="r1")

    assert response.answer_mode == AnswerMode.LLM
    assert response.abstained is False
    assert response.citations[0].excerpt.startswith("... Birthing parents receive 16 weeks")
