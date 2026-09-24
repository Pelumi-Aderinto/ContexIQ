"""Tolerant parsing of LLM answers and verification of their citations.

The model is asked for ``{"answer": str, "citations": ["S1", ...], "insufficient_evidence": bool}``
but real outputs are messy: fenced JSON, JSON embedded in prose, or plain prose with inline
``[S1]`` markers. Everything here degrades gracefully and never raises on odd input. Labels are
verified against the ``label_map`` produced by ``prompts.format_context`` so a citation can only
ever point at a chunk that was actually retrieved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.models.domain import ScoredChunk
from app.models.schemas import Citation

_LABEL_RE = re.compile(r"^S(\d+)$")
# "[S1]", "[s1, S3]", "[S1][S2]" -> bracket groups containing only labels.
_INLINE_GROUP_RE = re.compile(r"\[\s*([Ss]\s*\d+(?:\s*[,;/]?\s*[Ss]\s*\d+)*)\s*\]")
_INLINE_LABEL_RE = re.compile(r"[Ss]\s*(\d+)")
_FENCE_OPEN_RE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?```\s*$")
_ELLIPSIS = "..."


class LLMAnswer(BaseModel):
    """Structured answer the LLM is asked to return."""

    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value: object) -> str:
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list | tuple):
            return []
        coerced: list[str] = []
        for item in value:
            if isinstance(item, str):
                coerced.append(item)
            elif isinstance(item, int) and not isinstance(item, bool):
                coerced.append(str(item))
        return coerced

    @field_validator("insufficient_evidence", mode="before")
    @classmethod
    def _coerce_flag(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        if isinstance(value, int | float):
            return bool(value)
        return False


def normalize_label(raw: str) -> str | None:
    """Normalise ``"s1"``, ``"[S1]"``, ``"S 1"`` to ``"S1"``; anything else -> ``None``."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    cleaned = re.sub(r"\s+", "", cleaned).upper()
    match = _LABEL_RE.match(cleaned)
    if match is None:
        return None
    return f"S{int(match.group(1))}"


def extract_inline_labels(text: str) -> list[str]:
    """Return labels cited inline as ``[S#]`` in order of first appearance, deduplicated."""
    labels: list[str] = []
    for group in _INLINE_GROUP_RE.finditer(text):
        for number in _INLINE_LABEL_RE.findall(group.group(1)):
            label = f"S{int(number)}"
            if label not in labels:
                labels.append(label)
    return labels


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (```json ... ``` or ``` ... ```)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = _FENCE_OPEN_RE.sub("", stripped, count=1)
    return _FENCE_CLOSE_RE.sub("", stripped, count=1).strip()


def _iter_balanced_objects(text: str) -> Iterator[str]:
    """Yield each top-level ``{...}`` substring, honouring quoted strings inside objects."""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if depth > 0 and in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start : index + 1]
        elif char == '"' and depth > 0:
            in_string = True


def _load_json_object(text: str) -> dict[str, object] | None:
    """Parse ``text`` as a JSON object, or the first balanced ``{...}`` embedded in it."""
    candidates: list[str] = [text]
    candidates.extend(_iter_balanced_objects(text))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _fallback_answer(text: str) -> LLMAnswer:
    """Treat the raw output as prose, citing whatever inline labels it contains."""
    return LLMAnswer(answer=text.strip(), citations=extract_inline_labels(text))


def _merge_inline_labels(parsed: LLMAnswer) -> LLMAnswer:
    """Union the explicit citation list with labels cited inline in the answer text."""
    seen = {normalize_label(label) or label.strip() for label in parsed.citations}
    merged = list(parsed.citations)
    for label in extract_inline_labels(parsed.answer):
        if label not in seen:
            merged.append(label)
            seen.add(label)
    return parsed.model_copy(update={"citations": merged})


def parse_llm_answer(text: str) -> LLMAnswer:
    """Parse the model output tolerantly; never raises.

    Strips code fences, tries the whole text as JSON, then the first balanced ``{...}`` object.
    If nothing parses, the raw text becomes the answer and inline ``[S#]`` markers become the
    citations. Inline markers are always merged into the citation list.
    """
    payload = _load_json_object(_strip_code_fences(text))
    if payload is None:
        return _merge_inline_labels(_fallback_answer(text))
    try:
        parsed = LLMAnswer.model_validate(payload)
    except ValidationError:
        return _merge_inline_labels(_fallback_answer(text))
    return _merge_inline_labels(parsed)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TERM_RE = re.compile(r"[a-z0-9]+")
_LEADING_ELLIPSIS = "... "
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "does", "do", "did", "for", "from",
        "has", "have", "how", "in", "is", "it", "its", "many", "much", "of", "on", "or", "that",
        "the", "their", "there", "these", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "why", "will", "with", "would", "you", "your", "about", "into", "than",
        "then", "them", "they", "can", "could", "should", "may", "might", "per", "each", "any",
        "all", "some", "also", "between", "during", "under", "over", "after", "before", "within",
        "without",
    }
)  # fmt: skip


def _focus_terms(focus: str) -> set[str]:
    """Distinctive lowercase tokens of ``focus`` (numbers of 2+ digits, words of 3+ letters)."""
    terms: set[str] = set()
    for token in _TERM_RE.findall(focus.lower()):
        if token in _STOPWORDS:
            continue
        min_len = 2 if token.isdigit() else 3
        if len(token) >= min_len:
            terms.add(token)
    return terms


def _best_sentence_offset(collapsed: str, terms: set[str]) -> int:
    """Start offset of the sentence sharing the most terms with the focus (0 if none match).

    ``collapsed`` must have single-space whitespace, so sentence separators are exactly one
    character wide and offsets can be accumulated without re-scanning.
    """
    best_offset, best_score, offset = 0, 0, 0
    for sentence in _SENTENCE_SPLIT_RE.split(collapsed):
        lowered = sentence.lower()
        score = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}", lowered))
        if score > best_score:
            best_offset, best_score = offset, score
        offset += len(sentence) + 1
    return best_offset


def _truncate_tail(text: str, max_chars: int) -> str:
    """Cut ``text`` at a word boundary with a trailing ellipsis if it exceeds ``max_chars``."""
    if len(text) <= max_chars:
        return text
    budget = max(1, max_chars - len(_ELLIPSIS))
    cut = text[:budget]
    boundary = cut.rfind(" ")
    if boundary >= budget // 2:
        cut = cut[:boundary]
    return cut.rstrip() + _ELLIPSIS


def make_excerpt(text: str, max_chars: int, *, focus: str | None = None) -> str:
    """Collapse whitespace and return at most ``max_chars`` characters of ``text``.

    Without ``focus`` the excerpt is the start of the text. With ``focus`` (typically the
    user's question) the excerpt starts at the sentence that shares the most distinctive
    terms with it, prefixed with an ellipsis when that is not the beginning, so a citation
    shows the passage that actually supports the answer rather than the top of the chunk.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    start = 0
    if focus:
        terms = _focus_terms(focus)
        if terms:
            start = _best_sentence_offset(collapsed, terms)
    if start == 0:
        return _truncate_tail(collapsed, max_chars)
    budget = max(1, max_chars - len(_LEADING_ELLIPSIS))
    return _LEADING_ELLIPSIS + _truncate_tail(collapsed[start:], budget)


def build_citations(
    labels: list[str],
    label_map: dict[str, ScoredChunk],
    *,
    excerpt_chars: int,
    focus: str | None = None,
) -> tuple[list[Citation], list[str]]:
    """Map cited labels to verified ``Citation`` objects.

    Returns ``(valid citations in label order, invalid labels)``. Labels are normalised and
    de-duplicated while preserving order; anything not in ``label_map`` is reported invalid.
    ``focus`` (the question) steers each excerpt towards the supporting sentence.
    """
    citations: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        label = normalize_label(raw)
        key = label or raw.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        sc = label_map.get(label) if label else None
        if sc is None:
            invalid.append(key)
            continue
        citations.append(_to_citation(key, sc, excerpt_chars, focus))
    return citations, invalid


def _to_citation(
    label: str, sc: ScoredChunk, excerpt_chars: int, focus: str | None = None
) -> Citation:
    chunk = sc.chunk
    return Citation(
        citation_id=label,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        excerpt=make_excerpt(chunk.text, excerpt_chars, focus=focus),
        score=sc.score,
    )
