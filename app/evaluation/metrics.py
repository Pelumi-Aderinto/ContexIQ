"""Pure evaluation metrics for ContextIQ.

Every function here is side-effect free and works on plain values (plus the public
``Citation`` schema), so the evaluation runner (``scripts/evaluate.py``) can score each
question and aggregate however it likes. Nothing in this module performs I/O, loads models
or touches the network.

Conventions
-----------
* Retrieval metrics operate on ``(filename, page_number)`` pairs rather than chunk ids: a page
  is the unit of provenance a human can verify against the PDF.
* Metrics that are undefined for an input (for example page accuracy with no citations) return
  ``None`` instead of a misleading number; the runner should skip those when averaging.
* Keyword coverage is a *proxy* for correctness. It rewards answers containing the expected
  distinctive strings and is not calibrated against human judgement.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Collection, Iterable, Sequence
from typing import NamedTuple

from app.models.schemas import Citation

SourceRef = tuple[str, int]
"""A ``(filename, page_number)`` pair identifying one page of one document."""

# A comma sitting between a digit and a group of exactly three digits, i.e. "1,240" but
# not the list separator in "1,2" or the decimal-ish "1,2400".
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _dedupe(pairs: Iterable[SourceRef]) -> list[SourceRef]:
    """Return ``pairs`` without duplicates, keeping the first occurrence of each in order."""
    seen: set[SourceRef] = set()
    ordered: list[SourceRef] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            ordered.append(pair)
    return ordered


def _require_positive_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _citation_pages(citations: Iterable[Citation]) -> list[SourceRef]:
    return [(c.filename, c.page_number) for c in citations]


def normalize_for_matching(text: str) -> str:
    """Canonical form used for keyword matching.

    Applies NFKC, case folding, removal of thousands separators inside numbers ("1,240" ->
    "1240") and whitespace collapsing, so "1,500  US Dollars" and "$1500 us dollars" compare
    on equal footing.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = _THOUSANDS_SEPARATOR.sub("", folded)
    return _WHITESPACE.sub(" ", folded).strip()


# --------------------------------------------------------------------------------------------
# Retrieval metrics over (filename, page) pairs
# --------------------------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[SourceRef], expected: Sequence[SourceRef], k: int) -> float:
    """Fraction of expected pages found among the first ``k`` distinct retrieved pages.

    Retrieved pairs are de-duplicated (preserving order) *before* the cut-off, so several
    chunks from the same page occupy one slot. With no expected pages there is nothing to
    miss and the result is ``1.0``; the runner should exclude such questions (typically the
    unanswerable ones) from retrieval averages.
    """
    _require_positive_k(k)
    wanted = set(expected)
    if not wanted:
        return 1.0
    top = set(_dedupe(retrieved)[:k])
    return len(wanted & top) / len(wanted)


def hit_at_k(retrieved: Sequence[SourceRef], expected: Sequence[SourceRef], k: int) -> bool:
    """True when at least one expected page is among the first ``k`` distinct retrieved pages."""
    _require_positive_k(k)
    wanted = set(expected)
    return any(pair in wanted for pair in _dedupe(retrieved)[:k])


def mrr(retrieved: Sequence[SourceRef], expected: Sequence[SourceRef]) -> float:
    """Reciprocal rank of the first expected page in the de-duplicated retrieved list.

    Returns ``0.0`` when no expected page was retrieved (or nothing was expected).
    """
    wanted = set(expected)
    for rank, pair in enumerate(_dedupe(retrieved), start=1):
        if pair in wanted:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------------------------
# Citation metrics
# --------------------------------------------------------------------------------------------


def citation_validity(citations: Sequence[Citation], retrieved_chunk_ids: Collection[str]) -> float:
    """Fraction of citations whose ``chunk_id`` was actually retrieved.

    ``1.0`` when every citation is valid, including the trivial case of no citations. The
    generation layer is supposed to guarantee this invariant; the metric exists to catch
    regressions.
    """
    if not citations:
        return 1.0
    valid = sum(1 for c in citations if c.chunk_id in retrieved_chunk_ids)
    return valid / len(citations)


def citation_page_accuracy(
    citations: Sequence[Citation], expected_pages: Sequence[SourceRef]
) -> float | None:
    """Fraction of citations pointing at an expected ``(filename, page)``.

    Returns ``None`` when there are no citations, since accuracy is then undefined.
    """
    if not citations:
        return None
    wanted = set(expected_pages)
    on_target = sum(1 for pair in _citation_pages(citations) if pair in wanted)
    return on_target / len(citations)


# --------------------------------------------------------------------------------------------
# Answer metrics
# --------------------------------------------------------------------------------------------


def keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> float:
    """Fraction of distinct expected keywords (or phrases) present in ``answer``.

    Matching is a substring test after :func:`normalize_for_matching` on both sides, so it is
    case-insensitive, whitespace-insensitive and tolerant of thousands separators. Blank
    keywords are ignored; with no usable keywords the coverage is vacuously ``1.0``.
    """
    keywords = list(dict.fromkeys(normalize_for_matching(k) for k in expected_keywords))
    keywords = [k for k in keywords if k]
    if not keywords:
        return 1.0
    haystack = normalize_for_matching(answer)
    hits = sum(1 for k in keywords if k in haystack)
    return hits / len(keywords)


def answer_correctness(
    answer: str, expected_keywords: Sequence[str], *, abstained: bool, answerable: bool
) -> float:
    """Score one answer in ``[0, 1]``.

    * Unanswerable question: ``1.0`` if the system abstained, otherwise ``0.0``.
    * Answerable question: ``0.0`` if the system abstained, otherwise the keyword coverage.
    """
    if not answerable:
        return 1.0 if abstained else 0.0
    if abstained:
        return 0.0
    return keyword_coverage(answer, expected_keywords)


def forbidden_terms_found(answer: str, terms: Sequence[str]) -> list[str]:
    """Return the ``terms`` that occur in ``answer`` (normalized, case-insensitive match).

    Used for hard checks such as a dataset entry's ``must_not_contain`` list, which verifies
    that the system did not obey an instruction embedded in a document.
    """
    haystack = normalize_for_matching(answer)
    return [t for t in terms if normalize_for_matching(t) and normalize_for_matching(t) in haystack]


class AbstentionOutcome(NamedTuple):
    """Whether the system abstained on a question and whether that question was answerable."""

    abstained: bool
    answerable: bool


def abstention_metrics(outcomes: Iterable[tuple[bool, bool]]) -> dict[str, float | None]:
    """Precision and recall of abstention, with "abstained on an unanswerable question" as the
    positive class.

    * ``precision`` = correct abstentions / all abstentions
    * ``recall``    = correct abstentions / all unanswerable questions

    Each outcome is ``(abstained, answerable)`` (see :class:`AbstentionOutcome`). A ratio
    with a zero denominator is reported as ``None``.
    """
    true_positive = false_positive = false_negative = 0
    for abstained, answerable in outcomes:
        if abstained and not answerable:
            true_positive += 1
        elif abstained and answerable:
            false_positive += 1
        elif not abstained and not answerable:
            false_negative += 1
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    return {
        "precision": true_positive / predicted if predicted else None,
        "recall": true_positive / actual if actual else None,
    }


# --------------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------------


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile ``p`` (0-100) of an already sorted, non-empty sequence."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be within [0, 100], got {p}")
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return float(sorted_values[rank - 1])


def latency_summary(values_ms: Sequence[float]) -> dict[str, float]:
    """Summarize latencies: ``p50``, ``p95`` (nearest-rank), ``mean``, ``max`` and ``n``.

    An empty input yields zeros with ``n == 0`` so the runner can always emit the same keys.
    """
    if not values_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    ordered = sorted(float(v) for v in values_ms)
    return {
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
        "n": len(ordered),
    }
