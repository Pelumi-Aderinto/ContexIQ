# ContextIQ evaluation report

## Configuration

| Setting | Value |
|---|---|
| Timestamp (UTC) | 2026-09-24T07:07:00+00:00 |
| Git commit | c1047f6 |
| Dataset | evaluation/dataset.json (v1) |
| Questions evaluated | 27 |
| Documents | 5 |
| Embedding model | BAAI/bge-small-en-v1.5 (384-d) |
| Chunking | 1000 chars, overlap 150 |
| Retrieval mode | dense |
| top_k (answer path) | 5 |
| k values (retrieval metrics) | 1, 3, 5, 10 |
| Reranker | disabled |
| LLM provider / model | extractive / none |
| Answer mode | extractive |

**Extractive mode (no LLM).** The answer step returned the top passages verbatim instead of a synthesized answer, so *answer correctness* here means the expected keywords appear in the returned excerpts (each truncated to `citation_excerpt_chars`), not that a question was answered. In this mode the system only abstains when nothing is retrieved, so *abstention accuracy* on unanswerable questions is expected to be 0 and *false abstention* 0; use the retrieval and citation metrics to compare configurations.

## Summary

| Metric | Value |
|---|---|
| Recall@1 | 0.591 |
| Recall@3 | 0.750 |
| Recall@5 | 0.864 |
| Recall@10 | 0.909 |
| Hit rate@1 | 0.727 |
| Hit rate@3 | 0.864 |
| Hit rate@5 | 0.909 |
| Hit rate@10 | 0.955 |
| MRR | 0.814 |
| Citation validity | 1.000 |
| Citation page accuracy | 0.455 |
| Answer correctness (answerable) | 0.678 |
| Abstention accuracy (unanswerable) | 0.000 |
| False abstention rate (answerable) | 0.000 |
| Abstention precision / recall | n/a / 0.000 |
| Injection checks passed | 1.000 (1 checks) |
| Retrieval latency p50 / p95 (ms) | 15.9 / 20.6 |
| Generation latency p50 / p95 (ms) | 0.0 / 0.0 |
| Total latency p50 / p95 (ms) | 17.1 / 22.0 |

## By question type

| Type | n | Recall@5 | Hit@5 | MRR | Cit. valid | Cit. page acc. | Correctness | Abstained |
|---|---|---|---|---|---|---|---|---|
| answerable | 16 | 0.938 | 0.938 | 0.797 | 1.000 | 0.417 | 0.719 | 0.000 |
| cross_document | 6 | 0.667 | 0.833 | 0.861 | 1.000 | 0.556 | 0.569 | 0.000 |
| unanswerable | 5 | n/a | n/a | n/a | 1.000 | n/a | 0.000 | 0.000 |

## Per question

| id | type | hit@5 | citations valid | correctness | abstained | notes |
|---|---|---|---|---|---|---|
| q01 | answerable | yes | 1.00 | 1.00 | no | ok |
| q02 | answerable | yes | 1.00 | 0.50 | no | keywords not in answer: 16 weeks |
| q03 | answerable | yes | 1.00 | 1.00 | no | ok |
| q04 | answerable | yes | 1.00 | 1.00 | no | ok |
| q05 | answerable | yes | 1.00 | 0.00 | no | keywords not in answer: EN 4709-001, C3 |
| q06 | answerable | yes | 1.00 | 0.50 | no | keywords not in answer: 5 m/s |
| q07 | answerable | yes | 1.00 | 1.00 | no | ok |
| q08 | answerable | no | 1.00 | 0.50 | no | not in top-10: aurora_x200_installation_and_maintenance_guide p3; top retrieved: aurora_x200_installation_and_maintenance_guide p4, aurora_x200_technical_specification p6, aurora_x200_installation_and_maintenance_guide p5; keywords not in answer: 150 flight hours |
| q09 | answerable | yes | 1.00 | 0.00 | no | keywords not in answer: E-401, E-503, E-999 |
| q10 | answerable | yes | 1.00 | 1.00 | no | ok |
| q11 | answerable | yes | 1.00 | 1.00 | no | ok |
| q12 | answerable | yes | 1.00 | 1.00 | no | ok |
| q13 | answerable | yes | 1.00 | 1.00 | no | ok |
| q14 | answerable | yes | 1.00 | 0.00 | no | keywords not in answer: Norrland Grid Services, Kestrel Offshore Energy, Bluewater Port Authority |
| q15 | answerable | yes | 1.00 | 1.00 | no | ok |
| q16 | cross_document | yes | 1.00 | 0.67 | no | keywords not in answer: section 4 |
| q17 | cross_document | yes | 1.00 | 0.00 | no | not in top-10: halcyon_q2_fy2026_business_review p2; keywords not in answer: 4.2.1, RTK correction timeout, 1.5 m |
| q18 | cross_document | no | 1.00 | 0.75 | no | below top-5: halcyon_q2_fy2026_business_review p1 at rank 8, aurora_x200_technical_specification p6 at rank 6; top retrieved: halcyon_q2_fy2026_business_review p2, aurora_x200_installation_and_maintenance_guide p1, halcyon_q2_fy2026_business_review p4; keywords not in answer: 28,400 |
| q19 | cross_document | yes | 1.00 | 1.00 | no | ok |
| q20 | cross_document | yes | 1.00 | 1.00 | no | ok |
| q21 | cross_document | yes | 1.00 | 0.00 | no | not in top-10: halcyon_q2_fy2026_business_review p2; keywords not in answer: E-455, propeller imbalance, rotation direction |
| q22 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q23 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q24 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q25 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q26 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q27 | answerable | yes | 1.00 | 1.00 | no | ok |

## Ingestion

| File | Outcome | Pages | Chunks | ms |
|---|---|---|---|---|
| aurora_x200_installation_and_maintenance_guide.pdf | indexed | 6 | 18 | 1191 |
| aurora_x200_technical_specification.pdf | indexed | 6 | 15 | 1222 |
| halcyon_employee_handbook.pdf | indexed | 7 | 20 | 1174 |
| halcyon_information_security_policy.pdf | indexed | 6 | 18 | 1046 |
| halcyon_q2_fy2026_business_review.pdf | indexed | 4 | 11 | 685 |

Scores are uncalibrated proxies (see `evaluation/README.md`): retrieval metrics are page based, correctness is keyword coverage and abstention scoring is binary.
