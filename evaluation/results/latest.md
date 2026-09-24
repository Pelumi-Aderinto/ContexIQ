# ContextIQ evaluation report

## Configuration

| Setting | Value |
|---|---|
| Timestamp (UTC) | 2026-09-24T07:06:49+00:00 |
| Git commit | c1047f6 |
| Dataset | evaluation/dataset.json (v1) |
| Questions evaluated | 27 |
| Documents | 5 |
| Embedding model | BAAI/bge-small-en-v1.5 (384-d) |
| Chunking | 1000 chars, overlap 150 |
| Retrieval mode | hybrid |
| top_k (answer path) | 5 |
| k values (retrieval metrics) | 1, 3, 5, 10 |
| Reranker | disabled |
| LLM provider / model | extractive / none |
| Answer mode | extractive |

**Extractive mode (no LLM).** The answer step returned the top passages verbatim instead of a synthesized answer, so *answer correctness* here means the expected keywords appear in the returned excerpts (each truncated to `citation_excerpt_chars`), not that a question was answered. In this mode the system only abstains when nothing is retrieved, so *abstention accuracy* on unanswerable questions is expected to be 0 and *false abstention* 0; use the retrieval and citation metrics to compare configurations.

## Summary

| Metric | Value |
|---|---|
| Recall@1 | 0.705 |
| Recall@3 | 0.886 |
| Recall@5 | 0.932 |
| Recall@10 | 1.000 |
| Hit rate@1 | 0.864 |
| Hit rate@3 | 0.955 |
| Hit rate@5 | 0.955 |
| Hit rate@10 | 1.000 |
| MRR | 0.909 |
| Citation validity | 1.000 |
| Citation page accuracy | 0.515 |
| Answer correctness (answerable) | 0.811 |
| Abstention accuracy (unanswerable) | 0.000 |
| False abstention rate (answerable) | 0.000 |
| Abstention precision / recall | n/a / 0.000 |
| Injection checks passed | 1.000 (1 checks) |
| Retrieval latency p50 / p95 (ms) | 17.5 / 24.2 |
| Generation latency p50 / p95 (ms) | 0.0 / 0.0 |
| Total latency p50 / p95 (ms) | 18.2 / 25.5 |

## By question type

| Type | n | Recall@5 | Hit@5 | MRR | Cit. valid | Cit. page acc. | Correctness | Abstained |
|---|---|---|---|---|---|---|---|---|
| answerable | 16 | 0.906 | 0.938 | 0.948 | 1.000 | 0.521 | 0.854 | 0.000 |
| cross_document | 6 | 1.000 | 1.000 | 0.806 | 1.000 | 0.500 | 0.694 | 0.000 |
| unanswerable | 5 | n/a | n/a | n/a | 1.000 | n/a | 0.000 | 0.000 |

## Per question

| id | type | hit@5 | citations valid | correctness | abstained | notes |
|---|---|---|---|---|---|---|
| q01 | answerable | yes | 1.00 | 1.00 | no | ok |
| q02 | answerable | yes | 1.00 | 0.50 | no | keywords not in answer: 16 weeks |
| q03 | answerable | yes | 1.00 | 1.00 | no | ok |
| q04 | answerable | yes | 1.00 | 1.00 | no | ok |
| q05 | answerable | yes | 1.00 | 1.00 | no | below top-5: aurora_x200_technical_specification p5 at rank 6 |
| q06 | answerable | yes | 1.00 | 0.50 | no | keywords not in answer: 5 m/s |
| q07 | answerable | yes | 1.00 | 1.00 | no | ok |
| q08 | answerable | no | 1.00 | 0.00 | no | below top-5: aurora_x200_installation_and_maintenance_guide p3 at rank 6; top retrieved: aurora_x200_installation_and_maintenance_guide p4, aurora_x200_installation_and_maintenance_guide p1, aurora_x200_technical_specification p6; keywords not in answer: 150 flight hours, 12 months |
| q09 | answerable | yes | 1.00 | 1.00 | no | ok |
| q10 | answerable | yes | 1.00 | 1.00 | no | ok |
| q11 | answerable | yes | 1.00 | 1.00 | no | ok |
| q12 | answerable | yes | 1.00 | 1.00 | no | ok |
| q13 | answerable | yes | 1.00 | 1.00 | no | ok |
| q14 | answerable | yes | 1.00 | 0.67 | no | keywords not in answer: Bluewater Port Authority |
| q15 | answerable | yes | 1.00 | 1.00 | no | ok |
| q16 | cross_document | yes | 1.00 | 0.67 | no | keywords not in answer: section 4 |
| q17 | cross_document | yes | 1.00 | 0.33 | no | keywords not in answer: RTK correction timeout, 1.5 m |
| q18 | cross_document | yes | 1.00 | 1.00 | no | ok |
| q19 | cross_document | yes | 1.00 | 0.50 | no | keywords not in answer: Elena Marchetti-Roy |
| q20 | cross_document | yes | 1.00 | 1.00 | no | ok |
| q21 | cross_document | yes | 1.00 | 0.67 | no | keywords not in answer: rotation direction |
| q22 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q23 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q24 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q25 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q26 | unanswerable | n/a | 1.00 | 0.00 | no | answered instead of abstaining |
| q27 | answerable | yes | 1.00 | 1.00 | no | ok |

## Ingestion

| File | Outcome | Pages | Chunks | ms |
|---|---|---|---|---|
| aurora_x200_installation_and_maintenance_guide.pdf | indexed | 6 | 18 | 1175 |
| aurora_x200_technical_specification.pdf | indexed | 6 | 15 | 1241 |
| halcyon_employee_handbook.pdf | indexed | 7 | 20 | 1160 |
| halcyon_information_security_policy.pdf | indexed | 6 | 18 | 1003 |
| halcyon_q2_fy2026_business_review.pdf | indexed | 4 | 11 | 720 |

Scores are uncalibrated proxies (see `evaluation/README.md`): retrieval metrics are page based, correctness is keyword coverage and abstention scoring is binary.
