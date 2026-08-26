# Measured Performance

_Generated 2026-08-26 18:46 UTC — every number below is produced by `python -m evals.*` against the live index and LLM; regenerate to refresh._

- Golden retrieval set: 43 answerable questions + 5 unanswerable controls
- Grounding-judge set: 30 labeled claims

## Retrieval quality

| Variant | n | Recall@5 | Recall@10 | MRR | p50 latency | Multi-hop R@5 |
|---|---|---|---|---|---|---|
| dense | 43 | 86.0% | 90.7% | 0.737 | 368 ms | 100.0% |
| fulltext | 43 | 83.7% | 93.0% | 0.745 | 25 ms | 100.0% |
| graph-expand | 43 | 93.0% | 95.3% | 0.758 | 570 ms | 100.0% |
| hybrid | 43 | 93.0% | 95.3% | 0.758 | 479 ms | 100.0% |
| neo4j-dense | 43 | 86.0% | 90.7% | 0.737 | 543 ms | 90.9% |

## Grounding-judge self-accuracy

n = 30 labeled claims

| Metric | Value |
|---|---|
| Accuracy | 96.7% |
| Ungrounded precision | 100.0% |
| Ungrounded recall | 90.0% |
| Ungrounded F1 | 0.947 |
| Valid inferences wrongly flagged | 6.7% |

Missed hallucinations (unsupported accepted): 0; valid claims flagged as hallucinations: 1.

## End-to-end answer quality (LLM-judged)

| Variant | slice | n | avg score (0-2) | correct | grounded | p50 s |
|---|---|---|---|---|---|---|
| hybrid+kg | multi_hop | 11 | 1.64 | 72.7% | 100.0% | 136.5 |
| hybrid | multi_hop | 11 | 1.64 | 72.7% | 90.9% | 82.2 |

_Judge = the pipeline's own model scoring against golden reference answers; self-judge bias applies._

## End-to-end runs (telemetry)

| Variant | n | success | p50 s | p95 s | llm calls/query | tok in/out | $/query |
|---|---|---|---|---|---|---|---|
| baseline-dense | 1 | 100.0% | 121.30 | 121.30 | 8.0 | 6056/2683 | $0.0000 |
| hybrid | 12 | 75.0% | 86.10 | 161.07 | 7.8 | 4950/2123 | $0.0000 |
| hybrid+kg | 11 | 81.8% | 136.46 | 238.09 | 11.3 | 6435/2849 | $0.0000 |

Outcome distribution — not_useful: 4 · success: 19 · ungrounded: 1
