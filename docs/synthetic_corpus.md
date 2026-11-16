# Synthetic corpus (1,000 documents)

After removing the 146 non-HR files, the real corpus is 151 documents (95 core HR, 56 HR-adjacent), below the brief's 500–2,000 range. To reach scale, `scripts/generate_synthetic_corpus.py` writes **1,000 synthetic HR / HR-adjacent documents** to `sourcegraph_files/synthetic/`.

**Disclosure:** these are generated from templates, not real Sourcegraph policy. Every file carries `synthetic: true` and a banner. The technical report must say the corpus is 151 real + 1,000 synthetic documents and must report real-only numbers alongside combined numbers.

## Families (1,000 files)
region guides (200: 40 countries × leave/benefits/expenses/time off/remote), onboarding (100), expense categories (60), parental-leave scenarios (120), manager guides (100), learning tracks (120), offboarding (50), wellbeing (80), recruiting stages (112), FAQs (58). Counts are trimmed round-robin to exactly 1,000.

## Consistency design
- One fact table in the generator. **REAL facts** (checked against the real docs by the verifier): 16 weeks paid parental leave within 12 months; receipts over $25; reimbursement in 7-10 days via Airbase; $100/month wellness stipend; $1000/year education budget; approval for software over $500/year; 100% premiums for US-based full-time employees; Leave Case Manager contact in 1-2 business days.
- **SYNTH facts** (invented process values, defined once, used identically everywhere): check-ins at day 30/60/90, 8-week buddy, 5-day offer response, 3-day equipment shipping, 7-day exit survey, 48h manager response, 24h scorecards.
- Synthetic docs never invent statutory or legal figures; country docs defer to the People Team.
- `python scripts/verify_synthetic_consistency.py` → checks real facts exist in the real corpus, every number in a synthetic doc matches the table, links resolve, no duplicate titles/bodies. Last run: OK (1000 docs, 1000 unique titles/bodies).

## Retrieval impact (heading_cap, LSA dense stand-in, lexical reranker, 55-question set; gold docs are real)
| corpus | mode | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| real only (1,695 chunks) | bm25 | 0.67 | 0.87 | 0.77 |
| | hybrid_rerank | 0.73 | 0.89 | 0.80 |
| real + synthetic (3,686 chunks) | bm25 | 0.62 | 0.84 | 0.74 |
| | dense (LSA) | 0.49 | 0.62 | 0.55 |
| | hybrid | 0.55 | 0.82 | 0.66 |
| | hybrid_rerank | 0.67 | 0.89 | 0.77 |

Synthetic near-duplicates act as distractors: they lower BM25 and hybrid, and reranking recovers most of the loss. Differences in hit@5 are within bootstrap CIs except dense (LSA), which is clearly worse. Real-only vs combined runs: `HR_EXCLUDE=synthetic/` restricts any script to real documents.
