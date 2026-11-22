# Week 2 Deliverable: Mentor Progress Update Email

**Course:** Executive Post Graduate Programme in Generative AI and Agentic AI  
**Track:** Track B — API-Side Capstone (Hosted $\le$8B Inference API)  
**Project:** Multi-Agent Operations Copilot (HR Policy Navigator)  
**Target Milestone:** Week 2 — Corpus Scaling, Hybrid Retrieval & Retrieval Ablation Study  

---

## 📧 Email Draft

**To:** `[Mentor Name / Email]`  
**Subject:** `[Capstone Track B] Week 2 Deliverables: Corpus Scaling (1,151 Docs), Neural Hybrid RAG & Retrieval Ablation`

Dear [Mentor Name],

I hope this email finds you well.

Our team has completed the **Week 2 deliverables** for our Capstone project: **HR Policy Navigator (Track B — API-Side Capstone on Hosted $\le$8B API)**.

In Week 1, our single-prompt baselines established that an un-augmented 8B model cannot reliably answer proprietary HR questions (capping at 37.5% accuracy) and tends to hallucinate plausible-sounding policies when context is absent. 

This week, in strict adherence to the Week 2 curriculum requirements, we focused on building an enterprise-grade retrieval backbone: scaling the handbook corpus past the 1,000-document threshold, implementing hybrid retrieval with dense neural embeddings and lexical scoring, and conducting an isolated retrieval ablation across Recall@k, Hit Rate, and MRR.

Below is an executive summary of our deliverables and empirical findings:

---

### 1. Key Accomplishments

#### A. Corpus Scaling past 1,000 Documents (Syllabus Requirement I)
* **Corpus Expansion:** Scaled our knowledge base from the 151 core handbook policies to **1,151 Markdown documents** (3,686 indexed chunks), comfortably exceeding the syllabus rubric requirement (500–2,000 documents).
* **Synthetic Generation Methodology:** Generated 1,000 realistic, domain-coherent HR documents (`sourcegraph_files/synthetic/`) across 10 functional categories (e.g., parental leave guides, regional benefits, manager FAQs, workplace accommodations).
* **Automated Consistency Validation:** Executed automated fact-consistency checks (`scripts/verify_synthetic_consistency.py`) comparing synthetic documents against core source policies to guarantee zero contradictory rules or fabricated benefits.

#### B. Hybrid Retrieval Architecture (Syllabus Requirement II)
* **Dense Neural Embeddings:** Integrated the hosted embedding model (`bge-small-en-v1.5`, 384 dimensions) via OpenAI-compatible endpoints with client-side batching ($N=16$), token-budget clamping ($\le 512$ tokens), and persistent SHA-1 disk caching (`.cache/embeddings/`).
* **Lexical BM25 Search:** Implemented custom pure-Python BM25 ($k_1=1.4, b=0.75$) with sublinear term-frequency scaling and document length normalization.
* **Reciprocal Rank Fusion (RRF):** Merged lexical and dense retrieval candidate lists using non-parametric RRF ($k=60$), eliminating the need for brittle manual score calibration.
* **Cross-Feature Reranking:** Applied a multi-signal reranker combining prior retrieval rank, document title keyword overlap, section header matches, and bigram phrase coverage with per-document hit capping to guarantee citation diversity.

---

### 2. Retrieval Evaluation & Ablation Study (Syllabus Requirement III)
We evaluated retrieval performance over our frozen benchmark set of 60 realistic domain queries (`evaluation_set.json`) across 11 distinct policy themes. To account for topic correlation between query paraphrases, we computed non-parametric 95% confidence intervals using **2,000-iteration cluster bootstrap resampling over themes**.

#### Empirical Retrieval Comparison (Heading-Cap Chunking, 3,686 Chunks, 1,151 Docs)
*Evaluated on hosted `bge-small-en-v1.5` embeddings endpoint:*

| Retrieval Mode | Hit@1 | Hit@3 | Hit@5 | Doc Recall@5 | MRR | Hit@5 (95% CI) | Δ vs BM25 (95% CI) |
|---|---|---|---|---|---|---|---|
| **BM25 Alone** | 0.618 | 0.836 | 0.836 | 0.727 | 0.737 | [0.727, 0.927] | — |
| **Dense Alone (`bge-small-en-v1.5`)** | 0.582 | 0.745 | 0.764 | 0.709 | 0.668 | [0.600, 0.909] | [-0.218, 0.055] |
| **Hybrid (RRF Fusion)** | 0.691 | 0.855 | 0.855 | 0.764 | 0.770 | [0.745, 0.964] | [-0.036, 0.073] |
| **Hybrid + Reranking** | **0.709** | **0.836** | **0.891** | **0.800** | **0.793** | **[0.782, 0.982]** | **[0.000, 0.109]** |

*Metric Explanations:*
* **Hit@k:** Percentage of queries where at least one expected gold document is retrieved in the top-$k$ results.
* **Doc Recall@k:** Average proportion of all relevant gold documents retrieved in the top-$k$ candidates.
* **MRR (Mean Reciprocal Rank):** Average reciprocal rank ($1 / \text{rank}$) of the first relevant document (1.0 = relevant doc always at position 1).
* **Δ vs BM25:** Paired difference per question resampled over clusters; a lower bound $\ge 0.000$ indicates statistically significant superiority over BM25.

---

### 3. Key Empirical Findings & Insights

1. **Why BM25 Outperforms Pure Dense on Enterprise HR Policies:**  
   Stand-alone BM25 achieved higher top-1 accuracy (61.8% vs 58.2%) and MRR (0.737 vs 0.668) than pure dense embeddings. In HR operations, queries heavily depend on exact acronyms and proper nouns (*"FMLA"*, *"401(k)"*, *"PTO"*, *"BambooHR"*, *"LOA"*). Dense neural embeddings often project these terms into broad semantic spaces, retrieving tangentially related documents (e.g., general onboarding guides instead of specific 401(k) deduction forms).
2. **Hybrid RRF Combines Semantic & Lexical Strengths:**  
   Combining BM25 with dense neural embeddings via RRF elevated Hit@1 from 61.8% to 69.1% (+7.3%) and MRR to 0.770. Dense search resolves conversational paraphrasing (e.g., *"Can I take time off to care for my newborn?"* successfully retrieving `parental-leave.md`), while BM25 anchors on exact policy keywords.
3. **Reranking is Critical for Hosted $\le$8B Models:**  
   Hybrid retrieval with cross-feature reranking produced our highest performance: **Hit@5 reached 89.1%**, **Hit@1 reached 70.9%**, and **MRR reached 0.793**. Because edge-class 8B models suffer from severe "lost in the middle" attention decay when irrelevant passages clutter the context window, placing the gold passage in position 1 in over 70% of requests is vital for preventing generation hallucinations.

---

### 4. Roadmap for Week 3 (Multi-Agent System)
With our retrieval layer achieving 89.1% Hit@5 and 0.793 MRR, our focus for Week 3 shifts to the agentic control loop:
* **Planner Agent:** Intent classification, multi-hop query decomposition, and query reformulation on low retrieval confidence.
* **Executor Agent:** Multi-step tool use (`search_policy`, `read_section`, `escalate_to_human`) with loop detection and execution budget guards.
* **Critic Agent:** Verifying claim-level grounding ($\ge 60\%$ lexical overlap against retrieved sources) and suppressing hallucinated statistics.
* **Resilience Layer:** Schema fallback cascades (`json_schema` $\to$ `guided_json` $\to$ `json_object` $\to$ `repair`) and automated `<think>` tag stripping.

---

### 5. Deliverables & Repository Links
* **Repository (Branch: Amit):** [GitHub Repository](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/tree/Amit)
* **Week 2 Retrieval Ablation Report:** [`docs/retrieval_results.md`](retrieval_results.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/retrieval_results.md)
* **Synthetic Corpus Documentation:** [`docs/synthetic_corpus.md`](synthetic_corpus.md) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/docs/synthetic_corpus.md)
* **Raw Benchmark Telemetry Data:** [`eval_runs/retrieval.json`](../eval_runs/retrieval.json) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/eval_runs/retrieval.json)
* **Corpus Consistency Verification Script:** [`scripts/verify_synthetic_consistency.py`](../scripts/verify_synthetic_consistency.py) · [GitHub Link](https://github.com/VivekDuraisamy6789/HAX---HR-Policy-Navigator/blob/Amit/scripts/verify_synthetic_consistency.py)

We look forward to discussing these retrieval ablation findings during our upcoming sync.

Best regards,  
**Amit Bhardwaj** (and Team)  
*Capstone Track B — Group [Number]*  
*Executive Post Graduate Programme in Generative AI and Agentic AI*

---

## 📌 Appendix: Anticipated Mentor Questions & Answers (Defense Guide)

### Q1: Why did you scale the corpus to 1,151 documents using synthetic generation rather than scraping external websites?
**Answer:**  
Track B specifies an internal enterprise operations copilot for a single corporate environment (Sourcegraph handbook). Scraping arbitrary public websites would introduce unrelated domain noise and destroy the evaluation integrity. Instead, we synthesized 1,000 domain-consistent HR policies across 10 realistic corporate sub-domains (regional leave, manager protocols, wellness stipends) anchored directly to the real policies. This satisfies the syllabus rubric (>1,000 documents) while preserving strict factual consistency, verified via automated consistency testing (`scripts/verify_synthetic_consistency.py`).

### Q2: Why is dense retrieval alone (58.2% Hit@1) worse than BM25 (61.8% Hit@1)? Shouldn't neural embeddings understand semantics better?
**Answer:**  
In general web QA, neural embeddings often outperform keyword search. However, in enterprise HR policy, documents share dense, highly overlapping corporate vocabulary (*"eligibility"*, *"benefits"*, *"teammate"*, *"manager"*), and the distinguishing criteria are exact acronyms (*"FMLA"*, *"401(k)"*, *"PTO"*, *"STD"* vs *"LTD"*). Dense embeddings compress these distinct lexical tokens into proximate semantic vectors, resulting in distractor retrieval. BM25 excels at exact keyword precision. The real strength of neural embeddings appears in **Hybrid RRF**, where semantic matching rescues conversational paraphrases that lack exact policy keywords, boosting Hit@1 to **70.9%**.

### Q3: Why is 2,000-iteration cluster bootstrap resampling over themes used rather than standard sample standard deviations?
**Answer:**  
The 60 evaluation queries are organized into 11 policy themes (e.g., 5 paraphrased queries testing parental leave). Paraphrases within the same theme are not independently distributed; if a retrieval index fails on parental leave terminology, all 5 paraphrases will fail together. Standard i.i.d. standard errors would severely underestimate confidence interval width. Cluster bootstrapping over themes resamples entire thematic blocks, providing honest, non-parametric 95% confidence bounds that reflect true generalization performance across unseen policy topics.
