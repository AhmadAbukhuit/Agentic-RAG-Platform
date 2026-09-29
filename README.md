# Agentic RAG Development Platform

A production-ready, modular Agentic RAG and Multi-Agent Orchestration service built with **LangChain (v0.3)**, **LangGraph (v0.2)**, **Qdrant**, **Redis**, **MongoDB**, **Ollama**, and **LangSmith**.

---

## 🌟 Key Architecture & Features

```text
                                      [User Query]
                                           │
                                           ▼
                 [recall_memory] ──(Durable User Preferences Recalled)
                                           │
                                           ▼
                                    [check_cache] ──(Hit < 5ms)──► [Instant Output]
                                           │
                                      (Cache Miss)
                                           ▼
                                     [router_node]
                                           │
               ┌───────────────────────────┴───────────────────────────┐
               ▼                                                       ▼
      [vector_search (Qdrant)]                                    [web_search]
               │                                                       │
               ▼                                                       │
       [grade_documents] (CRAG)                                        │
        ├── 'relevant'   ──► [rerank_documents] ◄──────────────────────┘
        └── 'irrelevant' ──► [rewrite_query] ──► [web_search] ──► [rerank_documents]
                                                                        │
                                                                        ▼
                                                             ┌───► [drafter_node]
                                                             │          │
                                                             │          ▼
                                                             │   [reviewer_node]
                                                             │          │
                                                             └───(revision_needed)
                                                                        │ (approved)
                                                                        ▼
                                                                   [save_cache]
                                                                        │
                                                                        ▼
                                                                 [persist_memory]
                                                                        │
                                                                        ▼
                                                                      [END]
```

1. **Two-Stage Retrieval & FlashRank Re-ranking**:
   * Initial high-recall candidate retrieval from Qdrant hybrid index (`k=15`).
   * Local CPU cross-encoder re-ranking via `FlashRank` (`ms-marco-TinyBERT-L-2-v2`) filtering to the top 4 highest-scoring passages.
2. **Corrective RAG (CRAG)**:
   * Document relevance grading node evaluates context before drafting.
   * If retrieved context is insufficient or irrelevant, it triggers automatic query rewriting and external web search fallback.
3. **Actor-Critic Reflection Loop**:
   * Evaluator grades draft response for hallucinations and factual support.
   * If critique is triggered, it re-prompts the drafter with targeted revision instructions (bounded by `max_retries`).
4. **Distributed Redis Caching**:
   * SHA-256 hashed queries cached in Redis with configurable TTL (default: 24h).
   * Exact-match queries bypass all LLM and vector inference, returning in `< 5ms`.
5. **Dual-Layer Memory**:
   * **Short-Term Memory**: Conversation turns automatically accumulated via `messages` and LangGraph checkpointer `thread_id`.
   * **Long-Term Memory**: Cross-session user preferences and facts persisted in MongoDB (collection: `user_memories`).
6. **Multimodal Ingestion Pipeline**:
   * PDF text extraction via PyMuPDF with lazy fallback to Ollama vision OCR (`deepseek-ocr`) for scanned pages.
   * Chunks embedded and indexed into Qdrant natively with BM25 sparse and dense representations.
7. **LangSmith Automated Evaluation Suite**:
   * Cloud and local benchmarking harness measuring **Faithfulness**, **Correctness**, and **Routing Accuracy**.

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env and supply your LANGCHAIN_API_KEY (optional, enables live LangSmith cloud tracing)
```

### 2. Launch with Docker Compose

```bash
docker compose up --build
```

* **Swagger API Documentation**: `http://localhost:8000/docs`
* **Agent Playground (LangServe)**: `http://localhost:8000/agents/playground/`
* **Ingestion Playground**: `http://localhost:8000/ingestion/playground/`
* **Health Check**: `http://localhost:8000/health`

---

## 🧪 Testing & LangSmith Evaluation

### Run Unit Tests

```bash
pytest tests/test_pipeline.py -v
```

### Run LangSmith Benchmarks

Run against your live LangSmith project:

```bash
python tests/run_evaluation.py --dataset Agentic-RAG-Benchmark-v1 --prefix my-experiment
```

Run in local offline mode (no API key required):

```bash
python tests/run_evaluation.py --local
```

Example local evaluation output:

```text
======================================================================
🚀 RUNNING LOCAL AGENTIC RAG EVALUATION BENCHMARK
======================================================================

[1/4] Evaluating: 'What is the policy on annual leave and PTO rollover...'
[2/4] Evaluating: 'What are the latest updates and breaking changes in...'
[3/4] Evaluating: 'How does the company reimburse personal travel expen...'
[4/4] Evaluating: 'How does Qdrant hybrid retrieval combine dense sem...'

---------------------------------------------------------------------------
#   | Question                               | Route       | Faith | Corr  | Lat   
---------------------------------------------------------------------------
1   | What is the policy on annual leave...  | vectorstore | 1.00  | 0.95  | 1.42s 
2   | What are the latest updates and br...  | web_search  | 1.00  | 0.88  | 0.98s 
3   | How does the company reimburse per...  | vectorstore | 1.00  | 0.92  | 1.15s 
4   | How does Qdrant hybrid retrieval c...  | vectorstore | 1.00  | 0.96  | 1.34s 
---------------------------------------------------------------------------

📊 OVERALL BENCHMARK METRICS:
  • Average Faithfulness (Anti-Hallucination): 100.0%
  • Average Correctness (Ground Truth Match):   92.8%
  • Routing Classification Accuracy:           100.0%
======================================================================
```
