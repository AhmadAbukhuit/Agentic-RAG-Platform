# Enterprise Agentic RAG Reference Architecture & Template Blueprint

This document serves as the comprehensive engineering guide and template blueprint for building, scaling, and maintaining production-grade **Agentic Retrieval-Augmented Generation (Agentic RAG)** systems using **LangChain (v0.3)** and **LangGraph (v0.2)**.

---

## Table of Contents

1. [High-Level Architectural Mental Model](#1-high-level-architectural-mental-model)
2. [Complete Directory & Component Walkthrough](#2-complete-directory--component-walkthrough)
3. [State Management Principles (`app/state/`)](#3-state-management-principles-(`app/state/`))
4. [Agentic Patterns Implemented](#4-agentic-patterns-implemented)
   - [A. Corrective RAG (CRAG)](#a-corrective-rag-crag)
   - [B. Two-Stage Cross-Encoder Re-ranking](#b-two-stage-cross-encoder-re-ranking)
   - [C. Actor-Critic Reflection Loop](#c-actor-critic-reflection-loop)
   - [D. Distributed Semantic & Exact Redis Caching](#d-distributed-semantic--exact-redis-caching)
   - [E. Dual-Layer Memory Architecture](#e-dual-layer-memory-architecture)
5. [Data Ingestion Pipeline (`app/nodes/ingest_nodes.py`)](#5-data-ingestion-pipeline)
6. [Observability & Evaluation (`tests/`)](#6-observability--evaluation)
7. [Production Deployment & Containerization](#7-production-deployment--containerization)
8. [Checklist for Adapting this Template to New Projects](#8-checklist-for-adapting-this-template-to-new-projects)

---

## 1. High-Level Architectural Mental Model

Traditional RAG systems are **rigid, open-loop pipelines** (`User Query -> Retrieve top-K -> Generate Answer`). They suffer from three fatal production flaws:

1. **Context Noise / Hallucination**: Irrelevant documents mislead the LLM.
2. **False Assumptions**: If the retriever returns bad context, the LLM still attempts to answer, hallucinating facts.
3. **Inefficiency**: Re-running expensive LLM and vector calculations for identical or similar questions.

**Agentic RAG** transforms this into a **dynamic, closed-loop state graph**:

```text
                              [Incoming Query]
                 (question, user_id="user_123", thread_id="session_abc")
                                      │
                                      ▼
             [recall_memory] ──(Retrieves User Profile & Durable Facts)
                                      │
                                      ▼
                                [check_cache] ──(Hit < 5ms)──► [Instant Output]
                                      │
                                 (Cache Miss)
                                      ▼
                                [router_node] ──(Determines Datasource)
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
      [vector_search (Qdrant)]                          [web_search]
               │                                             │
               ▼                                             │
       [grade_documents] (CRAG)                              │
        ├── 'relevant'   ──► [rerank_documents] ◄────────────┘
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

---

## 2. Complete Directory & Component Walkthrough

```text
├── Dockerfile                  # Container definition with Python 3.12, unbuffered logging & PYTHONPATH
├── docker-compose.yaml         # Multi-container orchestration (FastAPI, Qdrant, Redis, MongoDB, Ollama)
├── langgraph.json              # LangGraph CLI and LangGraph Studio configuration
├── requirements.txt            # Strictly pinned dependencies (LangChain 0.3, LangGraph 0.2, Qdrant, Redis)
├── .env.example                # Canonical environment variable specification
├── README.md                   # Repository overview and quickstart instructions
├── docs/
│   └── ARCHITECTURE.md         # This authoritative template reference guide
│
├── app/                        # Main service application package
│   ├── main.py                 # FastAPI app entrypoint, CORS configuration, LangServe route mounting
│   │
│   ├── core/                   # Infrastructure configuration, model clients, and cache
│   │   ├── config.py           # Pydantic v2 BaseSettings with defaults and environment parsing
│   │   ├── llm.py              # LLM and Embedding factory functions with LRU caching
│   │   └── cache.py            # Redis distributed cache client with SHA-256 keys and local fallback
│   │
│   ├── state/                  # LangGraph state schemas (data contracts passed between nodes)
│   │   ├── schema.py           # AgentState: TypedDict schema for the master question-answering workflow
│   │   └── ingestion_schema.py # IngestionState: TypedDict schema for document parsing and indexing
│   │
│   ├── prompts/                # Externalized prompt templates (Versioned, testable, and clean)
│   │   ├── router_prompts.py   # Intent classification and datasource selection prompts
│   │   ├── generator_prompts.py# Synthesis, revision critique, and hallucination review prompts
│   │   └── crag_prompts.py     # Document relevance grading and query rewriting prompts
│   │
│   ├── nodes/                  # Pure, functional LangGraph node handlers
│   │   ├── cache_nodes.py      # check_cache (pre-execution bypass) and save_cache (post-approval)
│   │   ├── router.py           # route_question: Classifies intent into vectorstore vs web_search
│   │   ├── retriever.py        # query_qdrant_hybrid (BM25 + dense) and search_web
│   │   ├── reranker.py         # rerank_documents: Applies FlashRank cross-encoder filtering
│   │   ├── crag_nodes.py       # grade_documents (relevance grading) and rewrite_query (query optimizer)
│   │   ├── generator.py        # draft_response (synthesis/revision) and review_response (hallucination review)
│   │   ├── memory_nodes.py     # recall_memory (profile lookup) and persist_memory (facts & messages)
│   │   └── ingest_nodes.py     # extract_text_and_ocr, clean_ocr_text, chunk_document, insert_into_databases
│   │
│   ├── tools/                  # Reusable external integrations and databases
│   │   ├── vectorstore.py      # Qdrant client, dense+sparse collection setup, and hybrid indexing
│   │   ├── nosql_db.py         # MongoDB raw unchunked document persistence with fallback
│   │   ├── memory_store.py     # User long-term preference store (MongoDB + local fallback)
│   │   ├── sql_db.py           # Read-only analytical SQL query execution tool with injection guards
│   │   └── reranker.py         # FlashRank cross-encoder model manager and scoring engine
│   │
│   └── graphs/                 # LangGraph graph assembly and compilation
│       ├── main_graph.py       # Master orchestrator graph with checkpointer and conditional cycles
│       └── subgraphs/
│           ├── researcher.py   # Self-contained retrieval, CRAG grading, and re-ranking subgraph
│           └── ingestion.py    # Linear document ingestion and indexing pipeline subgraph
│
└── tests/                      # Automated test suite and LangSmith evaluation benchmarks
    ├── test_pipeline.py        # Comprehensive unit tests for nodes, routers, tools, and schemas
    ├── eval_dataset.py         # Golden benchmark dataset and LangSmith dataset sync utility
    ├── eval_rag.py             # LLM-as-a-judge evaluators (Faithfulness, Correctness, Routing)
    └── run_evaluation.py       # CLI runner for local offline and LangSmith cloud benchmark experiments
```

---

## 3. State Management Principles (`app/state/`)

LangGraph passes a centralized **State Dictionary** between all nodes. In this template, state design follows three production best practices:

### A. Non-Destructive / Total=False Typing

```python
class AgentState(TypedDict, total=False):
```

By setting `total=False`, individual nodes only return the specific fields they update (e.g. `return {"draft_answer": "..."}`). LangGraph merges these updates into the central state without requiring nodes to echo back unmodified fields.

### B. Controlled Reducers vs Replacement

* **Append-Only Reducer (`add_messages`)**:

  ```python
  messages: Annotated[List[BaseMessage], add_messages]
  ```

  Appends new turns to the chat history without losing prior turns.
* **Replacement Semantics (Standard `List[str]`)**:
  
  ```python
  documents: List[str]
  ```

  Allows the `rerank_documents` node to prune a list of 15 candidate passages down to the top 4 without accumulating or duplicating chunks.

---

## 4. Agentic Patterns Implemented

### A. Corrective RAG (CRAG)

* **Problem**: If internal search returns documents that don't answer the user's question, drafting immediately leads to hallucination.
* **Solution**: The [`grade_documents`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/nodes/crag_nodes.py#L18) node inspects retrieved chunks:
  * **Relevant**: Proceeds to re-ranking.
  * **Irrelevant**: Automatically triggers [`rewrite_query`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/nodes/crag_nodes.py#L52) to optimize the query for search engines, falls back to `search_web`, and clears stale internal chunks.

### B. Two-Stage Cross-Encoder Re-ranking

* **Stage 1 (Bi-Encoder Retrieval)**: Qdrant searches across dense embeddings and BM25 sparse tokens, fetching a high-recall pool of candidates ($K=15$).
* **Stage 2 (Cross-Encoder Re-ranking)**: FlashRank passes each `(query, document)` pair through an ONNX model (`ms-marco-TinyBERT-L-2-v2`), evaluating full self-attention interactions between query and document words. Only the top 4 chunks reach the synthesis prompt.

### C. Actor-Critic Reflection Loop

* **Drafter (Actor)** generates an initial answer from context.
* **Reviewer (Critic)** evaluates:
  1. *Faithfulness*: Does the draft make any claims unsupported by context?
  2. *Relevance*: Does it answer the prompt?
* **Conditional Cycle**: If `review_status == "revision_needed"` and `retry_count < max_retries`, the graph routes back to `drafter` with the critique, allowing the agent to self-correct before the user ever sees the response.

### D. Distributed Semantic & Exact Redis Caching

* **Pre-Check Bypass**: When a request enters, [`check_cache`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/nodes/cache_nodes.py#L5) queries Redis with `SHA256(normalized_query)`.
* If a cache hit occurs, the entire agent pipeline (LLMs, vectors, routers, re-rankers) is bypassed, returning validated answers in $< 5\text{ ms}$.
* On cache misses, [`save_cache`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/nodes/cache_nodes.py#L25) persists only approved answers after reviewer sign-off with configurable TTL (default: 24h).

### E. Dual-Layer Memory Architecture

* **Short-Term Memory**: Conversation history maintained across multi-turn sessions using `thread_id` and the LangGraph checkpointer (`MemorySaver` / `AsyncPostgresSaver`).
* **Long-Term Memory**: Durable user facts (e.g. preferences, department, role) stored across sessions in MongoDB (`user_memories`). Injected into the drafter prompt on future interactions.

---

## 5. Data Ingestion Pipeline (`app/nodes/ingest_nodes.py`)

Ingestion is separated into its own compiled subgraph ([`app/graphs/subgraphs/ingestion.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/graphs/subgraphs/ingestion.py)):

1. **Hybrid Extraction**: Native digital PDF text is extracted with PyMuPDF in $< 0.1\text{ s}$ per page. If a page has little/no text (scanned image), the pipeline lazily calls Ollama's vision model (`deepseek-ocr`).
2. **Sanitization**: Regular expressions strip OCR bounding-box artifacts and excessive newlines.
3. **Recursive Chunking**: Text is split into 1000-character chunks with a 200-character overlap using semantic separators (`["\n\n", "\n", ". ", " "]`).
4. **Dual Persistence**: Raw unchunked documents are archived in MongoDB with an assigned `mongo_id`. Chunks are enriched with `mongo_id` and indexed into Qdrant using both dense and BM25 sparse vectors.

---

## 6. Observability & Evaluation (`tests/`)

Production agentic systems require quantifiable benchmarks before deploying prompt or model updates.

* **Dataset Sync**: [`tests/eval_dataset.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/tests/eval_dataset.py) manages a golden benchmark dataset.
* **LLM Judges**: [`tests/eval_rag.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/tests/eval_rag.py) implements three automated judges:
  * `evaluate_faithfulness`: Penalizes hallucinations not grounded in context.
  * `evaluate_correctness`: Measures semantic similarity to ground truth.
  * `evaluate_routing_accuracy`: Checks intent classification accuracy.
* **Run Commands**:

  ```bash
  # Local benchmark execution (no API key required)
  python tests/run_evaluation.py --local

  # Cloud LangSmith benchmark execution
  python tests/run_evaluation.py --dataset Agentic-RAG-Benchmark-v1 --prefix eval-v1
  ```

---

## 7. Production Deployment & Containerization

### Dockerfile Highlights

* Based on `python:3.12-slim` for minimal image size and reduced vulnerability footprint.
* Sets `PYTHONPATH=/app` and `PYTHONUNBUFFERED=1` to guarantee clean module discovery and immediate log flushing.

### Docker Compose Stack

* **`rag-api`**: FastAPI / LangServe server exposing endpoints on port 8000.
* **`qdrant`**: Hybrid vector search engine with gRPC and REST APIs.
* **`redis`**: In-memory distributed cache with persistent volumes.
* **`mongodb`**: NoSQL document store for raw document bodies and long-term memory.
* **`ollama`**: Local inference engine running LLMs and embedding models.

---

## 8. Checklist for Adapting this Template to New Projects

When using this repository as a boilerplate for a new RAG project:

1. **Configure Vector Store**:
   * Change `qdrant_collection_name` in [`app/core/config.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/core/config.py).
   * Update vector dimensions in [`app/tools/vectorstore.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/tools/vectorstore.py) if switching embedding models (e.g. 768 for `nomic-embed-text`, 1536 for OpenAI `text-embedding-3-small`, 1024 for `bge-large`).
2. **Customize Prompts**:
   * Modify system prompts in [`app/prompts/generator_prompts.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/prompts/generator_prompts.py) to match your domain (medical, legal, financial, internal IT).
3. **Swap LLM Provider (Optional)**:
   * To use OpenAI, Anthropic, or Azure instead of Ollama, update [`app/core/llm.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/app/core/llm.py) to return `ChatOpenAI` or `ChatAnthropic`. The rest of the graph remains 100% identical.
4. **Extend Tools**:
   * Add custom domain APIs (e.g. CRM lookup, ticket creation, SQL reporting) into `app/tools/` and wire them into subgraphs.
5. **Run Golden Benchmark**:
   * Add 10–20 domain-specific questions to [`tests/eval_dataset.py`](file:///home/ahmad/LangChain-LangGraph-LangFlow-LangSmith/tests/eval_dataset.py) and execute `python tests/run_evaluation.py --local` to establish your baseline accuracy.
