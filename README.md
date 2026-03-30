# RAG-Powered NLP System

A production-grade Retrieval-Augmented Generation (RAG) pipeline with semantic search, hybrid retrieval, and contextual reranking — served via FastAPI and containerised with Docker.

---

## Overview

This system combines dense vector retrieval with BM25 sparse search (hybrid retrieval), applies cross-encoder reranking, and feeds the retrieved context into a generative model to answer queries grounded in your document corpus. It is designed to be modular, observable, and straightforward to extend.

```
Query
  │
  ▼
Embedding Model (HuggingFace sentence-transformers)
  │
  ├──► Dense Vector Search  (FAISS)
  └──► Sparse BM25 Search   (rank_bm25)
           │
           ▼
      Reciprocal Rank Fusion  ──► Reranker (cross-encoder)
                                         │
                                         ▼
                                  Top-K Chunks  ──► LLM (via LangChain)
                                                         │
                                                         ▼
                                                    Answer + Sources
```

---

## Features

- **Hybrid Retrieval** — reciprocal rank fusion of dense (FAISS) and sparse (BM25) results
- **Cross-Encoder Reranking** — scores retrieved chunks for contextual relevance before generation
- **Flexible Chunking** — sentence-aware recursive splitting with configurable overlap
- **Async FastAPI** — non-blocking endpoints, proper error handling, structured logging
- **Pydantic v2 Schemas** — validated request/response contracts throughout
- **Streaming Support** — token-by-token SSE streaming on the `/query/stream` endpoint
- **Multi-collection** — ingest into named collections, query across one or all
- **Health & Metrics** — `/health`, `/metrics` endpoints for operational visibility
- **Docker Compose** — one command to run the full stack (API + optional Redis cache)
- **Pytest Suite** — unit tests for chunking, retrieval, and reranking; integration tests for API endpoints

---

## Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` (BAAI/bge-base-en-v1.5) |
| Vector Store | FAISS (in-process, persistable to disk) |
| Sparse Search | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM Orchestration | LangChain |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Caching | Redis (optional) |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions |

---

## Quickstart

### 1. Clone & configure

```bash
git clone https://github.com/mqzakir/rag-nlp-system.git
cd rag-nlp-system
cp .env.example .env
# Edit .env — set your LLM provider key if using OpenAI/Anthropic,
# or point OLLAMA_BASE_URL at a local Ollama instance
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

### 3. Run locally (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

---

## Ingesting Documents

```bash
# Ingest a directory of .txt / .pdf files into a named collection
python scripts/ingest.py --source data/raw/ --collection my_docs

# Ingest a single file
python scripts/ingest.py --source data/raw/report.pdf --collection finance
```

The script handles loading, chunking, embedding, and persisting the FAISS index + BM25 state to `data/processed/<collection>/`.

---

## API Reference

### `POST /ingest`

Ingest documents programmatically via the API.

```json
{
  "texts": ["Full document text here..."],
  "metadatas": [{"source": "report.pdf", "page": 1}],
  "collection": "my_docs"
}
```

### `POST /query`

Query a collection with hybrid retrieval + reranking.

```json
{
  "query": "What were the main findings of the 2023 review?",
  "collection": "my_docs",
  "top_k": 5,
  "use_reranker": true,
  "stream": false
}
```

**Response:**

```json
{
  "answer": "The 2023 review identified three primary findings...",
  "sources": [
    {
      "content": "...chunk text...",
      "metadata": {"source": "report.pdf", "page": 4},
      "score": 0.94
    }
  ],
  "retrieval_method": "hybrid",
  "latency_ms": 312
}
```

### `GET /query/stream`

Server-sent events stream. Pass query params:  
`?query=...&collection=...&top_k=5`

### `GET /collections`

List all available collections with document counts.

### `DELETE /collections/{name}`

Remove a collection and its index from disk.

### `GET /health`

```json
{"status": "ok", "collections_loaded": 3, "uptime_seconds": 4821}
```

---

## Configuration

All settings live in `.env` and are validated by `app/core/config.py`.

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the chosen provider |
| `LLM_API_KEY` | — | API key (OpenAI or Anthropic) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL if using Ollama |
| `CHUNK_SIZE` | `512` | Target token count per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `HYBRID_ALPHA` | `0.6` | Weight of dense vs sparse (0=BM25 only, 1=dense only) |
| `TOP_K_RETRIEVAL` | `20` | Candidates before reranking |
| `TOP_K_FINAL` | `5` | Chunks passed to LLM |
| `REDIS_URL` | — | Enable query-result caching (optional) |
| `INDEX_DIR` | `data/processed` | Persistence root for FAISS + BM25 |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` |

---

## Project Layout

```
rag-nlp-system/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan
│   ├── api/
│   │   ├── routes_ingest.py     # /ingest endpoint
│   │   ├── routes_query.py      # /query + /query/stream endpoints
│   │   └── routes_admin.py      # /collections, /health, /metrics
│   ├── core/
│   │   ├── config.py            # Pydantic settings
│   │   └── logging.py           # Structured JSON logger
│   ├── models/
│   │   └── schemas.py           # Request/response Pydantic models
│   └── services/
│       ├── chunker.py           # Recursive sentence-aware splitter
│       ├── embedder.py          # HuggingFace embedding wrapper
│       ├── retriever.py         # Hybrid FAISS + BM25 retrieval
│       ├── reranker.py          # Cross-encoder reranking
│       ├── generator.py         # LangChain LLM + prompt chain
│       └── pipeline.py          # Orchestrates retrieval → rerank → generate
├── scripts/
│   └── ingest.py                # CLI ingestion tool
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_reranker.py
│   └── integration/
│       └── test_api.py
├── docker/
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── pyproject.toml
```

---

## Extending the System

**Swap the vector store** — the `retriever.py` interface (`index`, `search`) is designed to be backed by Chroma, Weaviate, or Qdrant with minimal changes. FAISS is the default for zero-dependency local use.

**Add a document loader** — `scripts/ingest.py` currently handles `.txt` and `.pdf` (via `pypdf`). Drop a new loader function into `app/utils/loaders.py` and register it by extension.

**Change the LLM** — set `LLM_PROVIDER=ollama` and point `OLLAMA_BASE_URL` at any Ollama instance to run fully locally with no API costs.

**Persistent caching** — set `REDIS_URL` and query results are cached by a SHA-256 hash of `(query, collection, top_k)`. Cache TTL is configurable via `CACHE_TTL_SECONDS`.

---

## Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

---

## CI

GitHub Actions runs on every push and pull request:

1. Lint with `ruff`
2. Type-check with `mypy`
3. Unit tests
4. Integration tests (spins up the API in-process via `httpx.AsyncClient`)

---