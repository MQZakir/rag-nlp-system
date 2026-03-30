"""
Integration tests — spins up the full FastAPI app in-process via HTTPX.

Heavy model dependencies (embedder, reranker, generator) are mocked so
tests run fast without GPU or API keys. What's being tested:
  - Request validation and HTTP status codes
  - Router logic and error handling
  - End-to-end ingest → query flow with mocked inference
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Patch settings before the app is imported
_tmpdir = tempfile.mkdtemp()


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    """Override index_dir to a temp directory for each test."""
    monkeypatch.setenv("INDEX_DIR", _tmpdir)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")  # No API key required
    monkeypatch.setenv("LLM_MODEL", "llama3")


def _mock_embedder(dim: int = 16):
    svc = MagicMock()
    svc.embedding_dim = dim

    def encode_docs(texts):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def encode_query(q):
        vec = np.random.randn(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    svc.encode_documents.side_effect = encode_docs
    svc.encode_query.side_effect = encode_query
    return svc


def _mock_reranker():
    svc = MagicMock()

    def rerank(query, chunks, top_k=None):
        from app.services.retriever import RetrievedChunk

        result = [
            RetrievedChunk(
                content=c.content,
                metadata=c.metadata,
                score=0.9 - i * 0.1,
                rank=i,
            )
            for i, c in enumerate(chunks[: top_k or len(chunks)])
        ]
        return result

    svc.rerank.side_effect = rerank
    return svc


def _mock_generator():
    svc = MagicMock()
    svc.agenerate = AsyncMock(return_value="This is the generated answer.")

    async def astream(query, chunks):
        for token in ["This ", "is ", "streamed."]:
            yield token

    svc.astream = astream
    return svc


@pytest.fixture
def mock_services():
    """Patch all model-heavy services for the test session."""
    embedder = _mock_embedder()
    reranker = _mock_reranker()
    generator = _mock_generator()

    with (
        patch("app.services.retriever.get_embedding_service", return_value=embedder),
        patch("app.services.reranker.get_reranker", return_value=reranker),
        patch("app.services.generator.get_generator", return_value=generator),
        patch("app.services.pipeline.get_reranker", return_value=reranker),
        patch("app.services.pipeline.get_generator", return_value=generator),
    ):
        yield {"embedder": embedder, "reranker": reranker, "generator": generator}


@pytest_asyncio.fixture
async def client(mock_services):
    """Async HTTP client pointed at the test app instance."""
    # Import app after patching to ensure mocks are applied
    from app.main import create_app

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:
    async def test_health_ok(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "collections_loaded" in body
        assert "uptime_seconds" in body

    async def test_health_returns_model_info(self, client: AsyncClient):
        resp = await client.get("/health")
        body = resp.json()
        assert "embedding_model" in body
        assert "llm_provider" in body


# ── Collections ────────────────────────────────────────────────────────────────

class TestCollections:
    async def test_list_collections_empty(self, client: AsyncClient):
        resp = await client.get("/collections")
        assert resp.status_code == 200
        body = resp.json()
        assert "collections" in body
        assert isinstance(body["collections"], list)

    async def test_delete_nonexistent_collection_404(self, client: AsyncClient):
        resp = await client.delete("/collections/does_not_exist")
        assert resp.status_code == 404


# ── Ingestion ─────────────────────────────────────────────────────────────────

class TestIngest:
    async def test_ingest_single_document(self, client: AsyncClient, mock_services):
        resp = await client.post(
            "/ingest",
            json={
                "texts": ["The quick brown fox jumps over the lazy dog. " * 10],
                "collection": "test_col",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["collection"] == "test_col"
        assert body["documents_ingested"] == 1
        assert body["chunks_created"] >= 1

    async def test_ingest_multiple_documents(self, client: AsyncClient, mock_services):
        resp = await client.post(
            "/ingest",
            json={
                "texts": ["Document one " * 20, "Document two " * 20],
                "collection": "multi_col",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["documents_ingested"] == 2

    async def test_ingest_with_metadata(self, client: AsyncClient, mock_services):
        resp = await client.post(
            "/ingest",
            json={
                "texts": ["Some text here."],
                "metadatas": [{"source": "file.pdf", "page": 1}],
                "collection": "meta_col",
            },
        )
        assert resp.status_code == 201

    async def test_ingest_metadata_length_mismatch_422(self, client: AsyncClient):
        resp = await client.post(
            "/ingest",
            json={
                "texts": ["text one", "text two"],
                "metadatas": [{"source": "only one"}],
                "collection": "bad_col",
            },
        )
        assert resp.status_code == 422

    async def test_ingest_empty_texts_422(self, client: AsyncClient):
        resp = await client.post(
            "/ingest",
            json={"texts": [], "collection": "empty_col"},
        )
        assert resp.status_code == 422

    async def test_ingest_invalid_collection_name_422(self, client: AsyncClient):
        resp = await client.post(
            "/ingest",
            json={"texts": ["some text"], "collection": "bad name!"},
        )
        assert resp.status_code == 422


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQuery:
    async def _ingest(self, client: AsyncClient, collection: str = "q_col"):
        await client.post(
            "/ingest",
            json={
                "texts": [
                    "Retrieval-augmented generation combines dense retrieval with language models. " * 5,
                    "FAISS is a library for efficient similarity search developed by Meta. " * 5,
                    "BM25 is a ranking function used in information retrieval. " * 5,
                ],
                "collection": collection,
            },
        )

    async def test_query_returns_answer(self, client: AsyncClient, mock_services):
        await self._ingest(client)
        resp = await client.post(
            "/query",
            json={"query": "What is RAG?", "collection": "q_col"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    async def test_query_returns_sources(self, client: AsyncClient, mock_services):
        await self._ingest(client)
        resp = await client.post(
            "/query",
            json={"query": "What is FAISS?", "collection": "q_col"},
        )
        body = resp.json()
        assert "sources" in body
        assert isinstance(body["sources"], list)
        assert len(body["sources"]) > 0

    async def test_query_latency_field_present(self, client: AsyncClient, mock_services):
        await self._ingest(client)
        resp = await client.post(
            "/query",
            json={"query": "BM25", "collection": "q_col"},
        )
        body = resp.json()
        assert "latency_ms" in body
        assert isinstance(body["latency_ms"], int)

    async def test_query_nonexistent_collection_404(self, client: AsyncClient):
        resp = await client.post(
            "/query",
            json={"query": "anything", "collection": "no_such_col"},
        )
        assert resp.status_code == 404

    async def test_query_empty_query_422(self, client: AsyncClient):
        resp = await client.post(
            "/query",
            json={"query": "", "collection": "q_col"},
        )
        assert resp.status_code == 422

    async def test_query_retrieval_method_field(self, client: AsyncClient, mock_services):
        await self._ingest(client)
        resp = await client.post(
            "/query",
            json={"query": "retrieval", "collection": "q_col", "use_reranker": True},
        )
        body = resp.json()
        assert body["retrieval_method"] in ("hybrid", "hybrid+reranked")

    async def test_stream_endpoint_returns_sse(self, client: AsyncClient, mock_services):
        await self._ingest(client)
        resp = await client.get(
            "/query/stream",
            params={"query": "What is BM25?", "collection": "q_col"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ── Collection lifecycle ───────────────────────────────────────────────────────

class TestCollectionLifecycle:
    async def test_ingest_then_list_then_delete(self, client: AsyncClient, mock_services):
        # Ingest
        await client.post(
            "/ingest",
            json={"texts": ["Lifecycle test document " * 10], "collection": "lifecycle"},
        )

        # Appears in list
        resp = await client.get("/collections")
        names = [c["name"] for c in resp.json()["collections"]]
        assert "lifecycle" in names

        # Delete
        resp = await client.delete("/collections/lifecycle")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # No longer in list
        resp = await client.get("/collections")
        names = [c["name"] for c in resp.json()["collections"]]
        assert "lifecycle" not in names

        # Query now returns 404
        resp = await client.post(
            "/query",
            json={"query": "anything", "collection": "lifecycle"},
        )
        assert resp.status_code == 404
