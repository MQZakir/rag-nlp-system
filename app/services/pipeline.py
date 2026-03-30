"""
RAG Pipeline — orchestrates retrieval → reranking → generation.

This is the single entry point for all query handling. It:
1. Runs hybrid retrieval (FAISS + BM25 + RRF).
2. Optionally reranks via cross-encoder.
3. Feeds the top-K chunks into the LLM.
4. Tracks latency and updates in-memory metrics.

Caching
-------
If ``REDIS_URL`` is configured, query results are cached by a SHA-256
hash of (query, collection, top_k, use_reranker). Cache hits skip all
model inference and return immediately.

The pipeline is intentionally stateless between calls — all state lives
in the retriever's collection objects and the optional Redis cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import QueryResponse, SourceDocument
from app.services.generator import GeneratorService, get_generator
from app.services.reranker import RerankerService, get_reranker
from app.services.retriever import HybridRetriever, RetrievedChunk, get_retriever

log = get_logger(__name__)


@dataclass
class _Metrics:
    total_queries: int = 0
    total_ingestions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    _total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return round(self._total_latency_ms / self.total_queries, 1)

    def record_query(self, latency_ms: float, cache_hit: bool) -> None:
        self.total_queries += 1
        self._total_latency_ms += latency_ms
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1


_metrics = _Metrics()


def get_metrics() -> _Metrics:
    return _metrics


def _cache_key(query: str, collection: str, top_k: int, use_reranker: bool) -> str:
    payload = f"{query}|{collection}|{top_k}|{use_reranker}"
    return "rag:" + hashlib.sha256(payload.encode()).hexdigest()


def _get_redis():  # type: ignore[return]
    if not settings.redis_url:
        return None
    try:
        import redis

        return redis.from_url(settings.redis_url, decode_responses=True)
    except Exception:
        log.warning("redis unavailable, caching disabled")
        return None


_redis = _get_redis()


class RAGPipeline:
    """
    Stateless query pipeline.

    Instantiate once and reuse across all requests — the underlying
    retriever, reranker, and generator handle their own state.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: RerankerService,
        generator: GeneratorService,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator

    async def query(
        self,
        query: str,
        collection: str,
        top_k: int,
        use_reranker: bool,
    ) -> QueryResponse:
        t0 = time.perf_counter()
        cache_hit = False

        # ── Cache lookup ──────────────────────────────────────────────────────
        if _redis:
            key = _cache_key(query, collection, top_k, use_reranker)
            cached = _redis.get(key)
            if cached:
                log.debug("cache hit", collection=collection)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                _metrics.record_query(latency_ms, cache_hit=True)
                return QueryResponse(**json.loads(cached))

        # ── Retrieval ─────────────────────────────────────────────────────────
        candidates = self._retriever.search(
            collection=collection,
            query=query,
            top_k=settings.top_k_retrieval,
        )

        log.debug("retrieved candidates", count=len(candidates), collection=collection)

        # ── Reranking ────────────────────────────────────────────────────────
        if use_reranker and candidates:
            chunks = self._reranker.rerank(query, candidates, top_k=top_k)
            retrieval_method = "hybrid+reranked"
        else:
            chunks = candidates[:top_k]
            retrieval_method = "hybrid"

        # ── Generation ────────────────────────────────────────────────────────
        answer = await self._generator.agenerate(query, chunks)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "query complete",
            collection=collection,
            chunks_used=len(chunks),
            latency_ms=latency_ms,
            method=retrieval_method,
        )

        response = QueryResponse(
            answer=answer,
            sources=[
                SourceDocument(
                    content=c.content,
                    metadata=c.metadata,
                    score=round(c.score, 4),
                )
                for c in chunks
            ],
            retrieval_method=retrieval_method,
            collection=collection,
            latency_ms=latency_ms,
        )

        # ── Cache store ───────────────────────────────────────────────────────
        if _redis:
            _redis.setex(key, settings.cache_ttl_seconds, response.model_dump_json())

        _metrics.record_query(latency_ms, cache_hit=False)
        return response

    async def stream(
        self,
        query: str,
        collection: str,
        top_k: int,
        use_reranker: bool,
    ) -> AsyncIterator[str]:
        """Yield LLM token chunks as they arrive (SSE streaming)."""
        candidates = self._retriever.search(
            collection=collection,
            query=query,
            top_k=settings.top_k_retrieval,
        )

        if use_reranker and candidates:
            chunks = self._reranker.rerank(query, candidates, top_k=top_k)
        else:
            chunks = candidates[:top_k]

        async for token in self._generator.astream(query, chunks):
            yield token


def get_pipeline() -> RAGPipeline:
    return RAGPipeline(
        retriever=get_retriever(),
        reranker=get_reranker(),
        generator=get_generator(),
    )
