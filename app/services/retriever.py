"""
Hybrid retriever: dense FAISS search + sparse BM25, fused via Reciprocal Rank Fusion.

Architecture
------------
Each collection maintains two indexes side-by-side:

    FAISS IndexFlatIP
        Fast inner-product search over L2-normalised embeddings.
        Equivalent to cosine similarity. Suitable for up to ~1M vectors
        before requiring IVF quantisation.

    BM25Okapi (rank_bm25)
        Classic lexical retrieval over tokenised chunks.
        Catches exact-match / keyword queries that dense models miss.

Fusion
------
Both retrievers return ordered result lists. We apply Reciprocal Rank Fusion:

    RRF_score(d) = Σ  1 / (k + rank_i(d))

where k=60 is the standard smoothing constant and i iterates over rankers.
The fused list is then re-ranked by a cross-encoder (see reranker.py).

Persistence
-----------
Each collection is persisted to disk as:
    <index_dir>/<collection>/
        faiss.index      — FAISS binary index
        metadata.pkl     — list[dict] of chunk metadata + content
        bm25.pkl         — serialised BM25Okapi object

Collections are loaded lazily on first access and cached in memory.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger
from app.services.chunker import Chunk
from app.services.embedder import get_embedding_service

log = get_logger(__name__)

_RRF_K = 60  # Standard RRF smoothing constant


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    score: float  # Normalised 0–1; higher is more relevant
    rank: int


@dataclass
class _CollectionState:
    """In-memory state for a single collection."""

    faiss_index: faiss.IndexFlatIP
    bm25: BM25Okapi
    chunks: list[Chunk]  # Parallel to faiss_index rows
    tokenised_corpus: list[list[str]]  # Parallel to chunks, for BM25


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenisation for BM25."""
    return text.lower().split()


def _normalise_scores(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Min-max normalise to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.ones_like(scores)
    return (scores - mn) / (mx - mn)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = _RRF_K,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists of chunk indices using RRF.

    Returns a list of (chunk_idx, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Manages all collections and exposes a unified search interface.

    Thread safety: FAISS IndexFlatIP is read-safe for concurrent queries.
    Writes (index/add) should happen during ingestion, not during serving.
    """

    def __init__(self, index_dir: Path, alpha: float = 0.6) -> None:
        self.index_dir = index_dir
        self.alpha = alpha  # Weight of dense results in fusion (0=BM25, 1=dense)
        self._collections: dict[str, _CollectionState] = {}

    # ── Collection management ────────────────────────────────────────────────

    def collection_names(self) -> list[str]:
        """Names of all collections currently loaded in memory."""
        return list(self._collections.keys())

    def collection_info(self, name: str) -> dict[str, Any]:
        state = self._collections[name]
        path = self.index_dir / name
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        return {
            "name": name,
            "chunk_count": len(state.chunks),
            "document_count": len({c.metadata.get("doc_index") for c in state.chunks}),
            "index_size_mb": round(size_mb, 3),
        }

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(self, collection: str, chunks: list[Chunk]) -> None:
        """Embed and index a list of Chunks into the named collection."""
        if not chunks:
            return

        embedder = get_embedding_service()
        texts = [c.content for c in chunks]
        embeddings = embedder.encode_documents(texts)

        if collection not in self._collections:
            log.info("creating new collection", collection=collection)
            index = faiss.IndexFlatIP(embedder.embedding_dim)
            state = _CollectionState(
                faiss_index=index,
                bm25=BM25Okapi([[]]),  # placeholder, rebuilt below
                chunks=[],
                tokenised_corpus=[],
            )
            self._collections[collection] = state
        else:
            state = self._collections[collection]

        # Add to FAISS
        state.faiss_index.add(embeddings.astype(np.float32))  # type: ignore[arg-type]

        # Rebuild BM25 (rank_bm25 doesn't support incremental adds)
        state.chunks.extend(chunks)
        state.tokenised_corpus.extend([_tokenise(c.content) for c in chunks])
        state.bm25 = BM25Okapi(state.tokenised_corpus)

        log.info(
            "chunks indexed",
            collection=collection,
            added=len(chunks),
            total=len(state.chunks),
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """
        Hybrid search: dense + sparse via RRF.

        Returns up to ``top_k`` results sorted by fused score descending.
        """
        if collection not in self._collections:
            raise KeyError(f"Collection '{collection}' not found. Ingest documents first.")

        state = self._collections[collection]
        n = len(state.chunks)
        if n == 0:
            return []

        candidate_k = min(top_k * 3, n)  # Over-fetch before fusion

        # Dense retrieval
        dense_indices = self._dense_search(state, query, candidate_k)

        # Sparse retrieval
        sparse_indices = self._sparse_search(state, query, candidate_k)

        # Fuse
        fused = _reciprocal_rank_fusion([dense_indices, sparse_indices])

        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(fused[:top_k]):
            chunk = state.chunks[idx]
            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def _dense_search(
        self, state: _CollectionState, query: str, k: int
    ) -> list[int]:
        embedder = get_embedding_service()
        query_vec = embedder.encode_query(query).reshape(1, -1).astype(np.float32)
        k = min(k, state.faiss_index.ntotal)
        _, indices = state.faiss_index.search(query_vec, k)  # type: ignore[arg-type]
        return [int(i) for i in indices[0] if i >= 0]

    def _sparse_search(
        self, state: _CollectionState, query: str, k: int
    ) -> list[int]:
        tokens = _tokenise(query)
        bm25_scores = state.bm25.get_scores(tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:k]
        return [int(i) for i in top_indices]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_collection(self, collection: str) -> None:
        """Persist a collection's FAISS index and BM25 state to disk."""
        if collection not in self._collections:
            raise KeyError(f"Collection '{collection}' not found")

        state = self._collections[collection]
        path = self.index_dir / collection
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(state.faiss_index, str(path / "faiss.index"))
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump((state.bm25, state.tokenised_corpus), f)
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(state.chunks, f)

        log.info("collection saved", collection=collection, path=str(path))

    def load_collection(self, collection: str) -> None:
        """Load a persisted collection from disk into memory."""
        path = self.index_dir / collection
        if not path.exists():
            raise FileNotFoundError(f"No saved collection at {path}")

        faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "bm25.pkl", "rb") as f:
            bm25, tokenised_corpus = pickle.load(f)
        with open(path / "metadata.pkl", "rb") as f:
            chunks = pickle.load(f)

        self._collections[collection] = _CollectionState(
            faiss_index=faiss_index,
            bm25=bm25,
            chunks=chunks,
            tokenised_corpus=tokenised_corpus,
        )
        log.info("collection loaded", collection=collection, chunks=len(chunks))

    def load_all(self) -> None:
        """Scan index_dir and load every persisted collection at startup."""
        if not self.index_dir.exists():
            return
        for child in self.index_dir.iterdir():
            if child.is_dir() and (child / "faiss.index").exists():
                try:
                    self.load_collection(child.name)
                except Exception:
                    log.exception("failed to load collection", collection=child.name)

    def delete_collection(self, collection: str) -> None:
        """Remove a collection from memory and delete its files from disk."""
        import shutil

        self._collections.pop(collection, None)
        path = self.index_dir / collection
        if path.exists():
            shutil.rmtree(path)
        log.info("collection deleted", collection=collection)


# Module-level singleton
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            index_dir=settings.index_dir,
            alpha=settings.hybrid_alpha,
        )
    return _retriever
