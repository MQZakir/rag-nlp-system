"""
Cross-encoder reranker.

Why rerank?
-----------
Bi-encoder retrieval (dense + BM25) is fast but scores queries and documents
independently. A cross-encoder receives the (query, passage) pair concatenated
and attends across both — it's slower but produces much better relevance scores.

We therefore use a two-stage approach:
  1. Fast retrieval fetches a large candidate set (e.g. top-20).
  2. The cross-encoder reranks those 20 candidates and we keep the top-K.

This gives near-reranker quality at a fraction of the cost of running the
cross-encoder over the full corpus.

Model
-----
Default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
A 6-layer MiniLM fine-tuned on MS MARCO passage ranking. Fast on CPU,
fits in ~45MB RAM. Swap for ``ms-marco-MiniLM-L-12-v2`` for better accuracy.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import get_logger
from app.services.retriever import RetrievedChunk

log = get_logger(__name__)


def _sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return 1.0 / (1.0 + np.exp(-x))


class RerankerService:
    """
    Wraps a cross-encoder model to rerank a list of retrieved chunks.

    The model is loaded lazily on first use to avoid blocking application
    startup (especially in test environments where reranking is mocked).
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    log.info("loading reranker model", model=self.model_name)
                    self._model = CrossEncoder(self.model_name)
                    log.info("reranker model loaded", model=self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Score each (query, chunk) pair and return re-ordered results.

        Parameters
        ----------
        query:
            The original user query.
        chunks:
            Candidate chunks from the retriever.
        top_k:
            If provided, return only the top-k highest-scoring chunks.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by cross-encoder score descending, with updated
            ``score`` and ``rank`` fields.
        """
        if not chunks:
            return []

        pairs = [[query, c.content] for c in chunks]
        raw_scores: NDArray[np.float32] = self.model.predict(pairs, show_progress_bar=False)

        # Normalise cross-encoder logits to [0, 1] via sigmoid
        normalised: NDArray[np.float32] = _sigmoid(raw_scores)

        ranked = sorted(
            zip(chunks, normalised),
            key=lambda x: x[1],
            reverse=True,
        )

        result: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(ranked):
            result.append(
                RetrievedChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(score),
                    rank=rank,
                )
            )

        if top_k is not None:
            result = result[:top_k]

        log.debug(
            "reranking complete",
            input_count=len(chunks),
            output_count=len(result),
            top_score=result[0].score if result else None,
        )
        return result


@lru_cache(maxsize=1)
def get_reranker() -> RerankerService:
    return RerankerService(model_name=settings.reranker_model)
