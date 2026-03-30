"""
HuggingFace sentence-transformers embedding wrapper.

Handles:
- Lazy model loading (model is loaded on first use, not at import time)
- Batched encoding with configurable batch size
- BGE-style query prefix injection ("Represent this sentence: ")
- Normalised embeddings (unit L2 norm) for cosine similarity via dot product

The ``encode_query`` / ``encode_documents`` distinction matters for asymmetric
models like BGE, where query and passage representations use different prefixes.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Some models use specific query prefixes for better asymmetric search
_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class EmbeddingService:
    """
    Thread-safe singleton wrapper around a SentenceTransformer model.

    Uses a lock to ensure the model is only loaded once even under
    concurrent requests at startup.
    """

    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    log.info("loading embedding model", model=self.model_name, device=self.device)
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )
                    log.info(
                        "embedding model loaded",
                        model=self.model_name,
                        dim=self._model.get_sentence_embedding_dimension(),
                    )
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def _query_prefix(self) -> str:
        return _QUERY_PREFIXES.get(self.model_name, "")

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Encode a single query string with the appropriate prefix."""
        prefix = self._query_prefix()
        text = prefix + query if prefix else query
        vec: NDArray[np.float32] = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vec

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode a list of document passages in batches."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        log.debug("encoding documents", count=len(texts), batch_size=self.batch_size)
        vecs: NDArray[np.float32] = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vecs


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Module-level cached getter — returns the same instance every call."""
    return EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
