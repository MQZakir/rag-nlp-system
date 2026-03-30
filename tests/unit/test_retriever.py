"""Unit tests for the HybridRetriever."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.chunker import Chunk
from app.services.retriever import (
    HybridRetriever,
    RetrievedChunk,
    _reciprocal_rank_fusion,
    _tokenise,
)


class TestTokenise:
    def test_lowercases(self):
        assert _tokenise("Hello World") == ["hello", "world"]

    def test_splits_on_whitespace(self):
        assert _tokenise("one two three") == ["one", "two", "three"]

    def test_empty_string(self):
        assert _tokenise("") == []


class TestRRF:
    def test_single_list(self):
        results = _reciprocal_rank_fusion([[0, 1, 2]])
        # Index 0 should score highest (rank 0)
        indices = [idx for idx, _ in results]
        assert indices[0] == 0

    def test_two_lists_agreement_boosts_score(self):
        # Both lists agree: index 0 is best
        results = _reciprocal_rank_fusion([[0, 1, 2], [0, 2, 1]])
        top_idx = results[0][0]
        assert top_idx == 0

    def test_two_lists_disagreement_gives_consensus(self):
        # List 1 prefers 0, list 2 prefers 1
        # Index 0: 1/(60+1) + 1/(60+2), Index 1: 1/(60+2) + 1/(60+1) — tied
        results = _reciprocal_rank_fusion([[0, 1], [1, 0]])
        scores = dict(results)
        # Both should have equal scores
        assert abs(scores[0] - scores[1]) < 1e-9

    def test_returns_sorted_descending(self):
        results = _reciprocal_rank_fusion([[2, 1, 0], [2, 1, 0]])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_lists(self):
        results = _reciprocal_rank_fusion([[], []])
        assert results == []


def _make_chunk(content: str, idx: int = 0) -> Chunk:
    return Chunk(content=content, metadata={"doc_index": idx}, chunk_index=0)


def _make_mock_embedding_service(dim: int = 8):
    """Return a mock EmbeddingService that produces random unit vectors."""
    service = MagicMock()
    service.embedding_dim = dim

    def encode_docs(texts):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def encode_query(q):
        vec = np.random.randn(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    service.encode_documents.side_effect = encode_docs
    service.encode_query.side_effect = encode_query
    return service


class TestHybridRetriever:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.retriever = HybridRetriever(index_dir=Path(self.tmpdir), alpha=0.6)
        self.mock_embedder = _make_mock_embedding_service(dim=8)

    def _add_chunks(self, collection: str, contents: list[str]) -> None:
        chunks = [_make_chunk(c, i) for i, c in enumerate(contents)]
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            self.retriever.add_chunks(collection, chunks)

    def test_collection_created_on_add(self):
        self._add_chunks("test", ["hello world"])
        assert self.retriever.has_collection("test")

    def test_search_returns_retrieved_chunks(self):
        self._add_chunks("test", ["machine learning is great", "python is a language"])
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            results = self.retriever.search("test", "machine learning", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_search_scores_in_range(self):
        self._add_chunks("test", ["neural networks", "deep learning", "transformers"])
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            results = self.retriever.search("test", "neural networks", top_k=3)
        for r in results:
            assert r.score >= 0.0

    def test_search_nonexistent_collection_raises(self):
        with pytest.raises(KeyError, match="not found"):
            with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
                self.retriever.search("nonexistent", "query")

    def test_search_top_k_respected(self):
        self._add_chunks("test", [f"document {i}" for i in range(10)])
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            results = self.retriever.search("test", "document", top_k=3)
        assert len(results) <= 3

    def test_save_and_load_collection(self):
        self._add_chunks("persist_test", ["saved content here"])
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            self.retriever.save_collection("persist_test")

        # Create a fresh retriever and load from disk
        new_retriever = HybridRetriever(index_dir=Path(self.tmpdir))
        new_retriever.load_collection("persist_test")
        assert new_retriever.has_collection("persist_test")
        state = new_retriever._collections["persist_test"]
        assert len(state.chunks) == 1
        assert state.chunks[0].content == "saved content here"

    def test_delete_collection(self):
        self._add_chunks("to_delete", ["temporary content"])
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            self.retriever.save_collection("to_delete")
        self.retriever.delete_collection("to_delete")
        assert not self.retriever.has_collection("to_delete")
        assert not (Path(self.tmpdir) / "to_delete").exists()

    def test_collection_info(self):
        self._add_chunks("info_test", ["doc one", "doc two"])
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            self.retriever.save_collection("info_test")
        info = self.retriever.collection_info("info_test")
        assert info["chunk_count"] == 2
        assert info["name"] == "info_test"

    def test_incremental_add_chunks(self):
        self._add_chunks("incr", ["first batch"])
        self._add_chunks("incr", ["second batch"])
        state = self.retriever._collections["incr"]
        assert len(state.chunks) == 2

    def test_empty_collection_search_returns_empty(self):
        # Manually insert empty collection state
        import faiss
        from rank_bm25 import BM25Okapi
        from app.services.retriever import _CollectionState

        state = _CollectionState(
            faiss_index=faiss.IndexFlatIP(8),
            bm25=BM25Okapi([[]]),
            chunks=[],
            tokenised_corpus=[],
        )
        self.retriever._collections["empty"] = state
        with patch("app.services.retriever.get_embedding_service", return_value=self.mock_embedder):
            results = self.retriever.search("empty", "anything")
        assert results == []
