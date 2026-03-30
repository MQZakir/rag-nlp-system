"""Unit tests for the cross-encoder reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.reranker import RerankerService, _sigmoid
from app.services.retriever import RetrievedChunk


def _make_chunk(content: str, score: float = 0.5, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        metadata={"source": "test"},
        score=score,
        rank=rank,
    )


def _make_mock_cross_encoder(scores: list[float]):
    """Mock CrossEncoder.predict to return controlled scores."""
    model = MagicMock()
    model.predict.return_value = np.array(scores, dtype=np.float32)
    return model


class TestSigmoid:
    def test_zero_maps_to_half(self):
        result = _sigmoid(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-6

    def test_large_positive_approaches_one(self):
        result = _sigmoid(np.array([100.0]))
        assert result[0] > 0.999

    def test_large_negative_approaches_zero(self):
        result = _sigmoid(np.array([-100.0]))
        assert result[0] < 0.001

    def test_output_in_range(self):
        inputs = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        outputs = _sigmoid(inputs)
        assert np.all(outputs >= 0.0)
        assert np.all(outputs <= 1.0)


class TestRerankerService:
    def setup_method(self):
        self.reranker = RerankerService(model_name="mock-model")

    def _patch_model(self, scores: list[float]):
        mock_model = _make_mock_cross_encoder(scores)
        self.reranker._model = mock_model
        return mock_model

    def test_rerank_empty_returns_empty(self):
        results = self.reranker.rerank("query", [])
        assert results == []

    def test_rerank_sorts_by_score_descending(self):
        chunks = [
            _make_chunk("low relevance", rank=0),
            _make_chunk("high relevance", rank=1),
            _make_chunk("medium relevance", rank=2),
        ]
        # Cross-encoder scores: low=0.1, high=5.0, medium=2.0
        self._patch_model([0.1, 5.0, 2.0])
        results = self.reranker.rerank("test query", chunks)

        assert results[0].content == "high relevance"
        assert results[1].content == "medium relevance"
        assert results[2].content == "low relevance"

    def test_rerank_scores_are_normalised(self):
        chunks = [_make_chunk(f"chunk {i}", rank=i) for i in range(3)]
        self._patch_model([1.0, 2.0, 3.0])
        results = self.reranker.rerank("query", chunks)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_rerank_updates_rank_field(self):
        chunks = [_make_chunk(f"chunk {i}", rank=i) for i in range(3)]
        self._patch_model([3.0, 1.0, 2.0])
        results = self.reranker.rerank("query", chunks)
        for expected_rank, result in enumerate(results):
            assert result.rank == expected_rank

    def test_top_k_limits_output(self):
        chunks = [_make_chunk(f"chunk {i}", rank=i) for i in range(5)]
        self._patch_model([5.0, 4.0, 3.0, 2.0, 1.0])
        results = self.reranker.rerank("query", chunks, top_k=3)
        assert len(results) == 3

    def test_top_k_none_returns_all(self):
        chunks = [_make_chunk(f"chunk {i}", rank=i) for i in range(4)]
        self._patch_model([1.0, 2.0, 3.0, 4.0])
        results = self.reranker.rerank("query", chunks, top_k=None)
        assert len(results) == 4

    def test_single_chunk(self):
        chunks = [_make_chunk("only chunk")]
        self._patch_model([2.5])
        results = self.reranker.rerank("query", chunks)
        assert len(results) == 1
        assert results[0].rank == 0

    def test_cross_encoder_called_with_pairs(self):
        chunks = [_make_chunk("text A"), _make_chunk("text B")]
        mock_model = self._patch_model([1.0, 0.5])
        self.reranker.rerank("my query", chunks)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args[0] == ["my query", "text A"]
        assert call_args[1] == ["my query", "text B"]

    def test_metadata_preserved_after_rerank(self):
        chunks = [
            RetrievedChunk(
                content="text", metadata={"source": "doc1.pdf", "page": 3}, score=0.5, rank=0
            )
        ]
        self._patch_model([1.0])
        results = self.reranker.rerank("query", chunks)
        assert results[0].metadata["source"] == "doc1.pdf"
        assert results[0].metadata["page"] == 3
