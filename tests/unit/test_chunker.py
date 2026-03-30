"""Unit tests for RecursiveChunker."""

from __future__ import annotations

import pytest

from app.services.chunker import Chunk, RecursiveChunker, _estimate_tokens


class TestTokenEstimate:
    def test_empty(self):
        assert _estimate_tokens("") == 0

    def test_single_word(self):
        assert _estimate_tokens("hello") == int(1 * 1.3)

    def test_sentence(self):
        tokens = _estimate_tokens("the quick brown fox")
        assert tokens == int(4 * 1.3)


class TestRecursiveChunker:
    def setup_method(self):
        self.chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

    def test_short_text_single_chunk(self):
        text = "This is a short document."
        chunks = self.chunker.split(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_returns_chunk_objects(self):
        chunks = self.chunker.split("Hello world.")
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_text_returns_empty(self):
        chunks = self.chunker.split("   \n\n  ")
        assert chunks == []

    def test_long_text_splits_into_multiple_chunks(self):
        # ~500 words → should produce multiple chunks at size=100
        text = " ".join([f"word{i}" for i in range(500)])
        chunks = self.chunker.split(text)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_size(self):
        text = " ".join([f"word{i}" for i in range(500)])
        chunks = self.chunker.split(text)
        for chunk in chunks:
            assert chunk.token_estimate <= self.chunker.chunk_size * 1.2  # 20% tolerance

    def test_metadata_propagated(self):
        meta = {"source": "test.txt", "page": 1}
        chunks = self.chunker.split("Some text here.", metadata=meta)
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["page"] == 1

    def test_start_index_added(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=2, add_start_index=True)
        chunks = chunker.split(text)
        assert any("start_index" in c.metadata for c in chunks)

    def test_chunk_index_sequential(self):
        text = " ".join([f"w{i}" for i in range(300)])
        chunks = self.chunker.split(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_split_batch_doc_index(self):
        texts = ["Document one content.", "Document two content."]
        chunks = self.chunker.split_batch(texts)
        doc_indices = {c.metadata.get("doc_index") for c in chunks}
        assert 0 in doc_indices
        assert 1 in doc_indices

    def test_split_batch_metadatas(self):
        texts = ["Text A.", "Text B."]
        metadatas = [{"author": "Alice"}, {"author": "Bob"}]
        chunks = self.chunker.split_batch(texts, metadatas=metadatas)
        authors = {c.metadata.get("author") for c in chunks}
        assert "Alice" in authors
        assert "Bob" in authors

    def test_paragraph_boundary_preferred(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunker = RecursiveChunker(chunk_size=15, chunk_overlap=2)
        chunks = chunker.split(text)
        # Each paragraph should ideally be its own chunk
        assert len(chunks) >= 2

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            RecursiveChunker(chunk_size=50, chunk_overlap=50)

    def test_whitespace_normalisation(self):
        text = "Word   with   extra   spaces.\n\n\n\nToo many newlines."
        chunks = self.chunker.split(text)
        for chunk in chunks:
            assert "\n\n\n" not in chunk.content
