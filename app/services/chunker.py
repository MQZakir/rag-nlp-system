"""
Sentence-aware recursive text chunker.

Strategy
--------
1. Try to split on double newlines (paragraph boundaries).
2. Fall back to single newlines, then sentence-ending punctuation, then spaces.
3. Merge splits greedily until the chunk would exceed `chunk_size` tokens,
   then emit the current chunk and start a new one carrying `chunk_overlap`
   tokens of context from the tail of the previous chunk.

Token counting uses a fast whitespace approximation (word count × 1.3) to
avoid loading a full tokenizer here. The estimate is conservative enough that
chunks will always fit within a typical transformer context window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    token_estimate: int = 0


# Ordered from coarsest to finest — we try each separator in turn
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words × 1.3 (accounts for subword tokenization)."""
    return int(len(text.split()) * 1.3)


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to the left of each part (except the first) so
    # that merging later doesn't lose punctuation.
    if separator in (". ", "! ", "? ", "; ", ", "):
        rejoined: list[str] = []
        for i, part in enumerate(parts):
            rejoined.append((separator.rstrip() + " " if i > 0 else "") + part)
        return [p for p in rejoined if p.strip()]
    return [p for p in parts if p.strip()]


def _merge_splits(splits: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily merge splits into chunks respecting size and overlap."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for split in splits:
        split_tokens = _estimate_tokens(split)

        # A single split larger than chunk_size is emitted as-is (can't split further)
        if split_tokens > chunk_size:
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.append(split)
            continue

        if current_tokens + split_tokens > chunk_size and current_parts:
            chunks.append(" ".join(current_parts))

            # Carry overlap: take splits from the end of current_parts until
            # we have approximately chunk_overlap tokens
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for part in reversed(current_parts):
                part_tokens = _estimate_tokens(part)
                if overlap_tokens + part_tokens > chunk_overlap:
                    break
                overlap_parts.insert(0, part)
                overlap_tokens += part_tokens

            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(split)
        current_tokens += split_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text using the first separator that produces manageable pieces."""
    for i, sep in enumerate(separators):
        splits = _split_on_separator(text, sep)
        # If every split is within chunk_size, we're done
        if all(_estimate_tokens(s) <= chunk_size for s in splits):
            return splits
        # Otherwise recurse with finer separators on oversized pieces
        if i < len(separators) - 1:
            result: list[str] = []
            for s in splits:
                if _estimate_tokens(s) > chunk_size:
                    result.extend(_recursive_split(s, separators[i + 1 :], chunk_size))
                else:
                    result.append(s)
            return result
    # Last resort: character-level split (never ideal but always terminates)
    return list(text)


class RecursiveChunker:
    """
    Splits documents into overlapping chunks using a hierarchy of separators.

    Parameters
    ----------
    chunk_size:
        Target maximum token count per chunk.
    chunk_overlap:
        Number of tokens to carry over from the previous chunk.
    add_start_index:
        If True, adds a ``start_index`` key to each chunk's metadata with its
        character offset in the original document.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        add_start_index: bool = True,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split a single document into Chunk objects."""
        metadata = metadata or {}
        text = re.sub(r"\s+\n", "\n", text)  # normalise trailing whitespace before newlines
        text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excessive blank lines

        raw_splits = _recursive_split(text, _SEPARATORS, self.chunk_size)
        merged = _merge_splits(raw_splits, self.chunk_size, self.chunk_overlap)

        chunks: list[Chunk] = []
        search_start = 0

        for i, content in enumerate(merged):
            content = content.strip()
            if not content:
                continue

            chunk_meta = dict(metadata)
            if self.add_start_index:
                # Best-effort character offset (not exact due to merging)
                idx = text.find(content[:50], search_start)
                if idx != -1:
                    chunk_meta["start_index"] = idx
                    search_start = idx + 1

            chunks.append(
                Chunk(
                    content=content,
                    metadata=chunk_meta,
                    chunk_index=i,
                    token_estimate=_estimate_tokens(content),
                )
            )

        return chunks

    def split_batch(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[Chunk]:
        """Split multiple documents, updating metadata with doc index."""
        metadatas = metadatas or [{} for _ in texts]
        all_chunks: list[Chunk] = []
        for doc_idx, (text, meta) in enumerate(zip(texts, metadatas)):
            doc_meta = {"doc_index": doc_idx, **meta}
            all_chunks.extend(self.split(text, doc_meta))
        return all_chunks
