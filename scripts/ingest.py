#!/usr/bin/env python3
"""
CLI ingestion tool.

Usage
-----
# Ingest a directory of documents into a named collection
python scripts/ingest.py --source data/raw/ --collection my_docs

# Ingest a single file
python scripts/ingest.py --source data/raw/report.pdf --collection finance

# Dry-run: show what would be ingested without actually indexing
python scripts/ingest.py --source data/raw/ --collection test --dry-run

# Override chunk size for this run
python scripts/ingest.py --source data/raw/ --collection my_docs \\
    --chunk-size 256 --chunk-overlap 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.services.chunker import RecursiveChunker
from app.services.retriever import get_retriever
from app.utils.loaders import load_directory, load_file

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into a RAG collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to a file or directory to ingest",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="Collection name (default: 'default')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Target tokens per chunk (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Overlap tokens between chunks (default: {settings.chunk_overlap})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without indexing",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan directories (default: True)",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    source: Path = args.source

    if not source.exists():
        log.error("source path does not exist", path=str(source))
        return 1

    # ── Load documents ────────────────────────────────────────────────────────
    print(f"\n  Loading documents from: {source}")
    t0 = time.perf_counter()

    if source.is_file():
        documents = load_file(source)
    else:
        documents = load_directory(source, recursive=args.recursive)

    if not documents:
        log.error("no supported documents found", source=str(source))
        return 1

    texts = [d[0] for d in documents]
    metadatas = [d[1] for d in documents]
    load_ms = int((time.perf_counter() - t0) * 1000)
    print(f"  Loaded {len(documents)} document(s) in {load_ms}ms")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunker = RecursiveChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = chunker.split_batch(texts=texts, metadatas=metadatas)
    print(f"  Produced {len(chunks)} chunk(s)  (size={args.chunk_size}, overlap={args.chunk_overlap})")

    if args.dry_run:
        print("\n  [dry-run] Showing first 3 chunks:\n")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  --- Chunk {i} ({chunk.token_estimate} tokens) ---")
            print(f"  {chunk.content[:200]}{'...' if len(chunk.content) > 200 else ''}")
            print(f"  Metadata: {chunk.metadata}\n")
        print("  [dry-run] No data was written to disk.\n")
        return 0

    # ── Embed & index ─────────────────────────────────────────────────────────
    print(f"\n  Embedding and indexing into collection '{args.collection}'...")
    t1 = time.perf_counter()

    retriever = get_retriever()
    retriever.add_chunks(collection=args.collection, chunks=chunks)
    retriever.save_collection(args.collection)

    index_ms = int((time.perf_counter() - t1) * 1000)
    total_ms = int((time.perf_counter() - t0) * 1000)

    print(f"  Indexed in {index_ms}ms  |  Total: {total_ms}ms")
    print(f"\n  ✓ Collection '{args.collection}' ready  ({len(chunks)} chunks from {len(documents)} docs)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
