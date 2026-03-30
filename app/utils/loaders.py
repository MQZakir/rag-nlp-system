"""
Document loaders for ingestion scripts.

Each loader accepts a file path and returns a list of (text, metadata) tuples.
Adding support for a new format means implementing one function and registering
it in the LOADERS dict at the bottom of this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_txt(path: Path) -> list[tuple[str, dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(text, {"source": path.name, "format": "txt"})]


def load_pdf(path: Path) -> list[tuple[str, dict[str, Any]]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required to load PDFs: pip install pypdf") from exc

    reader = PdfReader(str(path))
    results: list[tuple[str, dict[str, Any]]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            results.append(
                (text, {"source": path.name, "format": "pdf", "page": page_num})
            )
    return results


def load_md(path: Path) -> list[tuple[str, dict[str, Any]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(text, {"source": path.name, "format": "md"})]


# Registry: extension → loader function
LOADERS: dict[str, object] = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".md": load_md,
    ".markdown": load_md,
}


def load_file(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Dispatch to the appropriate loader based on file extension."""
    ext = path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(LOADERS)}"
        )
    return loader(path)  # type: ignore[operator]


def load_directory(
    directory: Path, recursive: bool = True
) -> list[tuple[str, dict[str, Any]]]:
    """Load all supported files from a directory."""
    pattern = "**/*" if recursive else "*"
    results: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in LOADERS:
            try:
                results.extend(load_file(path))
            except Exception as exc:
                # Log and continue — don't abort entire batch for one bad file
                print(f"[warn] skipping {path.name}: {exc}")
    return results
