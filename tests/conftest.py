"""
Shared pytest fixtures and configuration.

Fixtures defined here are automatically available to all test files
without explicit imports.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Ensure tests always run with a safe, isolated configuration."""
    tmpdir = tempfile.mkdtemp(prefix="rag_test_")
    overrides = {
        "INDEX_DIR": tmpdir,
        "LLM_PROVIDER": "ollama",
        "LLM_MODEL": "llama3",
        "LOG_LEVEL": "WARNING",   # Suppress logs during tests
        "EMBEDDING_DEVICE": "cpu",
    }
    original = {}
    for key, val in overrides.items():
        original[key] = os.environ.get(key)
        os.environ[key] = val

    yield tmpdir

    # Restore original environment
    for key, original_val in original.items():
        if original_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_val
