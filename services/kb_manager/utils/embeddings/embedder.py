"""Adapter that exposes embedding helpers for the KB manager utils."""

from __future__ import annotations

from agents.retrievers.utils.models_builder import getEmbeddingModel


def get_embedding_model():
    """Return the default embedding model shared with agent retrievers."""
    return getEmbeddingModel()
