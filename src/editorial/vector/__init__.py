"""Módulo Vetorial: embeddings e armazenamento/consulta vetorial."""

from .embeddings import EmbeddingModel
from .store import FaissStore, VectorStore, create_store

__all__ = [
    "EmbeddingModel",
    "FaissStore",
    "VectorStore",
    "create_store",
]
