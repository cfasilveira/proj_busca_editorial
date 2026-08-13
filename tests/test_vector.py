"""Testes do módulo Vetorial."""

from __future__ import annotations

from pathlib import Path

import pytest

from editorial.errors import VectorError
from editorial.vector import EmbeddingModel, FaissStore, create_store


def test_embedding_fit_and_transform_consistency():
    docs = ["política econômica", "transparência fiscal", "educação pública"]
    model = EmbeddingModel(dim=32).fit(docs)
    assert model.fitted
    v = model.transform("política econômica e fiscal")
    assert v.shape == (32,)
    assert v.dtype == "float32"


def test_embedding_empty_corpus_fails():
    with pytest.raises(VectorError, match="vazio"):
        EmbeddingModel(dim=32).fit([])


def test_embedding_transform_before_fit_fails():
    with pytest.raises(VectorError, match="treinado"):
        EmbeddingModel(dim=32).transform("texto")


def test_embedding_roundtrip(tmp_path: Path):
    docs = ["política econômica", "transparência fiscal", "educação pública"]
    model = EmbeddingModel(dim=32).fit(docs)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = EmbeddingModel.load(path)
    assert loaded.fitted
    assert loaded.transform("política").shape == (32,)


def test_embedding_load_missing_fails():
    with pytest.raises(VectorError):
        EmbeddingModel.load(Path("/nao/existe.joblib"))


def test_faiss_add_search_roundtrip(tmp_path: Path):
    import numpy as np

    store = FaissStore(dimension=4)
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    store.add(vectors, ["a", "b", "c"])
    assert store.count() == 3

    results = store.search(np.array([[1, 0, 0, 0]], dtype=np.float32), top_k=2)
    assert results[0][0] == "a"

    index_path = tmp_path / "index.faiss"
    store.save(index_path)
    loaded = FaissStore(dimension=4)
    loaded.load(index_path)
    assert loaded.count() == 3
    assert loaded.search(np.array([[0, 1, 0, 0]], dtype=np.float32))[0][0] == "b"


def test_faiss_dimension_mismatch_fails():
    import numpy as np

    store = FaissStore(dimension=4)
    with pytest.raises(VectorError, match="dimensão"):
        store.add(np.ones((2, 3), dtype=np.float32), ["a", "b"])


def test_faiss_ids_mismatch_fails():
    import numpy as np

    store = FaissStore(dimension=4)
    with pytest.raises(VectorError, match="quantidade"):
        store.add(np.ones((3, 4), dtype=np.float32), ["a"])


def test_faiss_search_empty_index_graceful():
    import numpy as np

    store = FaissStore(dimension=4)
    assert store.search(np.ones((1, 4), dtype=np.float32)) == []


def test_create_store_unknown_backend():
    with pytest.raises(VectorError, match="desconhecido"):
        create_store("weaviate", 32)
