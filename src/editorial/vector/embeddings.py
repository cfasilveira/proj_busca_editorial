"""Geração de embeddings densos a partir de TF-IDF + SVD (truncado).

Abordagem leve e offline: vetoriza o corpus com TF-IDF e projeta para um
espaço denso de dimensão fixa via TruncatedSVD. O pipeline é persistido
com joblib para permitir consultas consistentes após o treino.

Fail first: corpus vazio ou com variância nula é rejeitado. Falhas de
persistência/carga são registradas em log e propagadas como VectorError.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..errors import VectorError
from ..logging_setup import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper de TF-IDF + SVD. Produz vetores L2-normalizados de `dim`."""

    def __init__(self, dim: int = 256, *, min_df: int = 1) -> None:
        if dim < 2:
            raise VectorError(
                f"dimensão de embedding inválida: {dim}",
                user_message="A dimensão do embedding precisa ser ao menos 2.",
            )
        self.dim = dim
        self.min_df = min_df
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    @property
    def fitted(self) -> bool:
        return self._tfidf is not None and self._svd is not None

    def fit(self, documents: Sequence[str]) -> EmbeddingModel:
        """Treina o pipeline TF-IDF + SVD sobre o corpus de treino."""
        if not documents:
            raise VectorError(
                "Corpus vazio para treino de embeddings",
                user_message="Não é possível treinar embeddings sem documentos.",
            )
        texts = [doc for doc in documents if doc and doc.strip()]
        if not texts:
            raise VectorError(
                "Corpus sem texto válido para treino de embeddings",
                user_message="Nenhum documento com texto foi fornecido para o treino.",
            )

        tfidf = TfidfVectorizer(min_df=self.min_df, lowercase=True, strip_accents="unicode")
        tfidf_matrix = tfidf.fit_transform(texts)

        if tfidf_matrix.shape[1] == 0:
            raise VectorError(
                "Vocabulário vazio após TF-IDF",
                user_message="Nenhum termo aproveitável foi extraído dos documentos.",
            )

        components = min(self.dim, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1])
        if components < 2:
            logger.warning(
                "Corpus pequeno demais para a dimensão solicitada",
                extra={
                    "code": "svd_dim_reduced",
                    "metric": f"{components} componentes",
                },
            )
            components = max(1, components)

        svd = TruncatedSVD(n_components=components, random_state=42)
        svd.fit(tfidf_matrix)

        if (
            hasattr(svd, "explained_variance_ratio_")
            and float(svd.explained_variance_ratio_.sum()) < 0.5
        ):
            logger.warning(
                "Baixa variância explicada pelo SVD",
                extra={
                    "code": "low_explained_variance",
                    "metric": f"{svd.explained_variance_ratio_.sum():.3f}",
                },
            )

        self._tfidf = tfidf
        self._svd = svd
        logger.info(
            "Modelo de embeddings treinado",
            extra={
                "code": "embedding_fitted",
                "metric": f"dim={self.dim}, vocab={tfidf_matrix.shape[1]}",
            },
        )
        return self

    def transform(self, text: str) -> np.ndarray:
        """Embedding denso normalizado (L2) para um único texto."""
        if not self.fitted:
            raise VectorError(
                "Modelo de embeddings não treinado",
                user_message="Treine o modelo de embeddings antes de fazer consultas.",
            )
        if not text or not text.strip():
            raise VectorError(
                "Consulta vazia para embedding",
                user_message="Não é possível gerar embedding de um texto vazio.",
            )

        sparse = self._tfidf.transform([text])
        dense = self._svd.transform(sparse)
        vector = dense[0]
        if vector.shape[0] < self.dim:
            # SVD reduz a dimensionalidade em corpora pequenos; faz padding
            # com zeros para manter a dimensão fixa exigida pelo índice.
            padded = np.zeros(self.dim, dtype=vector.dtype)
            padded[: vector.shape[0]] = vector
            vector = padded
        elif vector.shape[0] > self.dim:
            vector = vector[: self.dim]
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

    def transform_many(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self.transform(t) for t in texts]
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(
                {
                    "dim": self.dim,
                    "min_df": self.min_df,
                    "tfidf": self._tfidf,
                    "svd": self._svd,
                },
                target,
            )
        except (OSError, ValueError) as exc:
            logger.error(
                "Falha ao salvar modelo de embeddings",
                extra={"code": "embedding_save_failed", "doc_id": str(target)},
            )
            raise VectorError(
                f"Não foi possível salvar o modelo em {target}: {exc}",
                user_message=f"Não foi possível salvar o modelo de embeddings em '{target}'.",
            ) from exc

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingModel:
        target = Path(path)
        if not target.exists():
            raise VectorError(
                f"Modelo de embeddings não encontrado em {target}",
                user_message=f"O modelo de embeddings '{target}' não existe.",
            )
        try:
            payload = joblib.load(target)
        except Exception as exc:
            logger.error(
                "Falha ao carregar modelo de embeddings",
                extra={"code": "embedding_load_failed", "doc_id": str(target)},
            )
            raise VectorError(
                f"Não foi possível carregar o modelo de {target}: {exc}",
                user_message=f"Não foi possível carregar o modelo de embeddings de '{target}'.",
            ) from exc

        model = cls(dim=int(payload["dim"]), min_df=int(payload["min_df"]))
        model._tfidf = payload["tfidf"]
        model._svd = payload["svd"]
        return model
