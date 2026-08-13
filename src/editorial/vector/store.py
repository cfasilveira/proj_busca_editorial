"""Camada de armazenamento vetorial com interface trocável.

`VectorStore` é o contrato abstrato. `FaissStore` é a implementação local
padrão (leve, sem servidor). Novos backends (Milvus, Pinecone) podem
implementar a mesma interface sem afetar o restante do pipeline.

Fail gracefully: consultas sem resultado ou falhas de indexação são
registradas em log estruturado e retornam resultado vazio claro.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..errors import VectorError, fail
from ..logging_setup import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> None:
        raise NotImplementedError


class FaissStore(VectorStore):
    """Índice FAISS local (Inner Product sobre vetores L2-normalizados)."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise VectorError(
                "Dimensão inválida para o índice FAISS",
                user_message="A dimensão do índice FAISS precisa ser maior que zero.",
            )
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VectorError(
                "faiss-cpu não está instalado",
                user_message="Instale o pacote 'faiss-cpu' para usar o armazenamento vetorial.",
            ) from exc

        self._faiss = faiss
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: Sequence[str]) -> int:
        """Adiciona vetores já L2-normalizados. Fail first em incompatibilidades."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise VectorError(
                f"Matriz de vetores deve ter 2 dimensões (recebido {matrix.ndim})",
                user_message="A lista de vetores precisa ser uma matriz 2D.",
            )
        if matrix.shape[0] != len(ids):
            raise VectorError(
                f"Vetores ({matrix.shape[0]}) e ids ({len(ids)}) em quantidade diferente",
                user_message="Cada vetor precisa ter um identificador correspondente.",
            )
        if matrix.shape[1] != self.dimension:
            raise VectorError(
                f"Vetor com dimensão {matrix.shape[1]} != índice {self.dimension}",
                user_message="Os vetores precisam ter a mesma dimensão do índice.",
            )

        self._index.add(matrix)
        self._ids.extend(str(uid) for uid in ids)
        logger.info(
            "Vetores indexados",
            extra={
                "code": "index_added",
                "metric": f"{len(ids)} vetores",
                "status": "ok",
            },
        )
        return len(ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dimension:
            raise VectorError(
                f"Consulta com dimensão {vector.shape[1]} != índice {self.dimension}",
                user_message="O vetor de consulta tem dimensão incompatível com o índice.",
            )
        if top_k < 1:
            raise VectorError(
                "top_k inválido para busca vetorial",
                user_message="O número de resultados precisa ser ao menos 1.",
            )

        if self.count() == 0:
            logger.warning(
                "Consulta vetorial em índice vazio",
                extra={"code": "empty_index_query"},
            )
            return []

        k = min(top_k, self.count())
        distances, indices = self._index.search(vector, k)

        results: list[tuple[str, float]] = []
        for distance, position in zip(distances[0], indices[0], strict=False):
            if position < 0:
                continue
            results.append((self._ids[int(position)], float(distance)))

        if not results:
            logger.warning(
                "Consulta vetorial sem resultados",
                extra={"code": "no_vector_results"},
            )
        else:
            logger.info(
                "Consulta vetorial concluída",
                extra={
                    "code": "vector_search_done",
                    "metric": f"{len(results)} resultados",
                },
            )
        return results

    def count(self) -> int:
        return int(self._index.ntotal)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._faiss.write_index(self._index, str(target))
            Path(str(target) + ".ids").write_text("\n".join(self._ids), encoding="utf-8")
        except (OSError, ValueError) as exc:
            fail(
                logger,
                VectorError,
                f"Falha ao salvar índice FAISS em {target}: {exc}",
                user_message=f"Não foi possível salvar o índice vetorial em '{target}'.",
                code="index_save_failed",
                doc_id=str(target),
            )
        logger.info("Índice FAISS salvo", extra={"code": "index_saved", "doc_id": str(target)})

    def load(self, path: str | Path) -> None:
        target = Path(path)
        if not target.exists():
            raise VectorError(
                f"Índice FAISS não encontrado em {target}",
                user_message=f"O índice vetorial '{target}' não existe.",
            )
        try:
            self._index = self._faiss.read_index(str(target))
            ids_path = Path(str(target) + ".ids")
            if ids_path.exists():
                self._ids = ids_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            fail(
                logger,
                VectorError,
                f"Falha ao carregar índice FAISS de {target}: {exc}",
                user_message=f"Não foi possível carregar o índice vetorial de '{target}'.",
                code="index_load_failed",
                doc_id=str(target),
            )
        logger.info(
            "Índice FAISS carregado",
            extra={"code": "index_loaded", "doc_id": str(target)},
        )


def create_store(kind: str, dimension: int) -> VectorStore:
    """Fábrica de stores — ponto único de troca de backend."""
    match kind:
        case "faiss":
            return FaissStore(dimension)
        case "milvus" | "pinecone":
            raise VectorError(
                f"Backend {kind} não implementado",
                user_message=f"O backend {kind} ainda não está implementado nesta versão.",
            )
        case _:
            raise VectorError(
                f"Backend vetorial desconhecido: {kind}",
                user_message=f"Backend vetorial '{kind}' não é reconhecido.",
            )
