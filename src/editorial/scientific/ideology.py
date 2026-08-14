"""Régua ideológica: posiciona textos entre dois perfis de referência.

Constrói os extremos como centroides (média L2-normalizada) dos embeddings
dos documentos de cada autor de referência e projeta novos textos sobre o
eixo direita-esquerda. O escore resultante está em [-1, 1]:

  - ~ -1  → muito próximo do perfil à esquerda (ex.: Miriam Leitão)
  - ~ +1  → muito próximo do perfil à direita  (ex.: Paulo Guedes)
  - ~  0  → centroide, sem desvio relevante para nenhum dos extremos

Complementarmente, `marker_frequencies` mede a frequência relativa de termos
de interesse (ex.: "estado", "mercado", "privatização") por autor/texto.
"""

from __future__ import annotations

import numpy as np

from ..errors import ScientificError
from ..logging_setup import get_logger

logger = get_logger(__name__)


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _word_count(text: str) -> int:
    return len(text.split())


class AuthorProfile:
    """Perfil de um autor de referência: centroide dos seus documentos."""

    def __init__(self, name: str, embeddings: np.ndarray, text: str = "") -> None:
        if embeddings.shape[0] == 0:
            raise ScientificError(
                f"Perfil '{name}' sem documentos para centroide",
                user_message=f"Não há documentos suficientes para montar o perfil de '{name}'.",
            )
        self.name = name
        self.centroid = _l2(embeddings.mean(axis=0))
        self.documents = int(embeddings.shape[0])
        self.text = text

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "centroid_l2": [float(v) for v in self.centroid],
            "documents": self.documents,
        }


class IdeologyRuler:
    """Eixo direita-esquerda definido por dois perfis de referência."""

    def __init__(self, left: AuthorProfile, right: AuthorProfile) -> None:
        if left.name == right.name:
            raise ScientificError(
                "Extremos da régua devem ser autores diferentes",
                user_message="Os dois extremos da régua precisam ser autores distintos.",
            )
        self.left = left
        self.right = right
        axis = right.centroid - left.centroid
        if float(np.linalg.norm(axis)) == 0:
            raise ScientificError(
                "Extremos com centroides idênticos",
                user_message="Os perfis de referência são idênticos; não dá para montar a régua.",
            )
        self.axis = _l2(axis)
        logger.info(
            "Régua ideológica construída",
            extra={
                "code": "ruler_built",
                "metric": f"{left.name}<->{right.name}",
                "doc_id": f"axis_dim={self.axis.size}",
            },
        )

    def position(self, embedding: np.ndarray) -> dict:
        """Posiciona um texto no eixo: escore calibrado, similaridades e classificação.

        O escore é normalizado pela distância entre os centroides (1 - cosseno),
        de modo que um texto idêntico ao extremo esquerdo vale -1, ao extremo
        direito vale +1, e o ponto equidistante vale 0.
        """
        vector = _l2(np.asarray(embedding, dtype=float))
        similarity_left = float(np.clip(vector @ self.left.centroid, -1.0, 1.0))
        similarity_right = float(np.clip(vector @ self.right.centroid, -1.0, 1.0))
        distance = 1.0 - float(self.left.centroid @ self.right.centroid)
        score = (similarity_right - similarity_left) / distance if distance > 1e-6 else 0.0
        score = float(np.clip(score, -1.0, 1.0))
        alignment = "direita" if score > 0.1 else "esquerda" if score < -0.1 else "centro"
        label = self.right.name if score > 0.1 else self.left.name if score < -0.1 else "centro"
        return {
            "score": round(score, 4),
            "label": label,
            "alignment": alignment,
            "similarity_left": round(similarity_left, 4),
            "similarity_right": round(similarity_right, 4),
        }

    def to_dict(self) -> dict:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "axis_l2": [float(v) for v in self.axis],
        }


def marker_frequencies(
    text: str,
    markers: tuple[str, ...] = (
        "estado",
        "mercado",
        "social",
        "privatização",
        "transparência",
        "redistribuição",
        "reforma",
        "imposto",
        "liberalização",
        "público",
        "privado",
        "regulação",
    ),
) -> dict[str, float]:
    """Frequência relativa (por 1000 palavras) de termos de interesse."""
    words = _word_count(text)
    if words == 0:
        return {}
    lowered = f" {text.lower()} "
    return {marker: round(lowered.count(f" {marker} "), 3) * 1000 / words for marker in markers}
