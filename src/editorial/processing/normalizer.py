"""Normalização estatística de corpus.

Provê estatísticas descritivas (comprimentos, frequência relativa de tokens)
e normalização de vetores/valores (min-max, z-score, L2). Textos vazios
são registrados em log; se o corpus inteiro for inválido, é levantado
ProcessingError (fail first no nível de coleção).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from ..errors import ProcessingError
from ..logging_setup import get_logger

logger = get_logger(__name__)


def token_frequencies(token_lists: Iterable[Sequence[str]]) -> Counter[str]:
    """Conta tokens e calcula frequência relativa por documento e no corpus."""
    corpus = Counter()
    for tokens in token_lists:
        corpus.update(tokens)

    if not corpus:
        logger.warning(
            "Corpus sem tokens para contagem de frequência",
            extra={"code": "empty_corpus_frequencies"},
        )

    return corpus


def relative_frequencies(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {token: count / total for token, count in counter.items()}


def length_stats(lengths: Sequence[int]) -> dict[str, float]:
    """Média, desvio padrão, min/max dos comprimentos dos documentos."""
    if not lengths:
        raise ProcessingError(
            "Nenhum documento para calcular estatísticas de comprimento",
            user_message="Não há documentos para calcular as estatísticas.",
        )
    arr = np.asarray(lengths, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def z_scores(values: Sequence[float]) -> list[float]:
    """Padroniza uma sequência de valores (z-score)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    std = arr.std()
    if std == 0:
        return [0.0 for _ in arr]
    return ((arr - arr.mean()) / std).tolist()


def min_max_normalize(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo == 0:
        return [0.0 for _ in arr]
    return ((arr - lo) / (hi - lo)).tolist()


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def profile(lengths: Sequence[int], token_lists: Iterable[Sequence[str]]) -> dict:
    """Perfil linguístico agregado do corpus (ex.: estilo editorial)."""
    freqs = relative_frequencies(token_frequencies(token_lists))
    top = sorted(freqs.items(), key=lambda item: item[1], reverse=True)[:20]

    stats = length_stats(lengths)
    logger.info(
        "Perfil linguístico calculado",
        extra={"code": "linguistic_profile", "metric": f"{len(top)} top tokens"},
    )
    return {
        "length_stats": stats,
        "lexical_diversity": len(freqs) / stats["mean"] if stats["mean"] else 0.0,
        "top_tokens": [{"token": token, "frequency": freq} for token, freq in top],
    }
