"""Matriz de decisão ponderada (weighted decision matrix).

Dada uma lista de alternativas avaliadas em múltiplos critérios e um vetor
de pesos, calcula a pontuação ponderada e o ranking final.

Fail first: pesos que não somam ~1, critérios ausentes ou alternativas vazias
são rejeitados no início. Inconsistências numéricas (ex.: NaN) são
registradas em log.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import ScientificError, fail
from ..logging_setup import get_logger

logger = get_logger(__name__)

_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ScoredAlternative:
    name: str
    score: float
    rank: int
    details: Mapping[str, float] = field(default_factory=dict)


class WeightedDecisionMatrix:
    """Alternativas: dicts {alternativa: {criterio: valor}}. Pesos somam 1."""

    def __init__(
        self,
        alternatives: Mapping[str, Mapping[str, float]],
        criteria: Sequence[str],
        weights: Sequence[float],
    ) -> None:
        if not alternatives:
            raise ScientificError(
                "Matriz de decisão sem alternativas",
                user_message="Informe ao menos uma alternativa para a matriz de decisão.",
            )
        if not criteria:
            raise ScientificError(
                "Matriz de decisão sem critérios",
                user_message="Informe ao menos um critério para a matriz de decisão.",
            )
        if len(criteria) != len(weights):
            raise ScientificError(
                f"Quantidade de critérios ({len(criteria)}) difere de pesos ({len(weights)})",
                user_message="Critérios e pesos precisam ter a mesma quantidade de itens.",
            )
        if not math.isclose(sum(weights), 1.0, abs_tol=_WEIGHT_TOLERANCE):
            raise ScientificError(
                f"Pesos somam {sum(weights):.4f} (esperado ~1.0)",
                user_message="Os pesos da matriz precisam somar 1.0.",
            )

        missing = [
            f"{name}.{criterion}"
            for name, row in alternatives.items()
            for criterion in criteria
            if criterion not in row
        ]
        if missing:
            raise ScientificError(
                f"Critérios ausentes em alternativas: {', '.join(missing[:5])}",
                user_message="Há alternativas sem valor para algum dos critérios.",
            )

        self.alternatives = alternatives
        self.criteria = list(criteria)
        self.weights = list(weights)

    def score(self) -> list[ScoredAlternative]:
        results: list[ScoredAlternative] = []
        for name, row in self.alternatives.items():
            details: dict[str, float] = {}
            weighted = 0.0
            for criterion, weight in zip(self.criteria, self.weights, strict=True):
                value = float(row[criterion])
                if not math.isfinite(value):
                    fail(
                        logger,
                        ScientificError,
                        f"Valor inválido para {name}.{criterion}",
                        user_message="Há valores inválidos (NaN/Inf) na matriz de decisão.",
                        level=logging.WARNING,
                        code="non_finite_value",
                        doc_id=f"{name}.{criterion}",
                    )
                details[criterion] = value
                weighted += value * weight
            results.append(ScoredAlternative(name=name, score=weighted, rank=0, details=details))

        results.sort(key=lambda item: item.score, reverse=True)
        ranked = [replace(item, rank=rank) for rank, item in enumerate(results, start=1)]

        logger.info(
            "Matriz de decisão avaliada",
            extra={"code": "decision_matrix", "metric": f"{len(ranked)} alternativas"},
        )
        return ranked

    def consistency_check(self) -> dict[str, float]:
        """Checagem de consistência simples (dispersão dos pesos)."""
        arr = np.asarray(self.weights)
        return {
            "weight_mean": float(arr.mean()),
            "weight_std": float(arr.std()),
            "weight_max": float(arr.max()),
            "weight_min": float(arr.min()),
        }
