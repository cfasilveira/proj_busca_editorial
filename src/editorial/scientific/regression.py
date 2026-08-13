"""Regressão linear (OLS) e validação cruzada de R².

Permite modelar, por exemplo, uma variável ideológica a partir de
frequências de tokens. Fail first: tamanhos incompatíveis ou dados
insuficientes abortam. Falhas de validação cruzada são registradas em log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from ..errors import ScientificError
from ..logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float
    n: int
    residuals: list[float]


def _as_float_array(values: Sequence[float], label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ScientificError(
            f"Entrada '{label}' contém valores não-finitos",
            user_message=f"Os valores de '{label}' contêm NaN ou infinitos.",
        )
    return arr


def linear_regression(x: Sequence[float], y: Sequence[float]) -> RegressionResult:
    """Regressão linear simples y = slope*x + intercept via mínimos quadrados."""
    x_arr = _as_float_array(x, "x")
    y_arr = _as_float_array(y, "y")

    if x_arr.size != y_arr.size:
        raise ScientificError(
            f"Dimensões diferentes: x={x_arr.size}, y={y_arr.size}",
            user_message="As variáveis x e y precisam ter o mesmo tamanho.",
        )
    if x_arr.size < 2:
        raise ScientificError(
            "Regressão exige ao menos 2 pontos",
            user_message="A regressão precisa de ao menos dois pontos de dados.",
        )
    if np.all(x_arr == x_arr[0]):
        raise ScientificError(
            "Variável x constante: regressão indefinida",
            user_message="A variável x é constante, impossível ajustar uma regressão.",
        )

    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    predicted = slope * x_arr + intercept
    residuals = y_arr - predicted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return RegressionResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        n=int(x_arr.size),
        residuals=residuals.tolist(),
    )


def cross_validation_r2(x: Sequence[float], y: Sequence[float], folds: int = 3) -> dict[str, float]:
    """R² médio por k-fold, sinalizando instabilidade (variância alta)."""
    x_arr = _as_float_array(x, "x")
    y_arr = _as_float_array(y, "y")

    if x_arr.size != y_arr.size or x_arr.size < 2:
        raise ScientificError(
            "Dados insuficientes para validação cruzada",
            user_message="A validação cruzada precisa de dados compatíveis e suficientes.",
        )
    if folds < 2:
        raise ScientificError(
            "Validação cruzada exige ao menos 2 folds",
            user_message="O número de folds precisa ser maior ou igual a 2.",
        )

    try:
        scores = np.asarray(
            cross_val_score(
                LinearRegression(),
                x_arr.reshape(-1, 1),
                y_arr,
                cv=min(folds, int(x_arr.size)),
            )
        )
    except ValueError as exc:
        logger.warning(
            "Falha na validação cruzada",
            extra={"code": "crossval_failure", "metric": str(exc)},
        )
        raise ScientificError(
            "Todos os folds de validação cruzada falharam",
            user_message="A validação cruzada não conseguiu produzir nenhum fold válido.",
        ) from exc

    result = {
        "mean_r2": float(scores.mean()),
        "std_r2": float(scores.std()),
        "min_r2": float(scores.min()),
        "max_r2": float(scores.max()),
        "folds": int(scores.size),
    }

    if scores.std() > 0.25:
        logger.warning(
            "Alta variância entre folds de validação cruzada",
            extra={"code": "crossval_high_variance", "metric": f"std={scores.std():.3f}"},
        )
    logger.info(
        "Validação cruzada concluída",
        extra={"code": "crossval_done", "metric": f"mean_r2={result['mean_r2']:.3f}"},
    )
    return result
