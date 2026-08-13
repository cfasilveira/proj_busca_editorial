"""Regressão linear (OLS) e validação cruzada de R².

Permite modelar, por exemplo, uma variável ideológica a partir de
frequências de tokens. Fail first: tamanhos incompatíveis ou dados
insuficientes abortam. Falhas de validação cruzada são registradas em log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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
        logger.warning(
            f"Entrada '{label}' contém valores não-finitos",
            extra={"code": "non_finite_input", "metric": label},
        )
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

    n = int(x_arr.size)
    folds = min(folds, n)
    fold_sizes = np.full(folds, n // folds, dtype=int)
    fold_sizes[: n % folds] += 1

    r2_scores: list[float] = []
    index = 0
    for fold_size in fold_sizes:
        test_idx = np.arange(index, index + fold_size)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        index += fold_size

        try:
            x_train, y_train = x_arr[train_idx], y_arr[train_idx]
            x_test, y_test = x_arr[test_idx], y_arr[test_idx]

            slope, intercept = np.polyfit(x_train, y_train, 1)
            predicted = slope * x_test + intercept
            ss_res = float(np.sum((y_test - predicted) ** 2))
            ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
            r2_scores.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning(
                "Falha em fold de validação cruzada",
                extra={"code": "crossval_fold_failure", "metric": str(exc)},
            )
            continue

    if not r2_scores:
        raise ScientificError(
            "Todos os folds de validação cruzada falharam",
            user_message="A validação cruzada não conseguiu produzir nenhum fold válido.",
        )

    arr = np.asarray(r2_scores)
    result = {
        "mean_r2": float(arr.mean()),
        "std_r2": float(arr.std()),
        "min_r2": float(arr.min()),
        "max_r2": float(arr.max()),
        "folds": len(r2_scores),
    }

    if arr.std() > 0.25:
        logger.warning(
            "Alta variância entre folds de validação cruzada",
            extra={"code": "crossval_high_variance", "metric": f"std={arr.std():.3f}"},
        )
    logger.info(
        "Validação cruzada concluída",
        extra={"code": "crossval_done", "metric": f"mean_r2={result['mean_r2']:.3f}"},
    )
    return result
