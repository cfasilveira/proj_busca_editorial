"""Módulo Científico: matriz de decisão, regressão e análise bayesiana."""

from .bayesian import (
    DifferenceEstimate,
    ProportionEstimate,
    bayesian_proportion,
    probability_of_superiority,
)
from .decision_matrix import ScoredAlternative, WeightedDecisionMatrix
from .regression import RegressionResult, cross_validation_r2, linear_regression

__all__ = [
    "DifferenceEstimate",
    "ProportionEstimate",
    "RegressionResult",
    "ScoredAlternative",
    "WeightedDecisionMatrix",
    "bayesian_proportion",
    "cross_validation_r2",
    "linear_regression",
    "probability_of_superiority",
]
