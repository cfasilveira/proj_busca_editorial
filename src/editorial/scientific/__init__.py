"""Módulo Científico: matriz de decisão, regressão, bayesiana, régua, viés, engano e validade."""

from .bayesian import (
    DifferenceEstimate,
    ProportionEstimate,
    bayesian_proportion,
    probability_of_superiority,
)
from .bias import CLASS_ORDER, bias_explanation, classify_position
from .deception import SignalResult, analyze_deception
from .decision_matrix import ScoredAlternative, WeightedDecisionMatrix
from .ideology import AuthorProfile, IdeologyRuler, marker_frequencies
from .regression import RegressionResult, cross_validation_r2, linear_regression
from .validity import assess_validity

__all__ = [
    "CLASS_ORDER",
    "AuthorProfile",
    "DifferenceEstimate",
    "IdeologyRuler",
    "ProportionEstimate",
    "RegressionResult",
    "ScoredAlternative",
    "SignalResult",
    "WeightedDecisionMatrix",
    "analyze_deception",
    "assess_validity",
    "bayesian_proportion",
    "bias_explanation",
    "classify_position",
    "cross_validation_r2",
    "linear_regression",
    "marker_frequencies",
    "probability_of_superiority",
]
