"""Testes da classificação de viés ideológico."""

from __future__ import annotations

import numpy as np

from editorial.scientific import (
    AuthorProfile,
    IdeologyRuler,
    bias_explanation,
    classify_position,
)


def _ruler():
    left = AuthorProfile("miriam", np.array([[1.0, 0.0], [0.9, 0.1]]))
    right = AuthorProfile("guedes", np.array([[-1.0, 0.0], [-0.9, -0.1]]))
    return IdeologyRuler(left, right)


def test_extreme_scores_classify_as_extremista():
    for score in (-1.0, -0.9, 0.95, 1.0):
        result = classify_position(score)
        assert result["primary"] == "extremista"


def test_neutral_scores_classify_as_ponderado():
    for score in (0.0, 0.05, -0.05):
        result = classify_position(score)
        assert result["primary"] == "ponderado"


def test_probabilities_sum_to_one():
    for score in (-0.4, 0.1, 0.6, 0.9):
        result = classify_position(score)
        total = sum(result["probabilities"].values())
        assert 0.999 <= total <= 1.001
        assert abs(total - 1.0) < 1e-2


def test_bias_percent_is_magnitude():
    assert classify_position(0.42)["bias_percent"] == 42.0
    assert classify_position(-0.42)["bias_percent"] == 42.0
    assert classify_position(0.0)["bias_percent"] == 0.0


def test_direction_from_score_sign():
    assert classify_position(0.5)["direction"] == "direita"
    assert classify_position(-0.5)["direction"] == "esquerda"
    assert classify_position(0.0)["direction"] == "centro"


def test_position_returns_calibrated_extremes():
    ruler = _ruler()
    assert ruler.position([1.0, 0.0])["score"] < -0.5
    assert ruler.position([-1.0, 0.0])["score"] > 0.5


def test_explanation_includes_method_and_class():
    ruler = _ruler()
    position = ruler.position([1.0, 0.0])
    classification = classify_position(position["score"])
    explanation = bias_explanation(position, classification, ruler, {"mercado": 1.0})
    assert "TF-IDF" in explanation
    assert "extremista" in explanation
    assert "mercado" in explanation
