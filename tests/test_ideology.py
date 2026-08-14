"""Testes da régua ideológica."""

from __future__ import annotations

import numpy as np
import pytest

from editorial.errors import ScientificError
from editorial.scientific import AuthorProfile, IdeologyRuler, marker_frequencies


def _profiles():
    left = AuthorProfile("miriam", np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]))
    right = AuthorProfile("guedes", np.array([[-1.0, 0.0], [-0.9, -0.1], [-0.8, -0.2]]))
    return left, right


def test_builds_ruler_and_positions_left_text():
    left, right = _profiles()
    ruler = IdeologyRuler(left, right)
    pos = ruler.position([1.0, 0.0])
    assert pos["score"] < 0
    assert pos["alignment"] == "esquerda"
    assert pos["label"] == "miriam"


def test_positions_right_text():
    left, right = _profiles()
    ruler = IdeologyRuler(left, right)
    pos = ruler.position([-1.0, 0.0])
    assert pos["score"] > 0
    assert pos["alignment"] == "direita"
    assert pos["label"] == "guedes"


def test_center_position_is_neutral():
    left, right = _profiles()
    ruler = IdeologyRuler(left, right)
    pos = ruler.position([0.0, 0.0])
    assert pos["alignment"] == "centro"


def test_same_extremes_raise():
    profile = AuthorProfile("x", np.array([[1.0, 0.0]]))
    with pytest.raises(ScientificError):
        IdeologyRuler(profile, AuthorProfile("x", np.array([[1.0, 0.0]])))


def test_empty_profile_raises():
    with pytest.raises(ScientificError):
        AuthorProfile("vazio", np.empty((0, 2)))


def test_marker_frequencies():
    freqs = marker_frequencies("o estado e o mercado, estado de mercado")
    assert freqs["estado"] > freqs["mercado"]
    assert freqs["estado"] > 0
    assert freqs["privatização"] == 0.0
