"""Testes do módulo Científico."""

from __future__ import annotations

import pytest

from editorial.errors import ScientificError
from editorial.scientific import (
    WeightedDecisionMatrix,
    bayesian_proportion,
    cross_validation_r2,
    linear_regression,
    probability_of_superiority,
)


class TestDecisionMatrix:
    def test_ranking_weights_sum_to_one(self):
        matrix = WeightedDecisionMatrix(
            alternatives={"A": {"c1": 1.0, "c2": 0.0}, "B": {"c1": 0.0, "c2": 1.0}},
            criteria=["c1", "c2"],
            weights=[0.8, 0.2],
        )
        scored = matrix.score()
        assert scored[0].name == "A"
        assert scored[1].name == "B"
        assert scored[0].rank == 1

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ScientificError, match="somam"):
            WeightedDecisionMatrix(
                alternatives={"A": {"c1": 1.0}},
                criteria=["c1"],
                weights=[0.5],
            )

    def test_missing_criteria_fails_first(self):
        with pytest.raises(ScientificError, match="ausentes"):
            WeightedDecisionMatrix(
                alternatives={"A": {"c1": 1.0}},
                criteria=["c1", "c2"],
                weights=[0.5, 0.5],
            )

    def test_empty_alternatives_fails_first(self):
        with pytest.raises(ScientificError):
            WeightedDecisionMatrix({}, ["c1"], [1.0])

    def test_non_finite_value_fails(self):
        matrix = WeightedDecisionMatrix(
            alternatives={"A": {"c1": float("nan")}},
            criteria=["c1"],
            weights=[1.0],
        )
        with pytest.raises(ScientificError, match="inválido"):
            matrix.score()


class TestRegression:
    def test_perfect_line(self):
        result = linear_regression([1, 2, 3, 4], [2, 4, 6, 8])
        assert result.slope == pytest.approx(2.0)
        assert result.r_squared == pytest.approx(1.0)
        assert result.n == 4

    def test_dimension_mismatch_fails(self):
        with pytest.raises(ScientificError, match="Dimensões"):
            linear_regression([1, 2, 3], [1, 2])

    def test_too_few_points_fails(self):
        with pytest.raises(ScientificError, match="2 pontos"):
            linear_regression([1], [1])

    def test_constant_x_fails(self):
        with pytest.raises(ScientificError, match="constante"):
            linear_regression([2, 2, 2], [1, 2, 3])

    def test_cross_validation(self):
        result = cross_validation_r2([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], folds=3)
        assert result["mean_r2"] == pytest.approx(1.0)
        assert result["folds"] == 3


class TestBayesian:
    def test_proportion_interval_contains_posterior_mean(self):
        est = bayesian_proportion(30, 100, prior_alpha=2, prior_beta=2)
        assert est.posterior_mean > 0
        assert est.credible_interval[0] <= est.posterior_mean <= est.credible_interval[1]

    def test_successes_out_of_range_fails(self):
        with pytest.raises(ScientificError, match="fora do intervalo"):
            bayesian_proportion(101, 100)

    def test_trials_zero_fails(self):
        with pytest.raises(ScientificError):
            bayesian_proportion(0, 0)

    def test_superiority_direction(self):
        est = probability_of_superiority(80, 100, 20, 100, samples=10_000, seed=1)
        assert est.probability_p1_gt_p2 > 0.9
        assert est.posterior_mean_p1 > est.posterior_mean_p2

    def test_superiority_reversed(self):
        est = probability_of_superiority(20, 100, 80, 100, samples=10_000, seed=1)
        assert est.probability_p1_gt_p2 < 0.1

    def test_superiority_rejects_few_samples(self):
        with pytest.raises(ScientificError, match="amostras"):
            probability_of_superiority(10, 100, 5, 100, samples=10)
