"""Análise bayesiana (Beta-Binomial) para proporções e diferenças.

Útil, por exemplo, para comparar a frequência relativa de um termo entre
dois corpora e estimar a probabilidade de uma verdadeira diferença
("probabilidade de superioridade"), com intervalo de credibilidade.

Fail first: contagens inválidas (k>n, negativas) são rejeitadas.
Inconsistências numéricas (amostras não-positivas) são registradas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ScientificError
from ..logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProportionEstimate:
    successes: int
    trials: int
    posterior_mean: float
    posterior_mode: float
    credible_interval: tuple[float, float]
    prior: tuple[float, float]


@dataclass(frozen=True)
class DifferenceEstimate:
    probability_p1_gt_p2: float
    posterior_mean_p1: float
    posterior_mean_p2: float
    samples: int


def bayesian_proportion(
    successes: int,
    trials: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credibility: float = 0.95,
) -> ProportionEstimate:
    """Estimativa bayesiana da proporção p ~ Beta(alpha + k, beta + n - k)."""
    if trials < 1:
        raise ScientificError(
            "Trials inválidos para proporção bayesiana",
            user_message="O número de tentativas (trials) precisa ser maior que zero.",
        )
    if not (0 <= successes <= trials):
        raise ScientificError(
            f"Sucessos ({successes}) fora do intervalo [0, {trials}]",
            user_message="A contagem de sucessos precisa estar entre 0 e o total de tentativas.",
        )
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ScientificError(
            "Hiperparâmetros do prior devem ser positivos",
            user_message="Os hiperparâmetros do prior precisam ser positivos.",
        )
    if not (0.0 < credibility < 1.0):
        raise ScientificError(
            "Nível de credibilidade fora de (0, 1)",
            user_message="O nível de credibilidade precisa estar entre 0 e 1.",
        )

    alpha_post = prior_alpha + successes
    beta_post = prior_beta + trials - successes

    mean = alpha_post / (alpha_post + beta_post)
    mode = (
        (alpha_post - 1) / (alpha_post + beta_post - 2)
        if alpha_post > 1 and beta_post > 1
        else mean
    )

    lower, upper = 0.05, 0.95
    if credibility < 1.0:
        tail = (1.0 - credibility) / 2.0
        lower, upper = tail, 1.0 - tail

    from scipy import stats as scipy_stats  # type: ignore[import-not-found]

    lo, hi = (
        scipy_stats.beta.ppf(lower, alpha_post, beta_post),
        scipy_stats.beta.ppf(upper, alpha_post, beta_post),
    )

    return ProportionEstimate(
        successes=successes,
        trials=trials,
        posterior_mean=float(mean),
        posterior_mode=float(mode),
        credible_interval=(float(lo), float(hi)),
        prior=(prior_alpha, prior_beta),
    )


def probability_of_superiority(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    *,
    samples: int = 20_000,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    seed: int | None = None,
) -> DifferenceEstimate:
    """Probabilidade de p_A > p_B via Monte Carlo (razão de betas)."""
    if samples < 1_000:
        raise ScientificError(
            f"Número de amostras insuficiente ({samples})",
            user_message="O número de amostras para Monte Carlo precisa ser maior que 1000.",
        )
    est_a = bayesian_proportion(successes_a, trials_a, prior_alpha, prior_beta)
    est_b = bayesian_proportion(successes_b, trials_b, prior_alpha, prior_beta)

    rng = np.random.default_rng(seed)
    alpha_a = prior_alpha + successes_a
    beta_a = prior_beta + trials_a - successes_a
    alpha_b = prior_alpha + successes_b
    beta_b = prior_beta + trials_b - successes_b

    from scipy import stats as scipy_stats  # type: ignore[import-not-found]

    draws_a = scipy_stats.beta.rvs(alpha_a, beta_a, size=samples, random_state=rng)
    draws_b = scipy_stats.beta.rvs(alpha_b, beta_b, size=samples, random_state=rng)
    prob = float(np.mean(draws_a > draws_b))

    return DifferenceEstimate(
        probability_p1_gt_p2=prob,
        posterior_mean_p1=est_a.posterior_mean,
        posterior_mean_p2=est_b.posterior_mean,
        samples=samples,
    )
