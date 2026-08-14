"""Validade integrada do discurso: contaminação ideológica + dissimulação.

Consolida a ideia central do projeto: um texto editorial é "empobrecido e
invalidado" na medida em que sua argumentação é contaminada por política,
ideologia e sociologia (eixo `bias`) e/ou passa a depender de mentiras,
afirmações desacreditadas e padrões de dissimulação (eixo `deception`).

O grau de invalidade segue a lógica **OR difusa**: um discurso só permanece
válido se os dois eixos estiverem limpos — basta um comprometido para que a
validade caia. Formalmente:

    invalidade = 1 - (1 - contaminação) * (1 - dissimulação)

com veredito:
    válido        (invalidade < 0.30)
    comprometido  (0.30 <= invalidade < 0.50)
    invalidado    (invalidade >= 0.50)

A interpretação da "força dominante" (enviesamento, dissimulação ou ambas)
ajuda a distinguir um texto tendencioso porém transparente de um texto que
oculta ou desacredita fatos.
"""

from __future__ import annotations

from typing import Any

_VERDICTS = (
    ("válido", 0.30),
    ("comprometido", 0.50),
    ("invalidado", 1.01),
)


def _interpret(invalidity: float) -> str:
    for name, limit in _VERDICTS:
        if invalidity < limit:
            return name
    return "invalidado"


def _dominant_force(verdict: str, contamination: float, dissimulation: float) -> str:
    if verdict == "válido":
        return "discurso equilibrado (baixa contaminação e baixo engano)"
    if verdict == "invalidado":
        if contamination >= 0.30 and dissimulation >= 0.30:
            return "dissimulação ideológica (contaminação e engano combinados)"
        if contamination >= 0.30:
            return "enviesamento ideológico explícito (contaminação alta, engano baixo)"
        return "dissimulação discursiva (engano alto, contaminação baixa)"
    if contamination >= 0.30:
        return "enviesamento ideológico (contaminação alta, engano baixo)"
    if dissimulation >= 0.30:
        return "dissimulação discursiva (engano alto, contaminação baixa)"
    return "acúmulo de contaminação e dissimulação moderados"


def _explain(
    invalidity: float,
    verdict: str,
    contamination: float,
    dissimulation: float,
    classification: dict,
    deception: dict,
) -> str:
    force = _dominant_force(verdict, contamination, dissimulation)
    return (
        "A validade do discurso consolida dois eixos da dissimulação: "
        f"contaminação ideológica ({classification['bias_percent']:.0f}%, classe "
        f"'{classification['primary']}', inclinação {classification['direction']}) e "
        f"padrões de engano (score {deception['score']:.2f}, nível "
        f"'{deception['level']}'). Pela lógica OR difusa, a invalidade é "
        f"1 - (1 - {contamination:.2f}) * (1 - {dissimulation:.2f}) = {invalidity:.2f}. "
        f"Veredito: '{verdict}'. Força dominante: {force}. Um discurso é empobrecido "
        "e invalidado quando a argumentação passa a depender de adesão a um polo "
        "ideológico ou de afirmações absolutas sem suporte, em vez de evidências."
    )


def assess_validity(
    *,
    position: dict[str, Any],
    classification: dict[str, Any],
    deception: dict[str, Any],
) -> dict[str, Any]:
    """Combina contaminação ideológica e dissimulação em um veredito de validade."""
    contamination = min(1.0, float(classification["bias_percent"]) / 100.0)
    dissimulation = float(deception["score"])
    invalidity = round(1.0 - (1.0 - contamination) * (1.0 - dissimulation), 3)
    validity = round(1.0 - invalidity, 3)
    verdict = _interpret(invalidity)

    return {
        "validity_score": validity,
        "invalidity_score": invalidity,
        "verdict": verdict,
        "components": {
            "contamination": {
                "value": round(contamination, 3),
                "bias_percent": classification["bias_percent"],
                "bias_class": classification["primary"],
                "direction": classification["direction"],
                "position_score": position["score"],
            },
            "dissimulation": {
                "value": dissimulation,
                "level": deception["level"],
            },
        },
        "weights": {
            "bias": 1.0,
            "dissimulation": 1.0,
            "model": "fuzzy_or",
        },
        "explanation": _explain(
            invalidity,
            verdict,
            contamination,
            dissimulation,
            classification,
            deception,
        ),
    }
