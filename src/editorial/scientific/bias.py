"""Classificação de viés ideológico de um texto na régua esquerda-direita.

Dado o escore calibrado em [-1, 1] (ver `ideology.IdeologyRuler.position`),
a magnitude |score| mede o quanto o texto se alinha com um dos extremos da
régua (contaminação ideológica). A distribuição de probabilidade entre as
classes é derivada de kernels gaussianos centrados em cada classe:

    P(classe) ∝ exp( -( |score| - centro_classe)^2 / (2 * bandwidth^2) )

Classes (grau de viés):
    ponderado   -> quase neutro, próximo ao ponto de equilíbrio
    moderado    -> leve inclinação a um dos lados
    tendencioso -> inclinação clara a um dos lados
    extremista  -> forte aderência a um dos extremos da régua
"""

from __future__ import annotations

import numpy as np

CLASS_ORDER = ("ponderado", "moderado", "tendencioso", "extremista")
_CLASS_CENTERS = (0.05, 0.30, 0.50, 0.85)
_BANDWIDTH = 0.18


def classify_position(score: float) -> dict:
    """Classifica um escore: classe principal, probabilidades e viés %."""
    magnitude = float(np.clip(abs(score), 0.0, 1.0))
    expo = np.exp(-((magnitude - np.asarray(_CLASS_CENTERS)) ** 2) / (2 * _BANDWIDTH**2))
    raw = expo / expo.sum()
    probabilities = {name: round(float(p), 3) for name, p in zip(CLASS_ORDER, raw, strict=True)}
    # garante que as probabilidades somem exatamente 1.0
    remainder = 1.0 - sum(probabilities.values())
    probabilities[max(probabilities, key=probabilities.get)] += round(remainder, 3)
    direction = "direita" if score > 0.1 else "esquerda" if score < -0.1 else "centro"
    return {
        "primary": max(probabilities, key=probabilities.get),
        "probabilities": probabilities,
        "bias_percent": round(magnitude * 100.0, 1),
        "direction": direction,
    }


def _marker_summary(markers: dict[str, float]) -> str:
    relevant = [(k, v) for k, v in markers.items() if v > 0]
    if not relevant:
        return "Nenhum termo de interesse identificado no texto."
    formatted = ", ".join(
        f"{k} ({v:.1f}/mil palavras)"
        for k, v in sorted(relevant, key=lambda kv: kv[1], reverse=True)
    )
    return f"Termos de interesse no texto: {formatted}."


def bias_explanation(position: dict, classification: dict, ruler, markers: dict) -> str:
    """Explicação científica, em texto, de como a classificação foi obtida."""
    magnitude = abs(position["score"])
    parts = [
        (
            f"O texto foi posicionado na régua ideológica entre '{ruler.left.name}' "
            f"(extremo à esquerda, {ruler.left.documents} documentos de referência) e "
            f"'{ruler.right.name}' (extremo à direita, {ruler.right.documents} documentos), "
            f"em um espaço vetorial TF-IDF+SVD de {ruler.axis.size} dimensões."
        ),
        (
            f"Escore calibrado: {position['score']:+.2f} (escala -1..+1; -1 = extremo esquerdo, "
            f"+1 = extremo direito, 0 = equilíbrio). Similaridade com {ruler.left.name}: "
            f"{position['similarity_left']:.2f}; com {ruler.right.name}: "
            f"{position['similarity_right']:.2f}. Inclinação: {classification['direction']}."
        ),
        (
            f"A magnitude {magnitude:.2f} é interpretada como contaminação ideológica de "
            f"{classification['bias_percent']:.0f}%. Distribuição probabilística por classe "
            f"(kernels gaussianos): "
            + ", ".join(
                f"{name}={classification['probabilities'][name] * 100:.0f}%" for name in CLASS_ORDER
            )
            + f". Classe principal: {classification['primary']}."
        ),
        _marker_summary(markers),
        (
            "Atenção: a régua é uma aproximação construída sobre os corpora de referência. "
            f"Com apenas {ruler.left.documents} e {ruler.right.documents} documentos por extremo, "
            "o eixo captura o estilo e o vocabulário dessas amostras; ampliar os corpora de "
            "referência aumenta a confiabilidade e a generalização da classificação."
        ),
    ]
    return "\n".join(parts)
