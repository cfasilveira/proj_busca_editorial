"""Testes da validade integrada do discurso (contaminação + dissimulação)."""

from __future__ import annotations

from editorial.scientific import analyze_deception, assess_validity, classify_position


def _assess(bias_percent: float, deception_score: float) -> dict:
    """Monta entradas sintéticas para isolar o cálculo da validade."""
    position = {"score": bias_percent / 100.0, "similarity_left": 0.4, "similarity_right": 0.6}
    classification = classify_position(position["score"])
    deception = {
        "score": deception_score,
        "level": "baixo"
        if deception_score < 0.25
        else "moderado"
        if deception_score < 0.5
        else "alto",
        "signals": [],
        "weights": [],
        "explanation": "",
    }
    return assess_validity(
        position=position,
        classification=classification,
        deception=deception,
    )


def test_output_structure():
    result = _assess(5.0, 0.1)
    assert {
        "validity_score",
        "invalidity_score",
        "verdict",
        "components",
        "weights",
        "explanation",
    } <= set(result)
    assert result["verdict"] in {"válido", "comprometido", "invalidado"}
    assert 0.0 <= result["validity_score"] <= 1.0
    assert 0.0 <= result["invalidity_score"] <= 1.0
    assert abs(result["validity_score"] + result["invalidity_score"] - 1.0) < 1e-6


def test_valid_low_bias_low_deception():
    result = _assess(5.0, 0.1)
    assert result["verdict"] == "válido"


def test_compromised_moderate_bias_only():
    result = _assess(40.0, 0.1)
    assert result["verdict"] == "comprometido"
    assert result["components"]["dissimulation"]["level"] == "baixo"


def test_invalidated_high_bias_and_deception():
    result = _assess(70.0, 0.8)
    assert result["verdict"] == "invalidado"
    assert result["invalidity_score"] >= 0.5


def test_high_bias_alone_invalidates():
    result = _assess(70.0, 0.1)
    assert result["verdict"] == "invalidado"
    assert "enviesamento" in result["explanation"]


def test_dissimulation_dominant_when_bias_low():
    result = _assess(5.0, 0.8)
    assert result["verdict"] == "invalidado"
    assert "dissimulação" in result["explanation"].lower()


def test_explanation_auditable():
    result = _assess(30.0, 0.4)
    explanation = result["explanation"]
    assert "(1 -" in explanation and "dissimulação" in explanation
    assert result["verdict"] in explanation
    assert result["weights"]["model"] == "fuzzy_or"


def test_force_label_consistent_with_verdict():
    valid = _assess(5.0, 0.1)
    assert valid["verdict"] == "válido"
    assert "equilibrado" in valid["explanation"]
    assert "comprometimento" not in valid["explanation"]
    compromised = _assess(40.0, 0.1)
    assert compromised["verdict"] == "comprometido"
    assert "equilibrado" not in compromised["explanation"]


def test_integration_with_real_modules():
    text = "Ninguém nunca viu nada igual, jamais houve tal coisa. Eu juro que é verdade."
    deception = analyze_deception(text)
    position = {"score": 0.8, "similarity_left": 0.3, "similarity_right": 0.9}
    classification = classify_position(position["score"])
    result = assess_validity(position=position, classification=classification, deception=deception)
    assert result["verdict"] == "invalidado"
    assert result["components"]["dissimulation"]["value"] == deception["score"]
