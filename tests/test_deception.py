"""Testes da análise cognitiva de padrões de engano."""

from __future__ import annotations

import pytest

from editorial.errors import ProcessingError
from editorial.scientific import analyze_deception


def test_empty_text_fails_first():
    with pytest.raises(ProcessingError, match="vazio"):
        analyze_deception("   ")
    with pytest.raises(ProcessingError):
        analyze_deception(None)


def test_output_structure():
    result = analyze_deception("Um texto comum de exemplo, sem muitos sinais.")
    assert {"score", "level", "signals", "weights", "explanation"} <= set(result)
    assert 0.0 <= result["score"] <= 1.0
    assert result["level"] in {"baixo", "moderado", "alto"}
    assert len(result["signals"]) == 6


def test_justification_markers_increase_signal():
    plain = analyze_deception("Eu fui ao mercado e comprei pão.")
    defensive = analyze_deception(
        "Eu juro que fui ao mercado. Na verdade, sinceramente, garanto que comprei pão."
    )
    plain_signal = next(s for s in plain["signals"] if s["name"] == "repeticao_justificativas")
    defensive_signal = next(
        s for s in defensive["signals"] if s["name"] == "repeticao_justificativas"
    )
    assert defensive_signal["score"] > plain_signal["score"]


def test_vague_terms_increase_signal():
    plain = analyze_deception("Compramos o equipamento na semana passada.")
    vague = analyze_deception("Talvez alguém tenha visto uma coisa por aí, mais ou menos.")
    plain_signal = next(s for s in plain["signals"] if s["name"] == "termos_vagos")
    vague_signal = next(s for s in vague["signals"] if s["name"] == "termos_vagos")
    assert vague_signal["score"] > plain_signal["score"]


def test_contradiction_patterns_detected():
    result = analyze_deception("Eu disse que viria, mas não vim. Contudo nunca prometi nada.")
    signal = next(s for s in result["signals"] if s["name"] == "contradicoes_internas")
    assert signal["evidence"]["padroes_contraste_negacao"] >= 2


def test_weights_are_auditable():
    result = analyze_deception("Texto qualquer para auditoria.")
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_unpunctuated_transcript_zeroes_cognitive_load():
    transcript = (
        "olha né eu acho que talvez a gente veja mais ou menos esse resultado né "
        "dessa empresa que cresceu crescendo aí então enfim a gente não espera "
        "grandes surpresas talvez de certa forma obviamente né"
    )
    result = analyze_deception(transcript)
    signal = next(s for s in result["signals"] if s["name"] == "carga_cognitiva")
    assert signal["score"] == 0.0
    assert signal["evidence"]["artefato_transcricao"] is True
    assert "transcrição" in signal["evidence"]["motivo"]


def test_punctuated_text_still_measures_cognitive_load():
    punctuated = (
        "O resultado veio acima das nossas expectativas; revisamos o modelo e "
        "subimos as estimativas em cerca de trinta por cento. O mercado ainda é "
        "muito pouco penetrado, então a tese de crescimento permanece intacta. "
        "Continuamos com recomendação de compra."
    )
    result = analyze_deception(punctuated)
    signal = next(s for s in result["signals"] if s["name"] == "carga_cognitiva")
    assert signal["evidence"].get("artefato_transcricao") is not True
    assert signal["score"] > 0.0
    assert signal["evidence"]["frases"] >= 3


def test_transcript_explanation_mentions_artifact():
    transcript = "olha né talvez mais ou menos né obviamente sem pontuação"
    result = analyze_deception(transcript)
    assert "transcrição" in result["explanation"]


def test_unsupported_claims_increase_signal():
    plain = analyze_deception("A empresa divulgou os resultados do trimestre passado.")
    absolute = analyze_deception(
        "Ninguém jamais viu nada igual. Nunca houve tal coisa na história. "
        "Todo mundo sabe disso, com certeza absoluta."
    )
    plain_signal = next(s for s in plain["signals"] if s["name"] == "afirmacoes_sem_suporte")
    absolute_signal = next(s for s in absolute["signals"] if s["name"] == "afirmacoes_sem_suporte")
    assert absolute_signal["score"] > plain_signal["score"]
    assert absolute_signal["evidence"]["ocorrencias"] >= 5


def test_moral_universals_discounted_from_empirical_absolutes():
    moral = analyze_deception(
        "Todos os homens são livres. Sempre lutaremos pela justiça. "
        "Jamais desistiremos de nossos ideais."
    )
    empirical = analyze_deception(
        "Nenhuma chance de melhora nos dados: 42, 87 e 13 confirmam. "
        "Absolutamente impossível resolver agora, é garantido."
    )
    moral_signal = next(s for s in moral["signals"] if s["name"] == "afirmacoes_sem_suporte")
    empirical_signal = next(
        s for s in empirical["signals"] if s["name"] == "afirmacoes_sem_suporte"
    )
    assert empirical_signal["score"] > moral_signal["score"]
    assert empirical_signal["evidence"]["absolutos_fortes"] >= 3
    assert empirical_signal["evidence"]["especificidade"] > 0.5
    assert moral_signal["evidence"]["universais_fracos"] >= 2
    assert moral_signal["evidence"]["especificidade"] == 0.0


def test_signal_names_and_weights_consistent():
    result = analyze_deception("Texto para auditoria dos sinais.")
    names = [s["name"] for s in result["signals"]]
    assert names[-1] == "afirmacoes_sem_suporte"
    assert len(result["weights"]) == len(names)
    assert abs(sum(result["weights"]) - 1.0) < 1e-6
