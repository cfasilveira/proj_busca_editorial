"""Análise cognitiva de padrões linguísticos associados a engano/mentira.

Baseia-se em evidências de psicologia cognitiva, neurolinguística e análise de
discurso: repetição de justificativas, excesso de detalhes irrelevantes,
contradições internas, termos vagos, carga cognitiva elevada e afirmações
absolutas sem suporte (proxy léxico de afirmações desacreditadas).

Projetada para ser **leve e segura**: varre o texto uma única vez (regex sobre
tokens de superfície), não treina nem carrega modelos pesados e trabalha com
scores parciais normalizados por 1000 palavras. Os scores parciais são
combinados por uma `WeightedDecisionMatrix` (auditável), seguindo a filosofia
fail first / fail gracefully do projeto.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..errors import ProcessingError
from ..logging_setup import get_logger
from .decision_matrix import WeightedDecisionMatrix

logger = get_logger(__name__)

# --- Marcadores (heurísticas científicas da literatura) ---

_JUSTIFICATION_MARKERS: tuple[str, ...] = (
    "eu juro",
    "na verdade",
    "honestamente",
    "sinceramente",
    "verdadeiramente",
    "juro por",
    "palavra de honra",
    "acredite em mim",
    "pode acreditar",
    "pode ter certeza",
    "tenha certeza",
    "tenho certeza",
    "temos certeza",
    "claro que",
    "certamente",
    "garanto que",
    "sem nenhum",
    "sem nenhuma",
    "todo mundo sabe",
    "é claro que",
    "obviamente",
    "sem dúvida",
    "com certeza",
)

_VAGUE_MARKERS: tuple[str, ...] = (
    "alguém",
    "coisa",
    "coisas",
    "talvez",
    "provavelmente",
    "mais ou menos",
    "em algum lugar",
    "em algum momento",
    "sei lá",
    "tipo assim",
    "depende",
    "nem sei",
    "quem sabe",
    "de certa forma",
    "digamos",
    "muita gente",
    "em geral",
)

_NEGATION_MARKERS: tuple[str, ...] = (
    "não",
    "nunca",
    "jamais",
    "nem",
    "ninguém",
    "nada",
    "sem",
    "nenhum",
    "nenhuma",
)

# Absolutos empíricos (fortes): asseverações de certeza e negações absolutas
# factuais ("nunca houve", "nenhuma chance", "absolutamente impossível") — em
# contexto factual, são afirmações desacreditadas sem suporte.
_ABS_STRONG_MARKERS: tuple[str, ...] = (
    "nunca",
    "jamais",
    "ninguém",
    "nada",
    "nenhum",
    "nenhuma",
    "absolutamente",
    "totalmente",
    "completamente",
    "impossível",
    "garantido",
    "com certeza",
    "sem dúvida",
    "inegavelmente",
    "inquestionavelmente",
)

# Quantificadores universais (fracos): comuns em princípios morais e verdades
# gerais ("todos os homens", "sempre", "tudo"), não são necessariamente
# afirmações factuais sem suporte. Entram com peso reduzido.
_ABS_WEAK_MARKERS: tuple[str, ...] = (
    "sempre",
    "todos",
    "tudo",
    "todo mundo",
    "qualquer",
)
_WEAK_ABS_WEIGHT = 0.35

# Modulador de especificidade: quando o texto não traz dados específicos
# (números), os absolutos tendem a ser universais morais/retóricos; quando
# apresenta números, a combinação "sobre-generalização + sobre-especificidade"
# caracteriza afirmação sem suporte. O fator varia em [0.35, 1.0].
_MIN_SPECIFICITY_FACTOR = 0.35

_CONTRADICTION_RE = re.compile(
    r"\b(?:mas|porém|contudo|entretanto|todavia)\s+(?:não|nunca|jamais|nem)\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"\b\d[\d.,%]*\b")
_WORD_RE = re.compile(r"\b[\wÀ-ÿ]+(?:[-']\w+)*\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[.!?…]+\s*")

# taxa mínima de pontuação (terminadores + pausas por 1000 palavras) para que
# comprimento de frase seja mensurável; abaixo disso o texto é tratado como
# transcrição sem pontuação (artefato) e o sinal de carga cognitiva é zerado.
_MIN_PUNCTUATION_RATE = 20.0

# Pesos padrão dos sinais (contradição é o sinal mais forte; afirmações
# absolutas sem suporte são o segundo, por serem o proxy léxico mais direto de
# afirmações desacreditadas).
_SIGNAL_WEIGHTS = (0.15, 0.15, 0.25, 0.10, 0.15, 0.20)
_SIGNAL_NAMES = (
    "repeticao_justificativas",
    "termos_vagos",
    "contradicoes_internas",
    "excesso_detalhes",
    "carga_cognitiva",
    "afirmacoes_sem_suporte",
)
# meia-vida (k) de cada sinal: taxa (por 1000 palavras) que dá score 0.5
_SIGNAL_HALF = (4.0, 8.0, 1.5, 20.0, None, 8.0)

_LEVELS = (
    ("baixo", 0.25),
    ("moderado", 0.50),
    ("alto", 1.01),
)


def _rate_to_score(rate: float, half: float) -> float:
    """Mapeia taxa (ocorrências/1000 palavras) para score em [0, 1] via saturação."""
    return rate / (rate + half)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_markers(text: str, markers: tuple[str, ...]) -> tuple[int, dict[str, int]]:
    total = 0
    evidence: dict[str, int] = {}
    lowered = f" {text.lower()} "
    for marker in markers:
        count = lowered.count(f" {marker} ")
        if count:
            evidence[marker] = count
            total += count
    return total, evidence


@dataclass(frozen=True)
class SignalResult:
    name: str
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)


def signal_repetition(text: str, words: int) -> SignalResult:
    """Repetição de justificativas/termos de reforço de credibilidade."""
    count, evidence = _count_markers(text, _JUSTIFICATION_MARKERS)
    rate = count * 1000 / words if words else 0.0
    return SignalResult(
        "repeticao_justificativas",
        round(_rate_to_score(rate, _SIGNAL_HALF[0]), 3),
        {"ocorrencias": count, "taxa_mil": round(rate, 2), "termos": evidence},
    )


def signal_vague_terms(text: str, words: int) -> SignalResult:
    """Uso de termos vagos (evita compromisso com fatos verificáveis)."""
    count, evidence = _count_markers(text, _VAGUE_MARKERS)
    rate = count * 1000 / words if words else 0.0
    return SignalResult(
        "termos_vagos",
        round(_rate_to_score(rate, _SIGNAL_HALF[1]), 3),
        {"ocorrencias": count, "taxa_mil": round(rate, 2), "termos": evidence},
    )


def signal_contradictions(text: str, words: int) -> SignalResult:
    """Contradições internas: contraste seguido de negação ('mas não', 'porém nunca')."""
    lowered = text.lower()
    patterns = len(_CONTRADICTION_RE.findall(lowered))
    negation = sum(lowered.count(f" {m} ") for m in _NEGATION_MARKERS)
    rate = patterns * 1000 / words if words else 0.0
    return SignalResult(
        "contradicoes_internas",
        round(_rate_to_score(rate, _SIGNAL_HALF[2]), 3),
        {
            "padroes_contraste_negacao": patterns,
            "termos_negacao": negation,
            "taxa_mil": round(rate, 2),
        },
    )


def signal_excess_details(text: str, words: int) -> SignalResult:
    """Excesso de detalhes irrelevantes: densidade de números/precisão numérica."""
    numerics = len(_NUMERIC_RE.findall(text))
    rate = numerics * 1000 / words if words else 0.0
    return SignalResult(
        "excesso_detalhes",
        round(_rate_to_score(rate, _SIGNAL_HALF[3]), 3),
        {"numeros": numerics, "taxa_mil": round(rate, 2)},
    )


def _punctuation_rate(text: str, words: int) -> float:
    terminators = len(_SENTENCE_RE.findall(text))
    clauses = sum(text.count(p) for p in (",", ";", ":"))
    return (terminators + clauses) * 1000 / words if words else 0.0


def signal_cognitive_load(text: str, words: int) -> SignalResult:
    """Carga cognitiva: frases longas e com muitas pausas (mais esforço mental).

    Transcrições de fala costumam não ter pontuação; nesses casos o comprimento
    de "frase" e as pausas são artefatos do formato, não sinais reais de esforço,
    então o sinal é zerado e marcado como `artefato_transcricao` na evidência.
    """
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    avg_len = (words / len(sentences)) if sentences else 0.0
    clauses = sum(text.count(p) for p in (",", ";", ":"))
    clause_rate = clauses * 1000 / words if words else 0.0
    punct_rate = _punctuation_rate(text, words)
    unpunctuated = punct_rate < _MIN_PUNCTUATION_RATE

    if unpunctuated:
        score = 0.0
        evidence = {
            "frases": len(sentences),
            "tokens_medio_frase": round(avg_len, 1),
            "pausas": clauses,
            "artefato_transcricao": True,
            "motivo": "texto sem pontuação de frases (transcrição de fala)",
            "taxa_pontuacao_mil": round(punct_rate, 2),
        }
    else:
        # normaliza comprimento médio (linha ~15-60 tokens) e pausas
        length_component = min(1.0, max(0.0, (avg_len - 15.0) / 45.0))
        pause_component = clause_rate / (clause_rate + 12.0)
        score = round(0.6 * length_component + 0.4 * pause_component, 3)
        evidence = {
            "frases": len(sentences),
            "tokens_medio_frase": round(avg_len, 1),
            "pausas": clauses,
        }

    return SignalResult("carga_cognitiva", score, evidence)


def signal_unsupported_claims(text: str, words: int) -> SignalResult:
    """Afirmações absolutas sem suporte: universais/definitivas que asseveram sem evidência.

    Distingue dois grupos: absolutos empíricos (asseverações de certeza e
    negações absolutas factuais, peso 1.0) e quantificadores universais (morais/
    retóricos, peso reduzido). Ambos são descontados por um modulador de
    especificidade: sem números/dados no texto, absolutos tendem a ser universais
    éticos ("todos os homens", "jamais"), não afirmações factuais sem suporte.
    """
    strong, strong_terms = _count_markers(text, _ABS_STRONG_MARKERS)
    weak, weak_terms = _count_markers(text, _ABS_WEAK_MARKERS)
    numerics = len(_NUMERIC_RE.findall(text))
    num_rate = numerics * 1000 / words if words else 0.0
    specificity = num_rate / (num_rate + 20.0)
    factor = _MIN_SPECIFICITY_FACTOR + (1.0 - _MIN_SPECIFICITY_FACTOR) * specificity
    weighted = strong + weak * _WEAK_ABS_WEIGHT
    effective_rate = weighted * factor * 1000 / words if words else 0.0
    return SignalResult(
        "afirmacoes_sem_suporte",
        round(_rate_to_score(effective_rate, _SIGNAL_HALF[5]), 3),
        {
            "ocorrencias": strong + weak,
            "absolutos_fortes": strong,
            "universais_fracos": weak,
            "numeros": numerics,
            "especificidade": round(specificity, 2),
            "fator_especificidade": round(factor, 2),
            "taxa_efetiva_mil": round(effective_rate, 2),
            "termos": {**strong_terms, **weak_terms},
        },
    )


_SIGNAL_FUNCS = (
    signal_repetition,
    signal_vague_terms,
    signal_contradictions,
    signal_excess_details,
    signal_cognitive_load,
    signal_unsupported_claims,
)


def _interpret(score: float) -> str:
    for name, limit in _LEVELS:
        if score < limit:
            return name
    return "alto"


def _explain(
    score: float, level: str, signals: list[SignalResult], weights: tuple[float, ...]
) -> str:
    lines = [
        "A análise de padrões linguísticos associados a engano (psicologia cognitiva e "
        "neurolinguística) combinou 6 sinais com pesos auditáveis: contradições internas "
        "têm maior peso (0.25), por serem o sinal mais confiável na literatura, e "
        "afirmações absolutas sem suporte (0.20) operam como proxy de afirmações "
        "desacreditadas.",
    ]
    claims = next((s for s in signals if s.name == "afirmacoes_sem_suporte"), None)
    if claims and claims.evidence.get("absolutos_fortes"):
        lines.append(
            "No sinal de afirmações sem suporte, absolutos empíricos (asseverações de "
            "certeza e negações absolutas factuais) pesam mais do que quantificadores "
            "universais (morais/retóricos); ambos são descontados quando o texto não "
            "apresenta dados específicos, pois aí tendem a ser universais éticos, não "
            "afirmações factuais."
        )
    lines.append(f"Score final: {score:.2f} (escala 0..1). Nível: {level}.")
    for signal, weight in zip(signals, weights, strict=True):
        lines.append(f"- {signal.name}: {signal.score:.2f} (peso {weight:.2f})")
    if any(s.evidence.get("artefato_transcricao") for s in signals):
        lines.append(
            "Nota: o texto parece ser transcrição de fala sem pontuação de frases; "
            "nesse caso o sinal de carga cognitiva é zerado (comprimento de frase e "
            "pausas seriam artefatos do formato, não medida real de esforço)."
        )
    lines.append(
        "Os scores de cada sinal são taxas de ocorrência por 1000 palavras, mapeadas "
        "por uma curva de saturação (x/(x+k)); não há modelo estatístico pesado, apenas "
        "varredura léxica única — custo computacional O(n)."
    )
    return "\n".join(lines)


def analyze_deception(text: str | None, *, weights: tuple[float, ...] | None = None) -> dict:
    """Analisa um texto e retorna score de engano, sinais e explicação."""
    if not text or not text.strip():
        raise ProcessingError(
            "Análise de engano recebeu texto vazio",
            user_message="Não é possível analisar um texto vazio.",
        )

    words = _word_count(text)
    signals = [func(text, words) for func in _SIGNAL_FUNCS]
    weights = weights or _SIGNAL_WEIGHTS

    alternatives = {"texto": {s.name: s.score for s in signals}}
    matrix = WeightedDecisionMatrix(alternatives, _SIGNAL_NAMES, list(weights))
    ranked = matrix.score()
    score = ranked[0].score
    level = _interpret(score)

    logger.info(
        "Análise de padrões de engano concluída",
        extra={
            "code": "deception_done",
            "status": "ok",
            "metric": f"score={score:.2f}, nível={level}",
        },
    )
    return {
        "score": round(score, 3),
        "level": level,
        "signals": [asdict(signal) for signal in signals],
        "weights": list(weights),
        "explanation": _explain(score, level, signals, tuple(weights)),
    }
