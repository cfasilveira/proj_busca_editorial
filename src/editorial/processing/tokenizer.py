"""Tokenização com cadeia de fallback: spaCy -> NLTK -> tokenizador interno.

Prioridade de qualidade:
  1. spaCy (se modelo instalado) — melhor segmentação, inclusive pt-BR.
  2. NLTK (se dado `punkt` disponível).
  3. Tokenizador regex interno (zero dependência).

Qualquer indisponibilidade de nível é registrada em log uma única vez.
Textos vazios são detectados (fail first) e levantados como ProcessingError.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..errors import ProcessingError
from ..logging_setup import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\b[\w]+(?:[-']\w+)*\b", re.UNICODE)


def _regex_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


class Tokenizer:
    """Tokenizador configurável com fallback automático."""

    def __init__(self, *, spacy_model: str | None = None, nltk_data_dir: str | None = None) -> None:
        self._spacy_nlp = None
        self._nltk_available = False
        self._active_backend = "internal"

        self._try_load_spacy(spacy_model)
        if self._spacy_nlp is None:
            self._try_load_nltk(nltk_data_dir)

    def _try_load_spacy(self, spacy_model: str | None) -> None:
        try:
            import spacy  # type: ignore[import-not-found]

            candidates = []
            if spacy_model:
                candidates.append(spacy_model)
            candidates.extend(["pt_core_news_sm", "en_core_web_sm"])

            for candidate in candidates:
                try:
                    self._spacy_nlp = spacy.load(candidate, disable=["parser", "ner"])
                    self._active_backend = f"spacy:{candidate}"
                    logger.info(
                        "Tokenizer spaCy ativo",
                        extra={
                            "code": "tokenizer_backend",
                            "metric": self._active_backend,
                        },
                    )
                    return
                except OSError:
                    continue
            logger.warning(
                "Modelo spaCy indisponível; tentando NLTK",
                extra={"code": "tokenizer_fallback", "metric": "spacy->nltk"},
            )
        except ImportError:
            logger.warning(
                "spaCy não instalado; tentando NLTK",
                extra={"code": "tokenizer_fallback", "metric": "spacy->nltk"},
            )

    def _try_load_nltk(self, nltk_data_dir: str | None) -> None:
        try:
            import nltk  # type: ignore[import-not-found]

            if nltk_data_dir:
                nltk.data.path.append(nltk_data_dir)
            nltk.data.find("tokenizers/punkt") or nltk.data.find("tokenizers/punkt_tab")
            self._nltk_available = True
            self._active_backend = "nltk"
            logger.info(
                "Tokenizer NLTK ativo",
                extra={"code": "tokenizer_backend", "metric": "nltk"},
            )
        except (ImportError, LookupError):
            logger.warning(
                "NLTK/punkt indisponível; usando tokenizador interno",
                extra={"code": "tokenizer_fallback", "metric": "nltk->internal"},
            )

    @property
    def backend(self) -> str:
        return self._active_backend

    def tokenize(self, text: str) -> list[str]:
        if not text or not text.strip():
            raise ProcessingError(
                "Tokenização recebeu texto vazio",
                user_message="Não é possível tokenizar um texto vazio.",
            )
        if self._spacy_nlp is not None:
            return [tok.text for tok in self._spacy_nlp(text)]
        if self._nltk_available:
            import nltk  # type: ignore[import-not-found]

            return nltk.tokenize.word_tokenize(text, language="portuguese")
        return _regex_tokenize(text)

    def tokenize_many(self, texts: Iterable[str]) -> list[list[str]]:
        tokens = []
        for text in texts:
            try:
                tokens.append(self.tokenize(text))
            except ProcessingError:
                tokens.append([])
        return tokens
