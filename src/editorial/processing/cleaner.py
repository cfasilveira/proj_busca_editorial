"""Limpeza e normalização de texto bruto.

Fail first: entrada inválida (None) é rejeitada. Fail gracefully: texto que
perde todo o conteúdo após limpeza é sinalizado por `had_content=False`,
permitindo ao chamador registrar sem travar o pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..errors import ProcessingError
from ..logging_setup import get_logger

logger = get_logger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CleanResult:
    text: str
    had_content: bool


class TextCleaner:
    def __init__(
        self,
        *,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
        min_chars: int = 1,
    ) -> None:
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        self.min_chars = min_chars

    def clean(self, text: str | None) -> CleanResult:
        if text is None:
            raise ProcessingError(
                "Limpeza recebeu texto None",
                user_message="Não é possível limpar um texto nulo.",
            )

        cleaned = text
        if self.normalize_unicode:
            cleaned = unicodedata.normalize("NFC", cleaned)
        if self.remove_urls:
            cleaned = _URL_RE.sub(" ", cleaned)
        if self.remove_emails:
            cleaned = _EMAIL_RE.sub(" ", cleaned)
        cleaned = _WS_RE.sub(" ", _CONTROL_RE.sub(" ", cleaned)).strip()

        had_content = bool(text.strip()) and len(cleaned) >= self.min_chars
        if text.strip() and not had_content:
            logger.warning(
                "Texto perdeu todo o conteúdo após limpeza",
                extra={"code": "cleaned_to_empty"},
            )
        return CleanResult(text=cleaned, had_content=had_content)
