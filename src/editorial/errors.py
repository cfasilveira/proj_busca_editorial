"""Exceções tipadas do domínio com mensagens amigáveis para o usuário final.

Filosofia de tratamento de erros:
  - Early return / Fail first: valide entradas no início e levante a exceção
    específica o quanto antes, abortando a operação.
  - Fail gracefully: cada exceção carrega `user_message` (mensagem clara ao
    usuário) e deve ser registrada em log estruturado pelo chamador.
"""

from __future__ import annotations

import logging
from typing import Any, NoReturn


class EditorialError(Exception):
    """Erro base de todo o domínio."""

    kind = "editorial_error"

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "message": self.message,
            "user_message": self.user_message,
        }


class IngestionError(EditorialError):
    kind = "ingestion_error"


class ProcessingError(EditorialError):
    kind = "processing_error"


class ScientificError(EditorialError):
    kind = "scientific_error"


class VectorError(EditorialError):
    kind = "vector_error"


class ReportError(EditorialError):
    kind = "report_error"


class SecurityError(EditorialError):
    kind = "security_error"


def fail(
    logger: logging.Logger,
    error_cls: type[EditorialError],
    message: str,
    *,
    user_message: str | None = None,
    level: int = logging.ERROR,
    **extra: Any,
) -> NoReturn:
    """Registra o erro em log estruturado e levanta exceção de domínio.

    Encapsula a filosofia "fail gracefully": mensagem clara ao usuário
    (`user_message`) + log auditável, sem travar o sistema.
    """
    details = {k: v for k, v in extra.items() if v is not None}
    logger.log(level, message, extra=details)
    raise error_cls(message, user_message=user_message)
