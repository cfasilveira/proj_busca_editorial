"""Controle de acesso simples por token (RBAC básico).

Tokens de leitura e escrita são lidos do ambiente. Qualquer tentativa de
acesso não autorizado é registrada em log estruturado com o token
mascarado, e levantada como SecurityError (fail first no chamador).
"""

from __future__ import annotations

import hmac
import logging

from ..config import Settings, get_settings
from ..errors import SecurityError, fail
from ..logging_setup import get_logger

logger = get_logger(__name__)


class AccessControl:
    """Valida tokens de acesso contra configuração."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @staticmethod
    def _mask(token: str) -> str:
        if len(token) <= 4:
            return "*" * len(token)
        return token[:2] + "..." + token[-2:]

    def _authorized(self, token: str | None, allowed: str | None) -> bool:
        if not token or not allowed:
            return False
        return hmac.compare_digest(token, allowed)

    def _reject(self, scope: str, token: str | None) -> None:
        fail(
            logger,
            SecurityError,
            f"Acesso não autorizado ao escopo '{scope}'",
            user_message="Credenciais inválidas. Verifique o token de acesso.",
            level=logging.WARNING,
            code="unauthorized_access",
            scope=scope,
            actor=self._mask(token) if token else "<ausente>",
            status="denied",
        )

    def require_read(self, token: str | None) -> None:
        """Exige token de leitura. Registra tentativa não autorizada."""
        if not self._authorized(token, self.settings.api_token):
            self._reject("read", token)

    def require_write(self, token: str | None) -> None:
        """Exige token de escrita (admin)."""
        if not self._authorized(token, self.settings.admin_token or self.settings.api_token):
            self._reject("write", token)
