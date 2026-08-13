"""Ingestão a partir de APIs REST/JSON.

Utiliza httpx com timeout explícito. Suporta autenticação por token
(controlada pelo módulo de Segurança). Fail first: URL inválida, resposta
não-2xx, payload sem a chave de texto ou lista vazia abortam a operação.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import httpx

from ..config import Settings, get_settings
from ..errors import IngestionError, fail
from ..logging_setup import get_logger
from ..security.access import AccessControl
from .base import Document, Source, stable_uid

logger = get_logger(__name__)


class ApiSource(Source):
    """Coleta documentos de um endpoint JSON.

    O payload deve ser uma lista de objetos com a chave `text_field`, ou um
    dicionário cuja chave `results_field` contém essa lista.
    """

    name = "api"

    def __init__(
        self,
        url: str,
        *,
        text_field: str = "text",
        results_field: str | None = None,
        id_field: str | None = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        token: str | None = None,
        settings: Settings | None = None,
        access: AccessControl | None = None,
    ) -> None:
        self.url = url
        self.text_field = text_field
        self.results_field = results_field
        self.id_field = id_field
        self.headers = dict(headers or {})
        self.params = dict(params or {})
        self.settings = settings or get_settings()
        self.access = access or AccessControl(self.settings)

    def _build_headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.settings.user_agent, **self.headers}
        if self.settings.api_token:
            headers["Authorization"] = f"Bearer {self.settings.api_token}"
        return headers

    def ingest(self) -> list[Document]:
        if not self.url.startswith(("http://", "https://")):
            raise IngestionError(
                f"URL de API inválida: {self.url}",
                user_message="A URL da API precisa começar com http:// ou https://.",
            )

        # Segurança: token informado deve ser autorizado.
        if self.settings.api_token:
            self.access.require_read(self.settings.api_token)

        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.settings.http_timeout) as client:
                response = client.get(
                    self.url,
                    headers=self._build_headers(),
                    params=self.params,
                )
        except httpx.HTTPError as exc:
            fail(
                logger,
                IngestionError,
                f"Falha HTTP ao acessar {self.url}: {exc}",
                user_message=f"Não foi possível acessar a API ({exc}).",
                source=self.name,
                status="error",
                code="http_failure",
                doc_id=self.url,
            )

        if response.status_code != 200:
            logger.warning(
                "Resposta não-2xx da API",
                extra={
                    "source": self.name,
                    "status": "error",
                    "code": "http_status",
                    "metric": str(response.status_code),
                    "doc_id": self.url,
                },
            )
            raise IngestionError(
                f"API retornou status {response.status_code}",
                user_message=f"A API retornou o status HTTP {response.status_code}.",
            )

        payload: Any = response.json()
        items = payload
        if self.results_field:
            if not isinstance(payload, dict) or self.results_field not in payload:
                raise IngestionError(
                    f"Campo de resultados '{self.results_field}' ausente na resposta",
                    user_message="A resposta da API não contém o campo de resultados esperado.",
                )
            items = payload[self.results_field]

        if not isinstance(items, list) or not items:
            raise IngestionError(
                "API retornou lista vazia",
                user_message="A API retornou uma lista vazia de documentos.",
            )

        documents: list[Document] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            raw_text = str(item.get(self.text_field) or "").strip()
            if not raw_text:
                logger.warning(
                    "Item de API inválido ignorado",
                    extra={"source": self.name, "status": "invalid", "code": "invalid_item"},
                )
                continue
            documents.append(
                Document(
                    uid=stable_uid(item.get(self.id_field) if self.id_field else None, index),
                    text=raw_text,
                    source=f"api:{self.url}",
                    status="ok",
                    metadata={
                        k: v for k, v in item.items() if k not in (self.text_field, self.id_field)
                    },
                )
            )

        if not documents:
            raise IngestionError(
                "Nenhum documento válido extraído da API",
                user_message="Nenhum documento válido foi extraído da API.",
            )

        logger.info(
            "API ingerida",
            extra={
                "source": self.name,
                "status": "ok",
                "code": "api_ingested",
                "doc_id": self.url,
                "metric": f"{len(documents)} docs",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return documents
