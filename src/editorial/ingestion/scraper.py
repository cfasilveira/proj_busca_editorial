"""Ingestão por scraping de páginas HTML.

Uso responsável: respeita User-Agent e timeout. Extrai o texto principal
com BeautifulSoup. Fail first: URL inválida ou resposta não-2xx aborta.
Fail gracefully: páginas individuais com erro são registradas e puladas.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable

import httpx
from bs4 import BeautifulSoup

from ..config import Settings, get_settings
from ..errors import IngestionError
from ..logging_setup import get_logger
from .base import Document, Source, stable_uid

logger = get_logger(__name__)

_TAG_BLACKLIST = {
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "noscript",
    "iframe",
    "aside",
}
_WHITESPACE = re.compile(r"\s+")


def _strip_tags(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


class ScraperSource(Source):
    """Coleta o texto principal de uma lista de URLs."""

    name = "scraper"

    def __init__(
        self,
        urls: Iterable[str],
        *,
        settings: Settings | None = None,
        content_selectors: tuple[str, ...] = ("article", "main", "body"),
    ) -> None:
        self.urls = list(urls)
        self.settings = settings or get_settings()
        self.content_selectors = content_selectors

    def ingest(self) -> list[Document]:
        if not self.urls:
            raise IngestionError(
                "Nenhuma URL fornecida para scraping",
                user_message="Informe ao menos uma URL para o scraping.",
            )

        documents: list[Document] = []
        with httpx.Client(timeout=self.settings.http_timeout) as client:
            for index, url in enumerate(self.urls):
                html = self._fetch(client, url)
                if html is None:
                    continue

                text = self._extract_text(html)
                if not text:
                    logger.warning(
                        "Página sem conteúdo extraível",
                        extra={
                            "source": self.name,
                            "status": "invalid",
                            "code": "empty_page",
                            "doc_id": url,
                        },
                    )
                    continue

                documents.append(
                    Document(
                        uid=stable_uid(None, index),
                        text=text[: self.settings.max_text_chars],
                        source=f"scrape:{url}",
                        status="ok",
                        metadata={"url": url, "chars": len(text)},
                    )
                )
                time.sleep(0.2)

        if not documents:
            raise IngestionError(
                "Nenhuma página válida coletada no scraping",
                user_message="Nenhuma página pôde ser coletada no scraping.",
            )

        logger.info(
            "Scraping concluído",
            extra={
                "source": self.name,
                "status": "ok",
                "code": "scrape_done",
                "metric": f"{len(documents)} docs",
            },
        )
        return documents

    def _fetch(self, client: httpx.Client, url: str) -> str | None:
        """Baixa a página e retorna HTML, ou None com o motivo logado."""
        if not url.startswith(("http://", "https://")):
            logger.warning(
                "URL de scraping inválida ignorada",
                extra={"source": self.name, "status": "invalid", "code": "bad_url", "doc_id": url},
            )
            return None
        try:
            response = client.get(url, headers={"User-Agent": self.settings.user_agent})
        except httpx.HTTPError:
            logger.error(
                "Falha HTTP no scraping",
                extra={
                    "source": self.name,
                    "status": "error",
                    "code": "http_failure",
                    "doc_id": url,
                },
            )
            return None
        if response.status_code != 200:
            logger.warning(
                "Status não-2xx no scraping",
                extra={
                    "source": self.name,
                    "status": "error",
                    "code": "http_status",
                    "metric": str(response.status_code),
                    "doc_id": url,
                },
            )
            return None
        return response.text

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(_TAG_BLACKLIST):
            tag.decompose()

        for selector in self.content_selectors:
            node = soup.find(selector)
            if node is not None:
                return _strip_tags(node.get_text(" ", strip=True))

        return _strip_tags(soup.get_text(" ", strip=True))
