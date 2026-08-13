"""Tipos-base do módulo de Ingestão.

`Document` é a unidade canônica de texto que atravessa todo o pipeline
(ingestão -> processamento -> análise -> relatório).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Document:
    """Um texto coletado com metadados de proveniência (auditoria)."""

    uid: str
    text: str
    source: str
    collected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "text": self.text,
            "source": self.source,
            "collected_at": self.collected_at,
            "status": self.status,
            "metadata": self.metadata,
        }


class Source(ABC):
    """Contrato de uma fonte de coleta (CSV, API, scraping...)."""

    name: str = "base"

    @abstractmethod
    def ingest(self) -> list[Document]:
        """Coleta e retorna documentos. Levanta IngestionError em falhas fatais."""
        raise NotImplementedError


def stable_uid(raw: Any, fallback: int) -> str:
    """Gera um uid estável e determinístico a partir de um valor bruto."""
    if raw is None or str(raw).strip() == "":
        return f"doc-{fallback}"
    return str(raw).strip()
