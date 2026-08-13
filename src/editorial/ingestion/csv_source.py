"""Ingestão a partir de arquivos CSV.

Fail first: arquivo inexistente, vazio, ou sem a coluna de texto exigida
aborta imediatamente com `IngestionError` e log estruturado.
Fail gracefully: linhas individuais inválidas são registradas e ignoradas;
apenas quando *nenhuma* linha válida restar a operação é abortada.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..errors import IngestionError
from ..logging_setup import get_logger
from .base import Document, Source, stable_uid

logger = get_logger(__name__)


class CsvSource(Source):
    name = "csv"

    def __init__(
        self,
        path: str | Path,
        *,
        text_column: str = "text",
        id_column: str | None = None,
        metadata_columns: tuple[str, ...] = (),
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        self.text_column = text_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns
        self.encoding = encoding

    def ingest(self) -> list[Document]:
        if not self.path.exists():
            raise IngestionError(
                f"Arquivo CSV não encontrado: {self.path}",
                user_message=f"O arquivo '{self.path}' não existe.",
            )
        if not self.path.is_file():
            raise IngestionError(
                f"Caminho não é um arquivo: {self.path}",
                user_message=f"'{self.path}' não é um arquivo.",
            )

        with self.path.open("r", encoding=self.encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise IngestionError(
                    f"CSV vazio ou sem cabeçalho: {self.path}",
                    user_message="O arquivo CSV está vazio ou sem cabeçalho.",
                )
            if self.text_column not in reader.fieldnames:
                raise IngestionError(
                    f"Coluna '{self.text_column}' ausente em {self.path}",
                    user_message=(
                        f"A coluna de texto '{self.text_column}' não existe no CSV. "
                        f"Colunas disponíveis: {', '.join(reader.fieldnames)}."
                    ),
                )
            rows = list(reader)

        if not rows:
            raise IngestionError(
                f"CSV sem linhas de dados: {self.path}",
                user_message="O arquivo CSV não contém linhas de dados.",
            )

        documents: list[Document] = []
        for index, row in enumerate(rows, start=2):
            raw_text = (row.get(self.text_column) or "").strip()
            if not raw_text:
                logger.warning(
                    "Linha ignorada: texto vazio",
                    extra={"source": self.name, "code": "empty_text_row", "doc_id": index},
                )
                continue

            documents.append(
                Document(
                    uid=stable_uid(row.get(self.id_column) if self.id_column else None, index),
                    text=raw_text,
                    source=f"csv:{self.path.name}",
                    status="ok",
                    metadata={
                        col: value
                        for col in self.metadata_columns
                        if (value := row.get(col)) not in (None, "")
                    },
                )
            )

        if not documents:
            raise IngestionError(
                f"Nenhuma linha válida em {self.path}",
                user_message="Nenhuma linha válida foi encontrada no CSV (todas sem texto).",
            )

        logger.info(
            "CSV ingerido",
            extra={
                "source": self.name,
                "status": "ok",
                "code": "csv_ingested",
                "doc_id": str(self.path.name),
                "metric": f"{len(documents)} documentos",
            },
        )
        return documents
