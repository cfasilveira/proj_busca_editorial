"""Ingestão a partir de arquivos de texto (.txt).

Aceita um arquivo único ou um diretório com vários `.txt`; cada arquivo
vira um documento (uid = nome do arquivo sem extensão). Fail first: caminho
inexistente ou sem nenhum `.txt` válido aborta com `IngestionError`.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import IngestionError
from ..logging_setup import get_logger
from .base import Document, Source

logger = get_logger(__name__)


class TxtSource(Source):
    name = "txt"

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        min_chars: int = 1,
    ) -> None:
        self.path = Path(path)
        self.encoding = encoding
        self.min_chars = min_chars

    def _files(self) -> list[Path]:
        if not self.path.exists():
            raise IngestionError(
                f"Caminho TXT não encontrado: {self.path}",
                user_message=f"O caminho '{self.path}' não existe.",
            )
        if self.path.is_file():
            files = [self.path]
        elif self.path.is_dir():
            files = sorted(p for p in self.path.rglob("*.txt"))
        else:
            raise IngestionError(
                f"Caminho TXT inválido: {self.path}",
                user_message=f"'{self.path}' não é um arquivo nem um diretório.",
            )

        if not files:
            raise IngestionError(
                f"Nenhum arquivo .txt em {self.path}",
                user_message=f"Não foram encontrados arquivos .txt em '{self.path}'.",
            )
        return files

    def ingest(self) -> list[Document]:
        documents: list[Document] = []
        for file in self._files():
            try:
                text = file.read_text(encoding=self.encoding).strip()
            except (OSError, UnicodeDecodeError):
                logger.warning(
                    "Arquivo ignorado: falha de leitura",
                    extra={"source": self.name, "code": "txt_read_failed", "doc_id": file.name},
                )
                continue

            if len(text) < self.min_chars:
                logger.warning(
                    "Arquivo ignorado: conteúdo abaixo do mínimo",
                    extra={"source": self.name, "code": "txt_empty", "doc_id": file.name},
                )
                continue

            documents.append(
                Document(
                    uid=file.stem,
                    text=text,
                    source=f"txt:{file.name}",
                    status="ok",
                )
            )

        if not documents:
            raise IngestionError(
                f"Nenhum arquivo .txt válido em {self.path}",
                user_message="Nenhum arquivo .txt com conteúdo válido foi encontrado.",
            )

        logger.info(
            "TXT ingerido",
            extra={
                "source": self.name,
                "status": "ok",
                "code": "txt_ingested",
                "doc_id": str(self.path.name),
                "metric": f"{len(documents)} documentos",
            },
        )
        return documents
