"""Configuração centralizada do aplicativo.

Valores lidos de variáveis de ambiente (com suporte a .env). A configuração
é imutável após a criação e segue a filosofia "fail first": valores inválidos
de chave obrigatória são rejeitados no boot.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: Any, cast: Callable[[str], Any]) -> Any:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default if default is None or not isinstance(default, str) else cast(default)
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return default


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Configuração do aplicativo, imutável após construção."""

    log_level: str = _env("LOG_LEVEL", "INFO", str)
    log_json: bool = _env("LOG_JSON", False, _as_bool)

    data_dir: Path = _env("DATA_DIR", "data", Path)
    reports_dir: Path = _env("REPORTS_DIR", "reports", Path)
    index_path: Path = _env("INDEX_PATH", "data/index.faiss", Path)
    model_path: Path = _env("MODEL_PATH", "data/embedding_model.joblib", Path)

    vector_dim: int = _env("VECTOR_DIM", 256, int)

    http_timeout: float = _env("HTTP_TIMEOUT", 15.0, float)
    user_agent: str = _env("USER_AGENT", "editorial-cli/0.1 (+contact)", str)

    api_token: str | None = _env("EDITORIAL_API_TOKEN", None, str)
    admin_token: str | None = _env("EDITORIAL_ADMIN_TOKEN", None, str)

    spacy_model: str = _env("SPACY_MODEL", "pt_core_news_sm", str)
    nltk_data_dir: Path | None = _env("NLTK_DATA_DIR", None, Path)

    max_text_chars: int = _env("MAX_TEXT_CHARS", 100_000, int)

    def ensure_dirs(self) -> None:
        """Cria os diretórios de saída. Fail first: falha se não puder criar."""
        for directory in (self.data_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
