"""Configuração centralizada do aplicativo.

Valores lidos de variáveis de ambiente (com suporte a .env). A configuração
é imutável após a criação e segue a filosofia "fail first": valores inválidos
de chave obrigatória são rejeitados no boot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DOTENV_LOADED: bool = load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Configuração do aplicativo, imutável após construção."""

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_json: bool = field(default_factory=lambda: _env_bool("LOG_JSON", False))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    reports_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", "reports")))
    index_path: Path = field(
        default_factory=lambda: Path(os.getenv("INDEX_PATH", "data/index.faiss"))
    )
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", "data/embedding_model.joblib"))
    )

    vector_dim: int = field(default_factory=lambda: _env_int("VECTOR_DIM", 256))

    http_timeout: float = field(default_factory=lambda: _env_float("HTTP_TIMEOUT", 15.0))
    user_agent: str = field(
        default_factory=lambda: os.getenv("USER_AGENT", "editorial-cli/0.1 (+contact)")
    )

    api_token: str | None = field(default_factory=lambda: os.getenv("EDITORIAL_API_TOKEN"))
    admin_token: str | None = field(default_factory=lambda: os.getenv("EDITORIAL_ADMIN_TOKEN"))

    spacy_model: str = field(default_factory=lambda: os.getenv("SPACY_MODEL", "pt_core_news_sm"))
    nltk_data_dir: Path | None = field(
        default_factory=lambda: (
            Path(os.getenv("NLTK_DATA_DIR")) if os.getenv("NLTK_DATA_DIR") else None
        )
    )

    max_text_chars: int = field(default_factory=lambda: _env_int("MAX_TEXT_CHARS", 100_000))

    def ensure_dirs(self) -> None:
        """Cria os diretórios de saída. Fail first: falha se não puder criar."""
        for directory in (self.data_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
