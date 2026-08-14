"""Módulo de Ingestão: coleta de textos de múltiplas fontes."""

from .api_source import ApiSource
from .base import Document, Source, stable_uid
from .csv_source import CsvSource
from .scraper import ScraperSource
from .txt_source import TxtSource

__all__ = [
    "ApiSource",
    "CsvSource",
    "Document",
    "ScraperSource",
    "Source",
    "TxtSource",
    "stable_uid",
]
