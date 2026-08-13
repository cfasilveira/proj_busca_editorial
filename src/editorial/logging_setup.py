"""Logging estruturado em JSON + console.

Dois formatadores:
  - `JsonFormatter`: emite uma única linha JSON por registro (auditável,
    consumível por ferramentas de observabilidade).
  - `ConsoleFormatter`: legível para humanos, uma linha por registro.

A função `get_logger` retorna um logger com contexto padrão do módulo.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_LOGGERS: dict[str, logging.Logger] = {}

_EXTRA_KEYS = (
    "source",
    "doc_id",
    "status",
    "code",
    "metric",
    "duration_ms",
    "actor",
    "scope",
)


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {key: value for key in _EXTRA_KEYS if (value := getattr(record, key, None)) is not None}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        payload.update(_extras(record))
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        prefix = f"{record.levelname:<7} {record.name} | {record.getMessage()}"
        extras = _extras(record)
        if extras:
            prefix += " [" + ", ".join(f"{k}={v}" for k, v in extras.items()) + "]"
        if record.exc_info:
            prefix += "\n" + self.formatException(record.exc_info)
        return prefix


def setup_logging(level: str = "INFO", *, json_enabled: bool = False) -> None:
    """Configura o logger raiz. Chamar uma única vez na entrada do processo."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_enabled else ConsoleFormatter())
    root.addHandler(handler)

    # Silencia libs ruidosas de terceiros.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Retorna (e cacheia) um logger para o módulo `name`."""
    logger = _LOGGERS.get(name)
    if logger is None:
        logger = logging.getLogger(name)
        _LOGGERS[name] = logger
    return logger
