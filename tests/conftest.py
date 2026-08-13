"""Configuração compartilhada dos testes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
SAMPLE_CSV = SAMPLES / "editorials.csv"


@pytest.fixture(scope="session")
def settings_factory():
    from editorial.config import Settings

    def make(**kw):
        return Settings(**kw)

    return make
