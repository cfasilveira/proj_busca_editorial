"""Testes do módulo de Relatórios."""

from __future__ import annotations

from pathlib import Path

import pytest

from editorial.errors import ReportError
from editorial.reports import ReportBuilder, export_pdf


def test_json_export(tmp_path: Path):
    builder = ReportBuilder("Teste")
    builder.add_section("results", [{"nome": "A", "score": 0.9}])
    target = builder.to_json(tmp_path / "out.json")
    assert target.exists()
    import json

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["meta"]["title"] == "Teste"
    assert payload["sections"]["results"][0]["score"] == 0.9


def test_csv_export(tmp_path: Path):
    builder = ReportBuilder("Teste")
    builder.add_section("results", [{"nome": "A", "score": 0.9}, {"nome": "B", "score": 0.7}])
    target = builder.to_csv(tmp_path / "out.csv", section="results")
    assert target.exists()
    assert "score" in target.read_text(encoding="utf-8")


def test_build_without_sections_fails():
    with pytest.raises(ReportError, match="seções"):
        ReportBuilder("vazio").build()


def test_csv_section_missing_fails(tmp_path: Path):
    builder = ReportBuilder("Teste")
    builder.add_section("outro", [1, 2])
    with pytest.raises(ReportError):
        builder.to_csv(tmp_path / "out.csv", section="results")


def test_pdf_stub_graceful():
    result = export_pdf({"sections": {}}, Path("/tmp/relatorio.pdf"))
    assert result["ok"] is False
    assert "não implementada" in result["reason"]
