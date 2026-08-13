"""Testes do módulo de Ingestão."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import SAMPLE_CSV

from editorial.errors import IngestionError
from editorial.ingestion import CsvSource


def test_csv_ingests_documents():
    docs = CsvSource(SAMPLE_CSV, text_column="text", id_column="id").ingest()
    assert len(docs) == 6
    assert all(doc.status == "ok" for doc in docs)
    assert all(doc.text.strip() for doc in docs)
    assert {doc.uid for doc in docs} == {"1", "2", "3", "4", "5", "6"}


def test_csv_metadata_columns():
    docs = CsvSource(
        SAMPLE_CSV,
        text_column="text",
        id_column="id",
        metadata_columns=("publicacao", "secao"),
    ).ingest()
    assert docs[0].metadata["publicacao"] == "Jornal A"
    assert docs[0].metadata["secao"] == "economia"


def test_csv_missing_file_fails_first(tmp_path: Path):
    with pytest.raises(IngestionError, match="não encontrado"):
        CsvSource(tmp_path / "nao_existe.csv", text_column="text").ingest()


def test_csv_missing_column_fails_first(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(IngestionError, match="ausente"):
        CsvSource(bad, text_column="text").ingest()


def test_csv_empty_fails_first(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(IngestionError):
        CsvSource(empty, text_column="text").ingest()


def test_csv_skips_empty_rows_gracefully(tmp_path: Path):
    partial = tmp_path / "partial.csv"
    partial.write_text("id,text\n1,\n2,texto valido\n", encoding="utf-8")
    docs = CsvSource(partial, text_column="text", id_column="id").ingest()
    assert len(docs) == 1
    assert docs[0].uid == "2"


def test_csv_all_empty_rows_aborts(tmp_path: Path):
    all_empty = tmp_path / "all_empty.csv"
    all_empty.write_text("id,text\n1,\n2,\n", encoding="utf-8")
    with pytest.raises(IngestionError, match="Nenhuma linha válida"):
        CsvSource(all_empty, text_column="text").ingest()
