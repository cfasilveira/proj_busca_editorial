"""Teste de ponta a ponta do pipeline e da CLI."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import SAMPLE_CSV

from editorial.cli import main
from editorial.pipeline import run_pipeline


def test_run_pipeline_end_to_end(tmp_path: Path):
    settings = __import__("editorial.config", fromlist=["Settings"]).Settings(
        vector_dim=32,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        index_path=tmp_path / "data" / "index.faiss",
        model_path=tmp_path / "data" / "model.joblib",
        spacy_model="pt_core_news_sm",
    )
    result = run_pipeline(SAMPLE_CSV, tmp_path / "out", settings)

    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    analysis = report["sections"]["analysis"]
    assert report["sections"]["ingestion"]["count"] == 6
    assert analysis["linguistic_profile"]["length_stats"]["count"] == 6
    assert analysis["regression"]["r_squared"] >= 0.0
    assert analysis["bayesian"]["probability_first_gt_second"] > 0.0

    store = __import__("editorial.vector.store", fromlist=["FaissStore"]).FaissStore(32)
    store.load(settings.index_path)
    assert store.count() == 6


def test_cli_pipeline_command(tmp_path: Path, capsys):
    code = main(
        [
            "pipeline",
            str(SAMPLE_CSV),
            "--outdir",
            str(tmp_path / "cli_out"),
        ]
    )
    assert code == 0
    assert (tmp_path / "cli_out" / "report.json").exists()


def test_cli_search_finds_document(tmp_path: Path):
    code = main(
        [
            "pipeline",
            str(SAMPLE_CSV),
            "--outdir",
            str(tmp_path / "search_src"),
        ]
    )
    assert code == 0
    settings = __import__("editorial.config", fromlist=["Settings"]).Settings()
    code = main(
        [
            "search",
            "política econômica",
            "--index",
            str(settings.data_dir / "index.faiss"),
            "--model",
            str(settings.data_dir / "embedding_model.joblib"),
            "--topk",
            "3",
        ]
    )
    assert code == 0


def test_cli_missing_csv_fails_gracefully(capsys):
    code = main(["pipeline", "/caminho/inexistente.csv", "--outdir", "/tmp/x"])
    assert code == 1
    captured = capsys.readouterr()
    assert "não existe" in captured.err


def test_cli_pipeline_with_txt(tmp_path: Path, capsys):
    txt = tmp_path / "editorial.txt"
    txt.write_text("Texto único de teste para o pipeline.", encoding="utf-8")
    code = main(
        [
            "pipeline",
            str(txt),
            "--outdir",
            str(tmp_path / "txt_out"),
        ]
    )
    assert code == 0
    report = json.loads((tmp_path / "txt_out" / "report.json").read_text(encoding="utf-8"))
    assert report["sections"]["ingestion"]["count"] == 1
    assert report["sections"]["documents"][0]["uid"] == "editorial"
    assert report["sections"]["documents"][0]["source"] == "txt:editorial.txt"
