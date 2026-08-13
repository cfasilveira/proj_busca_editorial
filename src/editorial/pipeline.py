"""Orquestração do pipeline completo: ingestão -> processamento -> ciência
-> vetorização -> relatório.

Cada estágio é uma função pura reutilizável; `run_pipeline` encadeia os
estágios e persiste artefatos intermediários (JSON) para auditoria.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import EditorialError
from .ingestion import CsvSource, Document
from .logging_setup import get_logger
from .processing import TextCleaner, Tokenizer, profile
from .reports import ReportBuilder
from .scientific import (
    linear_regression,
    probability_of_superiority,
)
from .vector import EmbeddingModel, FaissStore

logger = get_logger(__name__)


@dataclass
class ProcessedCorpus:
    documents: list[Document]
    cleaned_texts: list[str]
    token_lists: list[list[str]]
    lengths: list[int]


def ingest_csv(
    csv_path: str | Path, settings: Settings, text_column: str = "text"
) -> list[Document]:
    return CsvSource(csv_path, text_column=text_column).ingest()


def process_documents(documents: Sequence[Document], settings: Settings) -> ProcessedCorpus:
    cleaner = TextCleaner()
    tokenizer = Tokenizer(spacy_model=settings.spacy_model)

    kept: list[Document] = []
    cleaned: list[str] = []
    tokens: list[list[str]] = []
    lengths: list[int] = []
    dropped = 0

    for document in documents:
        result = cleaner.clean(document.text)
        if not result.had_content:
            dropped += 1
            logger.warning(
                "Documento descartado na limpeza",
                extra={
                    "code": "doc_dropped_clean",
                    "doc_id": document.uid,
                    "status": "invalid",
                },
            )
            continue
        kept.append(document)
        cleaned.append(result.text)
        doc_tokens = tokenizer.tokenize(result.text)
        tokens.append(doc_tokens)
        lengths.append(len(doc_tokens))

    if not cleaned:
        raise EditorialError(
            "Nenhum documento sobreviveu ao processamento",
            user_message="Todos os documentos foram descartados durante o processamento.",
        )

    logger.info(
        "Corpus processado",
        extra={
            "code": "corpus_processed",
            "status": "ok",
            "metric": f"{len(cleaned)} documentos, {dropped} descartados",
        },
    )
    return ProcessedCorpus(
        documents=kept,
        cleaned_texts=cleaned,
        token_lists=tokens,
        lengths=lengths,
    )


def analyze_corpus(corpus: ProcessedCorpus) -> dict[str, Any]:
    """Perfil linguístico + análise científica demonstrativa."""
    linguistic = profile(corpus.lengths, corpus.token_lists)

    top_token = linguistic["top_tokens"][0]["token"] if linguistic["top_tokens"] else ""

    regression: dict[str, Any] = {}
    if len(corpus.lengths) >= 3 and len(set(corpus.lengths)) > 1:
        reg = linear_regression(list(range(len(corpus.lengths))), corpus.lengths)
        regression = {
            "slope": reg.slope,
            "intercept": reg.intercept,
            "r_squared": reg.r_squared,
            "n": reg.n,
        }

    bayesian: dict[str, Any] = {}
    half = len(corpus.token_lists) // 2
    if half >= 1 and len(corpus.token_lists) >= 2:
        first = sum(len(t) for t in corpus.token_lists[:half])
        second = sum(len(t) for t in corpus.token_lists[half:])
        first_hit = sum(top_token in t for t in corpus.token_lists[:half])
        second_hit = sum(top_token in t for t in corpus.token_lists[half:])
        if first > 0 and second > 0:
            est = probability_of_superiority(first_hit, first, second_hit, second, seed=42)
            bayesian = {
                "comparison": f"'{top_token}' primeira-metade vs segunda-metade",
                "probability_first_gt_second": est.probability_p1_gt_p2,
                "posterior_mean_p1": est.posterior_mean_p1,
                "posterior_mean_p2": est.posterior_mean_p2,
            }

    return {
        "linguistic_profile": linguistic,
        "regression": regression,
        "bayesian": bayesian,
    }


def index_corpus(corpus: ProcessedCorpus, settings: Settings) -> dict[str, Any]:
    model = EmbeddingModel(dim=settings.vector_dim).fit(corpus.cleaned_texts)
    vectors = model.transform_many(corpus.cleaned_texts)

    store = FaissStore(settings.vector_dim)
    store.add(vectors, [d.uid for d in corpus.documents])

    model.save(settings.model_path)
    store.save(settings.index_path)

    return {
        "dimension": settings.vector_dim,
        "indexed_documents": store.count(),
        "index_path": str(settings.index_path),
        "model_path": str(settings.model_path),
    }


def build_report(
    documents: Sequence[Document],
    analysis: dict[str, Any],
    vector_info: dict[str, Any],
) -> ReportBuilder:
    builder = ReportBuilder(f"Análise editorial {datetime.now(UTC).date()}")
    builder.add_section(
        "ingestion",
        {
            "count": len(documents),
            "sources": sorted({d.source for d in documents}),
        },
    )
    builder.add_section("analysis", analysis)
    builder.add_section("vector_index", vector_info)
    builder.add_section(
        "documents",
        [
            {
                "uid": d.uid,
                "source": d.source,
                "collected_at": d.collected_at,
                "status": d.status,
            }
            for d in documents
        ],
    )
    return builder


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def run_pipeline(
    csv_path: str | Path,
    out_dir: str | Path,
    settings: Settings,
    text_column: str = "text",
) -> dict[str, Any]:
    settings.ensure_dirs()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    documents = ingest_csv(csv_path, settings, text_column=text_column)
    corpus = process_documents(documents, settings)
    analysis = analyze_corpus(corpus)
    vector_info = index_corpus(corpus, settings)

    builder = build_report(corpus.documents, analysis, vector_info)
    report = builder.build()
    report_path = target / "report.json"
    builder.to_json(report_path)

    _write_json(
        target / "documents.json",
        [d.to_dict() for d in corpus.documents],
    )
    _write_json(
        target / "tokens.json",
        [
            {"uid": d.uid, "tokens": toks}
            for d, toks in zip(corpus.documents, corpus.token_lists, strict=True)
        ],
    )

    logger.info(
        "Pipeline concluído",
        extra={"code": "pipeline_done", "status": "ok", "doc_id": str(report_path)},
    )
    return {
        "report": str(report_path),
        "documents": str(target / "documents.json"),
        "tokens": str(target / "tokens.json"),
        "index": vector_info["index_path"],
        "model": vector_info["model_path"],
        "summary": report["sections"]["analysis"],
    }
