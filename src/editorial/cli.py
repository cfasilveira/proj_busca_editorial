"""Interface de linha de comando (CLI) do aplicativo editorial.

Fluxo típico:
    editorial pipeline data/samples/editorials.csv --outdir reports
    editorial pipeline editorial_miriam.txt --outdir reports
    editorial axis miriam_leitao/ paulo_guedes/ --label-left miriam --label-right guedes
    editorial classify novo_editorial.txt --out resultados.json
    editorial veracity entrevista.txt --out analise_engano.json
    editorial validade novo_editorial.txt --out relatorio_validade.json
    editorial search "texto de busca" --index data/index.faiss --model data/embedding_model.joblib
    editorial audit

Erros são tratados com "fail gracefully": mensagens claras e exit code
diferente de zero, sem stack trace para o usuário final (a não ser com -v).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .errors import EditorialError
from .logging_setup import setup_logging
from .pipeline import ProcessedCorpus, ingest_path, process_documents, run_pipeline
from .scientific import (
    AuthorProfile,
    IdeologyRuler,
    analyze_deception,
    assess_validity,
    bias_explanation,
    classify_position,
    marker_frequencies,
)
from .security import AccessControl, audit_dependencies
from .vector import EmbeddingModel, FaissStore


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _cmd_pipeline(args: argparse.Namespace, settings: Settings) -> int:
    result = run_pipeline(args.source, args.outdir, settings, text_column=args.text_column)
    _print_json({"ok": True, **result["summary"]})
    return 0


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    model = EmbeddingModel.load(args.model)
    store = FaissStore(settings.vector_dim)
    store.load(args.index)
    results = store.search(model.transform(args.query), top_k=args.topk)
    if not results:
        print("Nenhum resultado encontrado para a consulta.")
        return 1
    for uid, score in results:
        print(f"{score:.4f}\t{uid}")
    return 0


def _cmd_axis(args: argparse.Namespace, settings: Settings) -> int:
    left = process_documents(ingest_path(args.left, settings, args.text_column), settings)
    right = process_documents(ingest_path(args.right, settings, args.text_column), settings)

    queries: list[tuple[str, ProcessedCorpus]] = []
    for source in args.query or []:
        queries.append(
            (source, process_documents(ingest_path(source, settings, args.text_column), settings))
        )

    texts = left.cleaned_texts + right.cleaned_texts
    for _, corpus in queries:
        texts += corpus.cleaned_texts

    model = EmbeddingModel(dim=settings.vector_dim).fit(texts)
    left_vecs = model.transform_many(left.cleaned_texts)
    right_vecs = model.transform_many(right.cleaned_texts)

    ruler = IdeologyRuler(
        AuthorProfile(args.label_left, left_vecs),
        AuthorProfile(args.label_right, right_vecs),
    )

    def positions(vecs, docs):
        return [{"uid": d.uid, **ruler.position(v)} for v, d in zip(vecs, docs, strict=True)]

    training = {
        args.label_left: positions(left_vecs, left.documents),
        args.label_right: positions(right_vecs, right.documents),
    }
    query_positions = [
        {"source": name, **pos}
        for name, corpus in queries
        for pos in positions(model.transform_many(corpus.cleaned_texts), corpus.documents)
    ]

    _print_json(
        {
            "ruler": ruler.to_dict(),
            "training": training,
            "queries": query_positions,
            "markers": {
                "left": marker_frequencies(" ".join(left.cleaned_texts)),
                "right": marker_frequencies(" ".join(right.cleaned_texts)),
            },
        }
    )
    return 0


def _cmd_classify(args: argparse.Namespace, settings: Settings) -> int:
    left = process_documents(ingest_path(args.left, settings, args.text_column), settings)
    right = process_documents(ingest_path(args.right, settings, args.text_column), settings)
    target = process_documents(ingest_path(args.source, settings, args.text_column), settings)

    texts = left.cleaned_texts + right.cleaned_texts + target.cleaned_texts
    model = EmbeddingModel(dim=settings.vector_dim).fit(texts)
    ruler = IdeologyRuler(
        AuthorProfile(args.label_left, model.transform_many(left.cleaned_texts)),
        AuthorProfile(args.label_right, model.transform_many(right.cleaned_texts)),
    )

    documents = []
    vectors = model.transform_many(target.cleaned_texts)
    for vector, doc in zip(vectors, target.documents, strict=True):
        position = ruler.position(vector)
        classification = classify_position(position["score"])
        markers = marker_frequencies(doc.text)
        documents.append(
            {
                "document": {"uid": doc.uid, "source": doc.source},
                "position": position,
                "classification": classification,
                "evidence": markers,
                "explanation": bias_explanation(position, classification, ruler, markers),
            }
        )

    result = {
        "method": "ruler_embeddings_tfidf_svd",
        "axis": {
            "left": {"name": ruler.left.name, "documents": ruler.left.documents},
            "right": {"name": ruler.right.name, "documents": ruler.right.documents},
        },
        "documents": documents,
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Resultado salvo em {out}")
    else:
        _print_json(result)
    return 0


def _cmd_veracity(args: argparse.Namespace, settings: Settings) -> int:
    corpus = process_documents(ingest_path(args.source, settings, args.text_column), settings)
    documents = [
        {"document": {"uid": doc.uid, "source": doc.source}, **analyze_deception(doc.text)}
        for doc in corpus.documents
    ]
    result = {"method": "regras_lexicais_cognitivas", "documents": documents}

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Resultado salvo em {out}")
    else:
        _print_json(result)
    return 0


def _cmd_validade(args: argparse.Namespace, settings: Settings) -> int:
    left = process_documents(ingest_path(args.left, settings, args.text_column), settings)
    right = process_documents(ingest_path(args.right, settings, args.text_column), settings)
    target = process_documents(ingest_path(args.source, settings, args.text_column), settings)

    texts = left.cleaned_texts + right.cleaned_texts + target.cleaned_texts
    model = EmbeddingModel(dim=settings.vector_dim).fit(texts)
    ruler = IdeologyRuler(
        AuthorProfile(args.label_left, model.transform_many(left.cleaned_texts)),
        AuthorProfile(args.label_right, model.transform_many(right.cleaned_texts)),
    )

    documents = []
    vectors = model.transform_many(target.cleaned_texts)
    for vector, doc in zip(vectors, target.documents, strict=True):
        position = ruler.position(vector)
        classification = classify_position(position["score"])
        deception = analyze_deception(doc.text)
        documents.append(
            {
                "document": {"uid": doc.uid, "source": doc.source},
                "position": position,
                "classification": classification,
                "deception": deception,
                "validity": assess_validity(
                    position=position, classification=classification, deception=deception
                ),
            }
        )

    result = {
        "method": "validade_régua_ideológica+dissimulação",
        "axis": {
            "left": {"name": ruler.left.name, "documents": ruler.left.documents},
            "right": {"name": ruler.right.name, "documents": ruler.right.documents},
        },
        "documents": documents,
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Resultado salvo em {out}")
    else:
        _print_json(result)
    return 0


def _cmd_audit(args: argparse.Namespace, settings: Settings) -> int:
    summary = audit_dependencies()
    _print_json(summary)
    return 0 if summary.get("ok") else 1


def _cmd_auth(args: argparse.Namespace, settings: Settings) -> int:
    access = AccessControl(settings)
    check = access.require_read if args.scope == "read" else access.require_write
    try:
        check(args.token)
    except EditorialError:
        print("Acesso negado.")
        return 1
    print("Acesso autorizado.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="editorial",
        description=(
            "Coleta, processa e analisa textos editoriais com perfis linguísticos e ideológicos."
        ),
    )
    parser.add_argument("--log-json", action="store_true", help="Logs em formato JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Exibir stack trace em erros.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pipe = subparsers.add_parser(
        "pipeline",
        help="Executa o pipeline completo a partir de um CSV, arquivo .txt ou diretório de .txt.",
    )
    p_pipe.add_argument(
        "source",
        help="CSV, arquivo .txt ou diretório contendo arquivos .txt.",
    )
    p_pipe.add_argument("--text-column", default="text", help="Coluna do CSV que contém o texto.")
    p_pipe.add_argument("--outdir", default="reports", help="Diretório de saída.")
    p_pipe.set_defaults(func=_cmd_pipeline)

    p_search = subparsers.add_parser("search", help="Consulta vetorial em corpus indexado.")
    p_search.add_argument("query", help="Texto da consulta.")
    p_search.add_argument("--index", default="data/index.faiss", help="Caminho do índice FAISS.")
    p_search.add_argument(
        "--model",
        default="data/embedding_model.joblib",
        help="Caminho do modelo de embeddings.",
    )
    p_search.add_argument("--topk", type=int, default=5, help="Número de resultados.")
    p_search.set_defaults(func=_cmd_search)

    p_audit = subparsers.add_parser("audit", help="Auditoria de dependências.")
    p_audit.set_defaults(func=_cmd_audit)

    p_axis = subparsers.add_parser(
        "axis",
        help="Régua ideológica entre dois corpora de referência (esquerda<->direita).",
    )
    p_axis.add_argument("left", help="Corpus/esquerda (diretório de .txt ou CSV).")
    p_axis.add_argument("right", help="Corpus/direita (diretório de .txt ou CSV).")
    p_axis.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="SOURCE",
        help="Texto/diretório extra para posicionar na régua (repetível).",
    )
    p_axis.add_argument("--label-left", default="esquerda", help="Nome do perfil à esquerda.")
    p_axis.add_argument("--label-right", default="direita", help="Nome do perfil à direita.")
    p_axis.add_argument("--text-column", default="text", help="Coluna do CSV que contém o texto.")
    p_axis.set_defaults(func=_cmd_axis)

    p_cls = subparsers.add_parser(
        "classify",
        help="Classifica editoriais/textos em ponderado, moderado, tendencioso ou extremista.",
    )
    p_cls.add_argument("source", help="Arquivo .txt, CSV ou diretório com os textos a classificar.")
    p_cls.add_argument("--left", default="miriam_leitao/", help="Corpus de referência à esquerda.")
    p_cls.add_argument("--right", default="paulo_guedes/", help="Corpus de referência à direita.")
    p_cls.add_argument("--label-left", default="miriam", help="Nome do perfil à esquerda.")
    p_cls.add_argument("--label-right", default="guedes", help="Nome do perfil à direita.")
    p_cls.add_argument("--out", default=None, help="Caminho do arquivo JSON de saída.")
    p_cls.add_argument("--text-column", default="text", help="Coluna do CSV que contém o texto.")
    p_cls.set_defaults(func=_cmd_classify)

    p_ver = subparsers.add_parser(
        "veracity",
        help="Analisa padrões linguísticos associados a engano/mentira em textos.",
    )
    p_ver.add_argument("source", help="Arquivo .txt, CSV ou diretório com os textos a analisar.")
    p_ver.add_argument("--out", default=None, help="Caminho do arquivo JSON de saída.")
    p_ver.add_argument("--text-column", default="text", help="Coluna do CSV que contém o texto.")
    p_ver.set_defaults(func=_cmd_veracity)

    p_val = subparsers.add_parser(
        "validade",
        help="Validade integrada: contaminação ideológica + dissimulação (veredito).",
    )
    p_val.add_argument("source", help="Arquivo .txt, CSV ou diretório com os textos a avaliar.")
    p_val.add_argument("--left", default="miriam_leitao/", help="Corpus de referência à esquerda.")
    p_val.add_argument("--right", default="paulo_guedes/", help="Corpus de referência à direita.")
    p_val.add_argument("--label-left", default="miriam", help="Nome do perfil à esquerda.")
    p_val.add_argument("--label-right", default="guedes", help="Nome do perfil à direita.")
    p_val.add_argument("--out", default=None, help="Caminho do arquivo JSON de saída.")
    p_val.add_argument("--text-column", default="text", help="Coluna do CSV que contém o texto.")
    p_val.set_defaults(func=_cmd_validade)

    p_auth = subparsers.add_parser("auth-check", help="Valida um token de acesso.")
    p_auth.add_argument("token", help="Token a validar.")
    p_auth.add_argument("--scope", choices=["read", "write"], default="read")
    p_auth.set_defaults(func=_cmd_auth)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(json_enabled=args.log_json)
    settings = get_settings()

    try:
        return int(args.func(args, settings))
    except EditorialError as exc:
        if args.verbose:
            raise
        print(f"Erro: {exc.user_message}", file=sys.stderr)
        return 1
    except Exception as exc:
        if args.verbose:
            raise
        print(f"Erro inesperado: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
