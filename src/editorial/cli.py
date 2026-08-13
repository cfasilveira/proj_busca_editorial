"""Interface de linha de comando (CLI) do aplicativo editorial.

Fluxo típico:
    editorial pipeline data/samples/editorials.csv --outdir reports
    editorial search "texto de busca" --index data/index.faiss --model data/embedding_model.joblib
    editorial audit

Erros são tratados com "fail gracefully": mensagens claras e exit code
diferente de zero, sem stack trace para o usuário final (a não ser com -v).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import Settings, get_settings
from .errors import EditorialError
from .logging_setup import setup_logging
from .pipeline import run_pipeline
from .security import AccessControl, audit_dependencies
from .vector import EmbeddingModel, FaissStore


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _cmd_pipeline(args: argparse.Namespace, settings: Settings) -> int:
    result = run_pipeline(args.csv, args.outdir, settings, text_column=args.text_column)
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
        "pipeline", help="Executa o pipeline completo a partir de um CSV."
    )
    p_pipe.add_argument("csv", help="Caminho do CSV de entrada.")
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
