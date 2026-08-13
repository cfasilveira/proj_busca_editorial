"""Auditoria de dependências (pip-audit / OSV).

Executa `pip-audit` (extra opcional `audit`) e consolida o resultado.
Fail gracefully: se o pip-audit não estiver instalado ou falhar, retorna
status claro com orientação, sem travar o processo. Vulnerabilidades
encontradas são registradas em log estruturado.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from ..errors import SecurityError, fail
from ..logging_setup import get_logger

logger = get_logger(__name__)


def _find_python() -> str:
    import sys

    return sys.executable


def audit_dependencies() -> dict[str, Any]:
    """Executa a auditoria e retorna resumo estruturado."""
    if shutil.which("pip-audit") is None:
        message = (
            "pip-audit não está instalado. Execute 'uv sync --extra audit' "
            "para habilitar a auditoria de dependências."
        )
        logger.warning(
            "Auditoria de dependências indisponível",
            extra={"code": "audit_tool_missing", "status": "skipped"},
        )
        return {"ok": False, "reason": message, "vulnerabilities": []}

    try:
        result = subprocess.run(
            [
                _find_python(),
                "-m",
                "pip_audit",
                "--format",
                "json",
                "--skip-editable",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        fail(
            logger,
            SecurityError,
            f"Auditoria de dependências falhou: {exc}",
            user_message="Não foi possível executar a auditoria de dependências.",
            code="audit_execution_failed",
        )

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}

    vulnerabilities: list[dict[str, Any]] = payload.get("vulnerabilities", []) or []
    dependency_count = len(payload.get("dependencies", []))
    vulnerable_dependencies = payload.get("dependencies_vulnerabilities", {}) or {}

    for finding in vulnerabilities:
        logger.warning(
            "Vulnerabilidade em dependência",
            extra={
                "code": "dependency_vulnerability",
                "metric": f"{finding.get('name')}=={finding.get('version')}",
            },
        )

    summary = {
        "ok": not bool(vulnerabilities),
        "vulnerabilities": vulnerabilities,
        "dependency_count": dependency_count,
        "vulnerable_dependencies": vulnerable_dependencies,
    }
    logger.info(
        "Auditoria de dependências concluída",
        extra={
            "code": "audit_done",
            "status": "ok" if summary["ok"] else "vulnerable",
            "metric": f"{len(vulnerabilities)} vulnerabilidades",
        },
    )
    return summary
