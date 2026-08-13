"""Módulo de Segurança: auditoria de dependências e controle de acesso."""

from .access import AccessControl
from .audit import audit_dependencies

__all__ = [
    "AccessControl",
    "audit_dependencies",
]
