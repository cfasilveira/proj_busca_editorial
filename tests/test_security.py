"""Testes do módulo de Segurança."""

from __future__ import annotations

import pytest

from editorial.errors import SecurityError
from editorial.security import AccessControl, audit_dependencies


@pytest.fixture
def access():
    from editorial.config import Settings

    settings = Settings(api_token="read-token", admin_token="admin-token")
    return AccessControl(settings)


def test_require_read_allows_correct_token(access):
    assert access.require_read("read-token") is None


def test_require_read_rejects_unknown(access):
    with pytest.raises(SecurityError, match="não autorizado"):
        access.require_read("errado")


def test_require_read_rejects_missing(access):
    with pytest.raises(SecurityError):
        access.require_read(None)


def test_require_write_requires_admin(access):
    assert access.require_write("admin-token") is None
    with pytest.raises(SecurityError):
        access.require_write("read-token")


def test_mask_hides_token():
    assert AccessControl._mask("abcdef123456") == "ab...56"


def test_audit_returns_structure_when_tool_missing():
    import shutil

    if shutil.which("pip-audit"):
        pytest.skip("pip-audit instalado; teste de indisponibilidade não aplicável")

    summary = audit_dependencies()
    assert summary["ok"] is False
    assert "vulnerabilities" in summary
