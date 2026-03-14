from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.security import require_api_key
from config.settings import get_settings


def _reset_settings_cache() -> None:
    get_settings.cache_clear()


def test_require_api_key_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.delenv("API_KEY", raising=False)
    _reset_settings_cache()
    require_api_key(None)


def test_require_api_key_enabled_missing_key(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "secret-123")
    _reset_settings_cache()
    with pytest.raises(HTTPException) as exc:
        require_api_key(None)
    assert exc.value.status_code == 401


def test_require_api_key_enabled_valid_key(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "secret-123")
    _reset_settings_cache()
    require_api_key("secret-123")


def test_validate_runtime_security_rejects_placeholder_api_key(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "change-me-strong-api-key")
    monkeypatch.setenv("NEO4J_PASSWORD", "super-strong-neo4j-password")
    _reset_settings_cache()

    with pytest.raises(ValueError, match="API_KEY"):
        get_settings().validate_runtime_security()


def test_validate_runtime_security_rejects_placeholder_neo4j_password(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("NEO4J_PASSWORD", "change-me-strong-neo4j-password")
    _reset_settings_cache()

    with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
        get_settings().validate_runtime_security()


def test_validate_runtime_security_accepts_strong_values(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "prod-key-very-strong")
    monkeypatch.setenv("NEO4J_PASSWORD", "prod-neo4j-very-strong")
    _reset_settings_cache()

    get_settings().validate_runtime_security()
