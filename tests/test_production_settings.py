"""Tests for environment-driven production settings."""

import pytest

from fraudshield.production.settings import ProductionSettings

_REQUIRED = {
    "FRAUDSHIELD_DATABASE_URL": "sqlite:///fraudshield.db",
    "FRAUDSHIELD_REDIS_URL": "redis://localhost:6379/0",
    "FRAUDSHIELD_MODEL_ARTIFACT": "artifacts/model.pkl",
    "FRAUDSHIELD_CALIBRATOR_ARTIFACT": "artifacts/calibrator.pkl",
}


def test_production_settings_load_required_environment(monkeypatch):
    for name, value in _REQUIRED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FRAUDSHIELD_ENV", "production")

    settings = ProductionSettings.from_env()

    assert settings.environment == "production"
    assert settings.database_url == "sqlite:///fraudshield.db"
    assert settings.redis_namespace == "fraudshield"
    assert settings.model_artifact_path.as_posix() == "artifacts/model.pkl"


def test_production_settings_fail_fast_when_required_value_missing(monkeypatch):
    for name in _REQUIRED:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="Missing required FraudShield settings"):
        ProductionSettings.from_env()


def test_production_settings_reject_empty_namespace(monkeypatch):
    for name, value in _REQUIRED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("FRAUDSHIELD_REDIS_NAMESPACE", "   ")

    with pytest.raises(RuntimeError, match="NAMESPACE"):
        ProductionSettings.from_env()
