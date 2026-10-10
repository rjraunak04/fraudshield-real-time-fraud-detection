"""Static contract tests for production deployment packaging."""

from pathlib import Path

import yaml


def _compose() -> dict:
    return yaml.safe_load(
        Path("compose.production.yaml").read_text(encoding="utf-8")
    )


def test_production_stack_declares_required_services():
    services = _compose()["services"]

    assert set(services) == {"api", "postgres", "redis"}


def test_api_uses_production_entrypoint_and_readiness_healthcheck():
    api = _compose()["services"]["api"]

    command = " ".join(api["command"])
    healthcheck = " ".join(api["healthcheck"]["test"])

    assert "fraudshield.production.asgi:app" in command
    assert "/ready/realtime" in healthcheck


def test_api_waits_for_stateful_dependencies():
    depends_on = _compose()["services"]["api"]["depends_on"]

    assert depends_on["postgres"]["condition"] == "service_healthy"
    assert depends_on["redis"]["condition"] == "service_healthy"


def test_model_artifacts_are_mounted_read_only():
    volumes = _compose()["services"]["api"]["volumes"]

    assert "./artifacts:/app/artifacts:ro" in volumes


def test_postgres_password_is_not_hard_coded():
    compose_text = Path("compose.production.yaml").read_text(encoding="utf-8")

    assert "FRAUDSHIELD_POSTGRES_PASSWORD:?" in compose_text
    assert "change-me" not in compose_text
