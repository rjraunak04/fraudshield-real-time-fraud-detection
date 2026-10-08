"""Tests for FraudShield API observability."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.scoring.service import FraudScore


class HealthyScorer:
    """Deterministic scorer used by API tests."""

    def score_one(
        self,
        transaction: dict[str, float],
    ) -> FraudScore:
        del transaction

        return FraudScore(
            raw_probability=0.05,
            calibrated_probability=0.06,
            threshold=0.048,
            is_high_risk=True,
            model_version="test-model-v1",
            calibration_method="sigmoid",
        )


def valid_payload() -> dict[str, float]:
    """Return one valid engineered transaction."""
    return {
        feature: 1.0
        for feature in MODEL_FEATURE_COLUMNS
    }


def test_request_id_is_generated() -> None:
    app = create_app(
        scorer_dependency=lambda: HealthyScorer()
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_client_request_id_is_preserved() -> None:
    app = create_app(
        scorer_dependency=lambda: HealthyScorer()
    )
    client = TestClient(app)

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": "fraudshield-test-123"
        },
    )

    assert (
        response.headers["X-Request-ID"]
        == "fraudshield-test-123"
    )


def test_process_time_header_exists() -> None:
    app = create_app(
        scorer_dependency=lambda: HealthyScorer()
    )
    client = TestClient(app)

    response = client.get("/health")

    latency = float(
        response.headers["X-Process-Time-Ms"]
    )

    assert latency >= 0.0


def test_ready_when_scorer_available() -> None:
    app = create_app(
        scorer_dependency=lambda: HealthyScorer()
    )
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "fraudshield",
    }


def test_not_ready_without_scorer() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503


def test_score_has_observability_headers() -> None:
    app = create_app(
        scorer_dependency=lambda: HealthyScorer()
    )
    client = TestClient(app)

    response = client.post(
        "/v1/score",
        json=valid_payload(),
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert (
        float(
            response.headers[
                "X-Process-Time-Ms"
            ]
        )
        >= 0.0
    )


def test_validation_error_keeps_request_id() -> None:
    app = create_app(
        scorer_dependency=lambda: HealthyScorer()
    )
    client = TestClient(app)

    response = client.post(
        "/v1/score",
        json={},
        headers={
            "X-Request-ID": "invalid-request-1"
        },
    )

    assert response.status_code == 422
    assert (
        response.headers["X-Request-ID"]
        == "invalid-request-1"
    )


def test_service_error_keeps_request_id() -> None:
    class BrokenScorer:
        def score_one(
            self,
            transaction: dict[str, float],
        ) -> FraudScore:
            del transaction
            raise RuntimeError(
                "model unavailable"
            )

    app = create_app(
        scorer_dependency=lambda: BrokenScorer()
    )
    client = TestClient(app)

    response = client.post(
        "/v1/score",
        json=valid_payload(),
        headers={
            "X-Request-ID": "failed-score-1"
        },
    )

    assert response.status_code == 500
    assert (
        response.headers["X-Request-ID"]
        == "failed-score-1"
    )


def test_openapi_contains_readiness() -> None:
    app = create_app(
        scorer_dependency=lambda: HealthyScorer()
    )
    client = TestClient(app)

    schema = client.get(
        "/openapi.json"
    ).json()

    assert "/ready" in schema["paths"]