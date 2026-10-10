"""End-to-end integration tests for FraudShield Day-04."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.scoring.service import FraudScore


class IntegrationScorer:
    """Deterministic scorer for Day-04 integration QA."""

    def __init__(
        self,
        calibrated_probability: float,
    ) -> None:
        self.calibrated_probability = (
            calibrated_probability
        )

    def score_one(
        self,
        transaction: dict[str, float],
    ) -> FraudScore:
        """Return a deterministic score."""
        if list(transaction) != list(
            MODEL_FEATURE_COLUMNS
        ):
            raise RuntimeError(
                "Feature contract changed."
            )

        probability = (
            self.calibrated_probability
        )

        return FraudScore(
            raw_probability=probability,
            calibrated_probability=probability,
            threshold=0.048,
            is_high_risk=probability >= 0.048,
            model_version=(
                "day-03-hgb-balanced-v1"
            ),
            calibration_method="sigmoid",
        )


def payload() -> dict[str, float]:
    """Return a valid engineered feature payload."""
    return {
        feature: 1.0
        for feature in MODEL_FEATURE_COLUMNS
    }


def client_for_probability(
    probability: float,
) -> TestClient:
    """Create API client with deterministic scorer."""
    scorer = IntegrationScorer(probability)

    return TestClient(
        create_app(
            scorer_dependency=lambda: scorer
        )
    )


def test_end_to_end_approve() -> None:
    client = client_for_probability(0.020)

    response = client.post(
        "/v1/score",
        json=payload(),
    )

    assert response.status_code == 200
    assert response.json()["action"] == "APPROVE"


def test_end_to_end_review() -> None:
    client = client_for_probability(0.060)

    response = client.post(
        "/v1/score",
        json=payload(),
    )

    assert response.status_code == 200
    assert response.json()["action"] == "REVIEW"


def test_end_to_end_block() -> None:
    client = client_for_probability(0.150)

    response = client.post(
        "/v1/score",
        json=payload(),
    )

    assert response.status_code == 200
    assert response.json()["action"] == "BLOCK"


def test_versions_survive_full_path() -> None:
    client = client_for_probability(0.060)

    response = client.post(
        "/v1/score",
        json=payload(),
    )

    body = response.json()

    assert (
        body["model_version"]
        == "day-03-hgb-balanced-v1"
    )
    assert (
        body["calibration_method"]
        == "sigmoid"
    )
    assert (
        body["policy_version"]
        == "day-04-three-way-policy-v1"
    )


def test_thresholds_survive_full_path() -> None:
    client = client_for_probability(0.060)

    body = client.post(
        "/v1/score",
        json=payload(),
    ).json()

    assert body["review_threshold"] == 0.048
    assert body["block_threshold"] == 0.100


def test_observability_survives_full_path() -> None:
    client = client_for_probability(0.060)

    response = client.post(
        "/v1/score",
        json=payload(),
        headers={
            "X-Request-ID": "day04-integration"
        },
    )

    assert (
        response.headers["X-Request-ID"]
        == "day04-integration"
    )

    assert (
        float(
            response.headers[
                "X-Process-Time-Ms"
            ]
        )
        >= 0.0
    )


def test_health_and_readiness() -> None:
    client = client_for_probability(0.060)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_invalid_payload_rejected() -> None:
    client = client_for_probability(0.060)

    invalid = payload()
    invalid.pop(MODEL_FEATURE_COLUMNS[0])

    response = client.post(
        "/v1/score",
        json=invalid,
    )

    assert response.status_code == 422