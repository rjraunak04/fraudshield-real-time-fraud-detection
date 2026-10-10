"""Tests for the FraudShield FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.explainability.reason_codes import ReasonCode
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.scoring.service import FraudScore


class StubScorer:
    """Deterministic scorer for API tests."""

    def __init__(
        self,
        probability: float,
    ) -> None:
        self.probability = probability
        self.last_transaction: (
            dict[str, float] | None
        ) = None

    def score_one(
        self,
        transaction: dict[str, float],
    ) -> FraudScore:
        self.last_transaction = transaction

        return FraudScore(
            raw_probability=self.probability,
            calibrated_probability=(
                self.probability
            ),
            threshold=0.048,
            is_high_risk=(
                self.probability >= 0.048
            ),
            model_version="test-model-v1",
            calibration_method="sigmoid",
        )


class StubReasonService:
    """Deterministic local reason service."""

    def explain(
        self,
        transaction: dict[str, float],
    ) -> tuple[ReasonCode, ...]:
        del transaction

        return (
            ReasonCode(
                code=(
                    "HIGH_TRANSACTION_AMOUNT_RISK"
                ),
                message=(
                    "Transaction amount contributes "
                    "to elevated model risk."
                ),
                feature="TX_AMOUNT",
                direction="increases_risk",
                probability_change=0.10,
                rank=1,
            ),
        )


def valid_payload() -> dict[str, float]:
    """Build one valid 21-feature request."""
    return {
        feature: 1.0
        for feature in MODEL_FEATURE_COLUMNS
    }


def client_for_probability(
    probability: float,
    *,
    with_reasons: bool = False,
) -> tuple[TestClient, StubScorer]:
    """Build test client with deterministic services."""
    scorer = StubScorer(probability)

    kwargs = {}

    if with_reasons:
        reason_service = StubReasonService()
        kwargs["reason_dependency"] = (
            lambda: reason_service
        )

    app = create_app(
        scorer_dependency=lambda: scorer,
        **kwargs,
    )

    return TestClient(app), scorer


def test_health_endpoint() -> None:
    client, _ = client_for_probability(
        0.01
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "fraudshield",
    }


def test_approve_response() -> None:
    client, _ = client_for_probability(
        0.020
    )

    response = client.post(
        "/v1/score",
        json=valid_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["action"] == "APPROVE"
    assert (
        body["calibrated_probability"]
        == 0.020
    )


def test_review_response() -> None:
    client, _ = client_for_probability(
        0.060
    )

    response = client.post(
        "/v1/score",
        json=valid_payload(),
    )

    assert response.status_code == 200
    assert (
        response.json()["action"]
        == "REVIEW"
    )


def test_block_response() -> None:
    client, _ = client_for_probability(
        0.150
    )

    response = client.post(
        "/v1/score",
        json=valid_payload(),
    )

    assert response.status_code == 200
    assert (
        response.json()["action"]
        == "BLOCK"
    )


def test_response_contains_versions() -> None:
    client, _ = client_for_probability(
        0.060
    )

    body = client.post(
        "/v1/score",
        json=valid_payload(),
    ).json()

    assert (
        body["model_version"]
        == "test-model-v1"
    )
    assert (
        body["calibration_method"]
        == "sigmoid"
    )
    assert (
        body["policy_version"]
        == "day-04-three-way-policy-v1"
    )


def test_response_contains_thresholds() -> None:
    client, _ = client_for_probability(
        0.060
    )

    body = client.post(
        "/v1/score",
        json=valid_payload(),
    ).json()

    assert body["review_threshold"] == 0.048
    assert body["block_threshold"] == 0.100


def test_reason_codes_are_returned() -> None:
    client, _ = client_for_probability(
        0.150,
        with_reasons=True,
    )

    body = client.post(
        "/v1/score",
        json=valid_payload(),
    ).json()

    assert len(body["reason_codes"]) == 1

    assert (
        body["reason_codes"][0]["code"]
        == "HIGH_TRANSACTION_AMOUNT_RISK"
    )


def test_request_uses_exact_feature_order() -> None:
    client, scorer = (
        client_for_probability(0.020)
    )

    response = client.post(
        "/v1/score",
        json=valid_payload(),
    )

    assert response.status_code == 200

    assert scorer.last_transaction is not None

    assert (
        list(scorer.last_transaction)
        == list(MODEL_FEATURE_COLUMNS)
    )


def test_missing_feature_returns_422() -> None:
    client, _ = client_for_probability(
        0.020
    )

    payload = valid_payload()
    payload.pop(
        MODEL_FEATURE_COLUMNS[0]
    )

    response = client.post(
        "/v1/score",
        json=payload,
    )

    assert response.status_code == 422


def test_extra_feature_returns_422() -> None:
    client, _ = client_for_probability(
        0.020
    )

    payload = valid_payload()
    payload["LEAKAGE_FEATURE"] = 1.0

    response = client.post(
        "/v1/score",
        json=payload,
    )

    assert response.status_code == 422


def test_non_numeric_feature_returns_422() -> None:
    client, _ = client_for_probability(
        0.020
    )

    payload = valid_payload()
    payload["TX_AMOUNT"] = "not-a-number"

    response = client.post(
        "/v1/score",
        json=payload,
    )

    assert response.status_code == 422


def test_null_feature_returns_422() -> None:
    client, _ = client_for_probability(
        0.020
    )

    payload = valid_payload()
    payload["TX_AMOUNT"] = None

    response = client.post(
        "/v1/score",
        json=payload,
    )

    assert response.status_code == 422


def test_service_failure_is_safe() -> None:
    class BrokenScorer:
        def score_one(
            self,
            transaction: dict[str, float],
        ) -> FraudScore:
            del transaction

            raise RuntimeError(
                "internal model failure"
            )

    app = create_app(
        scorer_dependency=(
            lambda: BrokenScorer()
        )
    )

    client = TestClient(app)

    response = client.post(
        "/v1/score",
        json=valid_payload(),
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "Fraud scoring failed safely."
        )
    }


def test_openapi_schema_available() -> None:
    client, _ = client_for_probability(
        0.020
    )

    response = client.get(
        "/openapi.json"
    )

    assert response.status_code == 200

    schema = response.json()

    assert "/health" in schema["paths"]
    assert "/v1/score" in schema["paths"]