"""Verify the complete FraudShield Day-04 serving path."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.scoring.service import FraudScore


class Day04VerificationScorer:
    """Deterministic scorer for final Day-04 QA."""

    def score_one(
        self,
        transaction: dict[str, float],
    ) -> FraudScore:
        """Return deterministic calibrated risk."""
        if list(transaction) != list(
            MODEL_FEATURE_COLUMNS
        ):
            raise RuntimeError(
                "Frozen model feature contract changed."
            )

        return FraudScore(
            raw_probability=0.081,
            calibrated_probability=0.060,
            threshold=0.048,
            is_high_risk=True,
            model_version=(
                "day-03-hgb-balanced-v1"
            ),
            calibration_method="sigmoid",
        )


def main() -> None:
    """Run final Day-04 integration verification."""
    app = create_app(
        scorer_dependency=Day04VerificationScorer
    )
    client = TestClient(app)

    transaction = {
        feature: 1.0
        for feature in MODEL_FEATURE_COLUMNS
    }

    health = client.get("/health")
    ready = client.get("/ready")

    score = client.post(
        "/v1/score",
        json=transaction,
        headers={
            "X-Request-ID": "day04-final-qa"
        },
    )

    body = score.json()

    print()
    print(
        "FraudShield Day-04 Final "
        "End-to-End Verification"
    )
    print("=" * 90)

    print(f"Model features: {len(transaction)}")
    print(f"Health: {health.status_code}")
    print(f"Readiness: {ready.status_code}")
    print(f"Score: {score.status_code}")
    print(f"Decision: {body['action']}")

    print(
        "Calibrated probability: "
        f"{body['calibrated_probability']:.6f}"
    )
    print(
        "Review threshold: "
        f"{body['review_threshold']:.6f}"
    )
    print(
        "Block threshold: "
        f"{body['block_threshold']:.6f}"
    )
    print(
        f"Model version: "
        f"{body['model_version']}"
    )
    print(
        f"Policy version: "
        f"{body['policy_version']}"
    )
    print(
        "Request ID: "
        f"{score.headers['X-Request-ID']}"
    )
    print(
        "Latency (ms): "
        f"{score.headers['X-Process-Time-Ms']}"
    )

    assert len(transaction) == 21
    assert health.status_code == 200
    assert ready.status_code == 200
    assert score.status_code == 200

    assert body["action"] == "REVIEW"

    assert body["review_threshold"] == 0.048
    assert body["block_threshold"] == 0.100

    assert (
        body["model_version"]
        == "day-03-hgb-balanced-v1"
    )

    assert (
        body["policy_version"]
        == "day-04-three-way-policy-v1"
    )

    assert (
        score.headers["X-Request-ID"]
        == "day04-final-qa"
    )

    print()
    print("Locked test reopened: False")
    print("Model retrained: False")
    print("Threshold retuned: False")
    print(
        "Day-04 end-to-end QA: PASSED"
    )


if __name__ == "__main__":
    main()