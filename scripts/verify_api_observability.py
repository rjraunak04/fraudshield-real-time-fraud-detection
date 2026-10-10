"""Verify FraudShield API robustness and observability."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.scoring.service import FraudScore


class VerificationScorer:
    """Deterministic scorer for observability QA."""

    def score_one(
        self,
        transaction: dict[str, float],
    ) -> FraudScore:
        if list(transaction) != list(
            MODEL_FEATURE_COLUMNS
        ):
            raise RuntimeError(
                "Feature contract changed."
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
    """Run CP-07 HTTP observability QA."""
    app = create_app(
        scorer_dependency=VerificationScorer
    )
    client = TestClient(app)

    payload = {
        feature: 1.0
        for feature in MODEL_FEATURE_COLUMNS
    }

    health = client.get("/health")
    ready = client.get("/ready")

    score = client.post(
        "/v1/score",
        json=payload,
        headers={
            "X-Request-ID": "cp07-verification"
        },
    )

    print()
    print(
        "FraudShield Day-04 CP-07 "
        "API Robustness + Observability"
    )
    print("=" * 90)

    print(
        f"Health: {health.status_code}"
    )
    print(
        f"Readiness: {ready.status_code}"
    )
    print(
        f"Score: {score.status_code}"
    )
    print(
        "Request ID: "
        f"{score.headers['X-Request-ID']}"
    )
    print(
        "Process time (ms): "
        f"{score.headers['X-Process-Time-Ms']}"
    )
    print(
        "Decision: "
        f"{score.json()['action']}"
    )

    assert health.status_code == 200
    assert ready.status_code == 200
    assert score.status_code == 200

    assert (
        score.headers["X-Request-ID"]
        == "cp07-verification"
    )

    assert (
        float(
            score.headers[
                "X-Process-Time-Ms"
            ]
        )
        >= 0.0
    )

    assert score.json()["action"] == "REVIEW"

    print()
    print("Locked test used: False")
    print("Model retrained: False")
    print(
        "CP-07 API robustness + "
        "observability QA: PASSED"
    )


if __name__ == "__main__":
    main()