"""Verify the FraudShield FastAPI contract."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudshield.api.app import create_app
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.scoring.service import FraudScore


class VerificationScorer:
    """Deterministic scorer for API verification."""

    def score_one(
        self,
        transaction: dict[str, float],
    ) -> FraudScore:
        """Return a deterministic fraud score."""
        if list(transaction) != list(
            MODEL_FEATURE_COLUMNS
        ):
            raise RuntimeError(
                "Model feature order changed."
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
    """Run HTTP-level inference QA."""
    scorer = VerificationScorer()

    app = create_app(
        scorer_dependency=lambda: scorer
    )

    client = TestClient(app)

    payload = {
        feature: 1.0
        for feature in MODEL_FEATURE_COLUMNS
    }

    health_response = client.get(
        "/health"
    )

    score_response = client.post(
        "/v1/score",
        json=payload,
    )

    invalid_payload = payload.copy()
    invalid_payload.pop(
        MODEL_FEATURE_COLUMNS[0]
    )

    invalid_response = client.post(
        "/v1/score",
        json=invalid_payload,
    )

    openapi_response = client.get(
        "/openapi.json"
    )

    print()
    print(
        "FraudShield Day-04 CP-06 "
        "FastAPI Inference Service"
    )
    print("=" * 100)

    print(
        f"Health status: "
        f"{health_response.status_code}"
    )
    print(
        f"Score status: "
        f"{score_response.status_code}"
    )
    print(
        f"Invalid request status: "
        f"{invalid_response.status_code}"
    )
    print(
        f"OpenAPI status: "
        f"{openapi_response.status_code}"
    )

    body = score_response.json()

    print()
    print("Inference response")
    print("-" * 100)

    print(
        f"Action: "
        f"{body['action']}"
    )
    print(
        "Raw probability: "
        f"{body['raw_probability']:.6f}"
    )
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

    if health_response.status_code != 200:
        raise RuntimeError(
            "Health endpoint failed."
        )

    if score_response.status_code != 200:
        raise RuntimeError(
            "Score endpoint failed."
        )

    if body["action"] != "REVIEW":
        raise RuntimeError(
            "Expected REVIEW action."
        )

    if invalid_response.status_code != 422:
        raise RuntimeError(
            "Invalid request was not rejected."
        )

    if openapi_response.status_code != 200:
        raise RuntimeError(
            "OpenAPI schema unavailable."
        )

    if (
        body["model_version"]
        != "day-03-hgb-balanced-v1"
    ):
        raise RuntimeError(
            "Unexpected model version."
        )

    if (
        body["policy_version"]
        != "day-04-three-way-policy-v1"
    ):
        raise RuntimeError(
            "Unexpected policy version."
        )

    print()
    print("Locked test used: False")
    print("Production model retrained: False")
    print(
        "CP-06 FastAPI inference-service QA: PASSED"
    )


if __name__ == "__main__":
    main()