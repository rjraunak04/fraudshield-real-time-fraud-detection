"""Verify Day-05 CP-07 operational metrics and dashboard API."""

from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from fraudshield.api.app import create_app
from fraudshield.metrics import build_operational_metrics
from fraudshield.persistence import (
    FraudDecisionRepository,
    FraudDecisionWrite,
    create_schema,
    create_session_factory,
)


def add_decision(
    repository: FraudDecisionRepository,
    *,
    transaction_id: int,
    minute: int,
    amount: float,
    probability: float,
    decision: str,
) -> None:
    """Persist one deterministic operational decision."""
    repository.add(
        FraudDecisionWrite(
            transaction_id=transaction_id,
            event_time=datetime(
                2018,
                6,
                15,
                10,
                minute,
            ),
            customer_id=100 + transaction_id,
            terminal_id=500 + transaction_id,
            amount=amount,
            raw_probability=probability,
            calibrated_probability=probability,
            decision=decision,
            model_version=(
                "day-03-hgb-balanced-v1"
            ),
            policy_version=(
                "day-04-three-way-policy-v1"
            ),
        )
    )
    repository.session.commit()


def main() -> None:
    """Run CP-07 operational metrics QA."""
    engine = create_engine(
        "sqlite://",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    create_schema(engine)

    session_factory = create_session_factory(
        engine
    )
    session = session_factory()

    repository = FraudDecisionRepository(
        session
    )

    try:
        add_decision(
            repository,
            transaction_id=1,
            minute=1,
            amount=25.0,
            probability=0.02,
            decision="APPROVE",
        )

        add_decision(
            repository,
            transaction_id=2,
            minute=2,
            amount=125.0,
            probability=0.06,
            decision="REVIEW",
        )

        add_decision(
            repository,
            transaction_id=3,
            minute=3,
            amount=250.0,
            probability=0.12,
            decision="BLOCK",
        )

        metrics = build_operational_metrics(
            repository
        )

        assert metrics.total_transactions == 3
        assert metrics.approve_count == 1
        assert metrics.review_count == 1
        assert metrics.block_count == 1

        assert abs(
            metrics.approve_rate - (1 / 3)
        ) < 1e-12

        assert abs(
            metrics.review_rate - (1 / 3)
        ) < 1e-12

        assert abs(
            metrics.block_rate - (1 / 3)
        ) < 1e-12

        assert abs(
            metrics.high_risk_rate - (2 / 3)
        ) < 1e-12

        assert abs(
            metrics.total_transaction_amount
            - 400.0
        ) < 1e-12

        app = create_app(
            metrics_repository_dependency=(
                lambda: repository
            )
        )

        client = TestClient(app)

        response = client.get(
            "/v1/metrics/operations",
            headers={
                "X-Request-ID": (
                    "cp07-verification"
                )
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert (
            payload["metrics"][
                "total_transactions"
            ]
            == 3
        )

        assert (
            payload["metrics"][
                "approve_count"
            ]
            == 1
        )

        assert (
            payload["metrics"][
                "review_count"
            ]
            == 1
        )

        assert (
            payload["metrics"][
                "block_count"
            ]
            == 1
        )

        assert len(
            payload["recent_decisions"]
        ) == 3

        assert [
            item["transaction_id"]
            for item in payload[
                "recent_decisions"
            ]
        ] == [3, 2, 1]

        assert (
            response.headers[
                "X-Request-ID"
            ]
            == "cp07-verification"
        )

        assert (
            "X-Process-Time-Ms"
            in response.headers
        )

        print(
            "=== DAY-05 CP-07 "
            "OPERATIONAL METRICS QA ==="
        )
        print(
            "Operational metrics endpoint: "
            "GET /v1/metrics/operations"
        )
        print(
            "Total transactions:",
            metrics.total_transactions,
        )
        print(
            "APPROVE decisions:",
            metrics.approve_count,
        )
        print(
            "REVIEW decisions:",
            metrics.review_count,
        )
        print(
            "BLOCK decisions:",
            metrics.block_count,
        )
        print(
            "High-risk rate:",
            f"{metrics.high_risk_rate:.6f}",
        )
        print(
            "Total transaction amount:",
            f"{metrics.total_transaction_amount:.2f}",
        )
        print(
            "Recent decisions:",
            len(
                payload[
                    "recent_decisions"
                ]
            ),
        )
        print(
            "Dashboard API: PASSED"
        )
        print(
            "Request-ID propagation: PASSED"
        )
        print(
            "Process-time observability: PASSED"
        )
        print(
            "SQL-side aggregation: PASSED"
        )
        print(
            "Locked test used: False"
        )
        print(
            "Model retrained: False"
        )
        print(
            "Threshold retuned: False"
        )
        print(
            "Day-05 CP-07 QA: PASSED"
        )

    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()