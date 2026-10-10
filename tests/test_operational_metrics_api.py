"""Tests for FraudShield operational dashboard API."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from fraudshield.api.app import create_app
from fraudshield.persistence import (
    FraudDecisionRepository,
    FraudDecisionWrite,
    create_schema,
    create_session_factory,
)


@pytest.fixture
def repository():
    """Create thread-safe shared in-memory dashboard repository."""
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

    repo = FraudDecisionRepository(session)

    yield repo

    session.close()
    engine.dispose()


def add_decision(
    repository: FraudDecisionRepository,
    *,
    transaction_id: int,
    minute: int,
    amount: float,
    probability: float,
    decision: str,
) -> None:
    """Persist one deterministic dashboard decision."""
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


def test_unconfigured_metrics_returns_503():
    """Default application must fail metrics safely."""
    app = create_app()
    client = TestClient(app)

    response = client.get(
        "/v1/metrics/operations"
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Operational metrics are unavailable."
        )
    }


def test_empty_dashboard_returns_zero_metrics(
    repository,
):
    """Empty history must produce zero-safe metrics."""
    app = create_app(
        metrics_repository_dependency=(
            lambda: repository
        )
    )
    client = TestClient(app)

    response = client.get(
        "/v1/metrics/operations"
    )

    assert response.status_code == 200

    payload = response.json()
    metrics = payload["metrics"]

    assert metrics["total_transactions"] == 0
    assert metrics["approve_count"] == 0
    assert metrics["review_count"] == 0
    assert metrics["block_count"] == 0

    assert metrics["approve_rate"] == 0.0
    assert metrics["review_rate"] == 0.0
    assert metrics["block_rate"] == 0.0

    assert (
        metrics["average_calibrated_probability"]
        == 0.0
    )

    assert metrics["high_risk_rate"] == 0.0

    assert (
        metrics["total_transaction_amount"]
        == 0.0
    )

    assert (
        metrics["average_transaction_amount"]
        == 0.0
    )

    assert payload["recent_decisions"] == []


def test_dashboard_returns_metrics_and_recent_decisions(
    repository,
):
    """Dashboard must expose aggregates and recent history."""
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

    app = create_app(
        metrics_repository_dependency=(
            lambda: repository
        )
    )
    client = TestClient(app)

    response = client.get(
        "/v1/metrics/operations"
    )

    assert response.status_code == 200

    payload = response.json()
    metrics = payload["metrics"]

    assert metrics["total_transactions"] == 3

    assert metrics["approve_count"] == 1
    assert metrics["review_count"] == 1
    assert metrics["block_count"] == 1

    assert metrics["approve_rate"] == pytest.approx(
        1 / 3
    )

    assert metrics["review_rate"] == pytest.approx(
        1 / 3
    )

    assert metrics["block_rate"] == pytest.approx(
        1 / 3
    )

    assert (
        metrics["average_calibrated_probability"]
        == pytest.approx(
            (0.02 + 0.06 + 0.12) / 3
        )
    )

    assert metrics["high_risk_rate"] == pytest.approx(
        2 / 3
    )

    assert (
        metrics["total_transaction_amount"]
        == pytest.approx(400.0)
    )

    assert (
        metrics["average_transaction_amount"]
        == pytest.approx(400.0 / 3)
    )

    recent = payload["recent_decisions"]

    assert len(recent) == 3

    assert [
        item["transaction_id"]
        for item in recent
    ] == [3, 2, 1]

    assert [
        item["decision"]
        for item in recent
    ] == [
        "BLOCK",
        "REVIEW",
        "APPROVE",
    ]

    assert (
        recent[0]["model_version"]
        == "day-03-hgb-balanced-v1"
    )

    assert (
        recent[0]["policy_version"]
        == "day-04-three-way-policy-v1"
    )


def test_dashboard_preserves_observability_headers(
    repository,
):
    """Dashboard must inherit API observability middleware."""
    app = create_app(
        metrics_repository_dependency=(
            lambda: repository
        )
    )
    client = TestClient(app)

    response = client.get(
        "/v1/metrics/operations",
        headers={
            "X-Request-ID": "cp07-dashboard-test"
        },
    )

    assert response.status_code == 200

    assert (
        response.headers["X-Request-ID"]
        == "cp07-dashboard-test"
    )

    assert (
        "X-Process-Time-Ms"
        in response.headers
    )