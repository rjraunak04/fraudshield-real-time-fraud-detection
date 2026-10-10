"""Tests for the raw-transaction real-time FraudShield API."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from fraudshield.api.app import create_app
from fraudshield.decision import RiskDecisionEngine
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.persistence import (
    FraudDecisionRepository,
    create_schema,
    create_session_factory,
)
from fraudshield.realtime import RealTimeFraudPipeline
from fraudshield.scoring.service import FraudScorer
from fraudshield.state.store import InMemoryStateStore


class FixedModel:
    """Deterministic model for API integration tests."""

    def __init__(self, probability: float) -> None:
        self._probability = probability

    def predict_proba(self, X):
        probability = np.full(
            len(X),
            self._probability,
            dtype=float,
        )

        return np.column_stack(
            (1.0 - probability, probability)
        )


class IdentityCalibrator:
    """Return model probabilities unchanged."""

    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=float)


class FailingRealtimeService:
    """Real-time service that fails deterministically."""

    def process(self, event):
        raise RuntimeError(
            "simulated real-time failure"
        )


@pytest.fixture
def realtime_components():
    """Create isolated real-time API dependencies."""
    database_engine = create_engine(
        "sqlite://",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    create_schema(database_engine)

    session_factory = create_session_factory(
        database_engine
    )
    session: Session = session_factory()

    state = InMemoryStateStore()
    feature_pipeline = OnlineFeaturePipeline(state)

    scorer = FraudScorer(
        model=FixedModel(0.06),
        calibrator=IdentityCalibrator(),
    )

    repository = FraudDecisionRepository(session)

    pipeline = RealTimeFraudPipeline(
        feature_pipeline=feature_pipeline,
        scorer=scorer,
        decision_engine=RiskDecisionEngine(),
        repository=repository,
        session=session,
    )

    yield {
        "pipeline": pipeline,
        "state": state,
        "repository": repository,
        "session": session,
    }

    session.close()
    database_engine.dispose()


def valid_payload(
    transaction_id: int = 1,
) -> dict[str, object]:
    """Return one valid raw transaction payload."""
    return {
        "transaction_id": transaction_id,
        "event_time": "2018-06-15T14:30:00",
        "customer_id": 101,
        "terminal_id": 501,
        "amount": 125.50,
    }


def test_realtime_readiness_unconfigured_returns_503():
    """Unconfigured real-time service must fail readiness."""
    app = create_app()
    client = TestClient(app)

    response = client.get(
        "/ready/realtime"
    )

    assert response.status_code == 503


def test_realtime_readiness_configured_returns_200(
    realtime_components,
):
    """Configured real-time service must report readiness."""
    pipeline = realtime_components["pipeline"]

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    response = client.get(
        "/ready/realtime"
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "fraudshield-realtime",
    }


def test_raw_transaction_returns_decision(
    realtime_components,
):
    """Valid raw transaction must return fraud decision."""
    pipeline = realtime_components["pipeline"]

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    response = client.post(
        "/v1/transactions/score",
        json=valid_payload(),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["transaction_id"] == 1
    assert payload["action"] == "REVIEW"

    assert payload[
        "raw_probability"
    ] == pytest.approx(0.06)

    assert payload[
        "calibrated_probability"
    ] == pytest.approx(0.06)

    assert payload[
        "review_threshold"
    ] == pytest.approx(0.048)

    assert payload[
        "block_threshold"
    ] == pytest.approx(0.100)

    assert (
        payload["model_version"]
        == "day-03-hgb-balanced-v1"
    )

    assert (
        payload["policy_version"]
        == "day-04-three-way-policy-v1"
    )


def test_success_persists_and_updates_state(
    realtime_components,
):
    """Successful API request must persist and update state."""
    pipeline = realtime_components["pipeline"]
    state = realtime_components["state"]
    repository = realtime_components["repository"]

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    response = client.post(
        "/v1/transactions/score",
        json=valid_payload(),
    )

    assert response.status_code == 200
    assert repository.count() == 1

    history = state.customer_history(
        101,
        datetime(2018, 6, 16),
    )

    assert history.count == 1


def test_duplicate_transaction_returns_409(
    realtime_components,
):
    """Duplicate transaction must return conflict."""
    pipeline = realtime_components["pipeline"]
    state = realtime_components["state"]
    repository = realtime_components["repository"]

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    first = client.post(
        "/v1/transactions/score",
        json=valid_payload(),
    )

    second = client.post(
        "/v1/transactions/score",
        json=valid_payload(),
    )

    assert first.status_code == 200
    assert second.status_code == 409

    assert second.json() == {
        "detail": (
            "Transaction has already been processed."
        )
    }

    assert repository.count() == 1

    history = state.customer_history(
        101,
        datetime(2018, 6, 16),
    )

    assert history.count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_id", -1),
        ("customer_id", -1),
        ("terminal_id", -1),
        ("amount", -1.0),
    ],
)
def test_invalid_nonnegative_fields_return_422(
    realtime_components,
    field,
    value,
):
    """Negative raw transaction values must be rejected."""
    pipeline = realtime_components["pipeline"]

    payload = valid_payload()
    payload[field] = value

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    response = client.post(
        "/v1/transactions/score",
        json=payload,
    )

    assert response.status_code == 422


def test_extra_field_is_rejected(
    realtime_components,
):
    """Simulator-only or unexpected fields must be rejected."""
    pipeline = realtime_components["pipeline"]

    payload = valid_payload()
    payload["TX_FRAUD"] = 1

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    response = client.post(
        "/v1/transactions/score",
        json=payload,
    )

    assert response.status_code == 422


def test_failure_returns_500():
    """Internal real-time failure must fail safely."""
    app = create_app(
        realtime_dependency=lambda: (
            FailingRealtimeService()
        )
    )
    client = TestClient(app)

    response = client.post(
        "/v1/transactions/score",
        json=valid_payload(),
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Real-time fraud processing failed safely."
        )
    }


def test_request_id_is_propagated(
    realtime_components,
):
    """Real-time endpoint must preserve request ID."""
    pipeline = realtime_components["pipeline"]

    app = create_app(
        realtime_dependency=lambda: pipeline
    )
    client = TestClient(app)

    response = client.post(
        "/v1/transactions/score",
        json=valid_payload(),
        headers={
            "X-Request-ID": "cp06-test-request"
        },
    )

    assert response.status_code == 200

    assert (
        response.headers["X-Request-ID"]
        == "cp06-test-request"
    )

    assert (
        "X-Process-Time-Ms"
        in response.headers
    )