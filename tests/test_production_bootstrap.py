"""Tests for production service composition."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from fraudshield.production.bootstrap import ProductionContainer
from fraudshield.production.settings import ProductionSettings
from fraudshield.scoring.service import FraudScorer


class FixedModel:
    """Minimal deterministic probability model."""

    def predict_proba(self, frame):
        return np.array([[0.94, 0.06]], dtype=float)


class IdentityCalibrator:
    """Return probability unchanged."""

    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=float)


def test_production_container_readiness_checks_database_and_redis():
    engine = MagicMock()
    connection = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    redis_client = MagicMock()

    container = ProductionContainer(
        settings=ProductionSettings(
            environment="test",
            database_url="sqlite://",
            redis_url="redis://localhost",
            model_artifact_path=Path("model.pkl"),
            calibrator_artifact_path=Path("calibrator.pkl"),
        ),
        database_engine=engine,
        session_factory=MagicMock(),
        redis_client=redis_client,
        scorer=FraudScorer(
            model=FixedModel(),
            calibrator=IdentityCalibrator(),
        ),
        decision_engine=MagicMock(),
        monitoring=MagicMock(),
        prometheus=MagicMock(),
    )

    container.readiness_check()

    engine.connect.assert_called_once_with()
    connection.execute.assert_called_once()
    redis_client.ping.assert_called_once_with()


def test_production_container_close_releases_dependencies():
    engine = MagicMock()
    redis_client = MagicMock()

    container = ProductionContainer(
        settings=ProductionSettings(
            environment="test",
            database_url="sqlite://",
            redis_url="redis://localhost",
            model_artifact_path=Path("model.pkl"),
            calibrator_artifact_path=Path("calibrator.pkl"),
        ),
        database_engine=engine,
        session_factory=MagicMock(),
        redis_client=redis_client,
        scorer=FraudScorer(
            model=FixedModel(),
            calibrator=IdentityCalibrator(),
        ),
        decision_engine=MagicMock(),
        monitoring=MagicMock(),
        prometheus=MagicMock(),
    )

    container.close()

    redis_client.close.assert_called_once_with()
    engine.dispose.assert_called_once_with()
