"""Tests for the FraudShield production scoring contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS
from fraudshield.scoring.service import (
    FROZEN_THRESHOLD,
    FraudScorer,
)


class StubModel:
    """Deterministic probability model for scorer tests."""

    def __init__(self, fraud_probability: float) -> None:
        self.fraud_probability = fraud_probability
        self.last_columns: list[str] | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.last_columns = list(X.columns)

        return np.array(
            [
                [
                    1.0 - self.fraud_probability,
                    self.fraud_probability,
                ]
            ]
        )


class StubCalibrator:
    """Deterministic calibrator for scorer tests."""

    def __init__(self, calibrated_probability: float) -> None:
        self.calibrated_probability = calibrated_probability

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        assert probabilities.shape == (1,)
        return np.array([self.calibrated_probability])


def valid_features() -> dict[str, float]:
    """Create one complete model-feature record."""
    return {
        column: float(index + 1)
        for index, column in enumerate(MODEL_FEATURE_COLUMNS)
    }


def test_score_one_returns_stable_contract() -> None:
    scorer = FraudScorer(
        model=StubModel(fraud_probability=0.70),
        calibrator=StubCalibrator(calibrated_probability=0.08),
    )

    result = scorer.score_one(valid_features())

    assert result.raw_probability == pytest.approx(0.70)
    assert result.calibrated_probability == pytest.approx(0.08)
    assert result.threshold == pytest.approx(FROZEN_THRESHOLD)
    assert result.is_high_risk is True
    assert result.model_version
    assert result.calibration_method == "sigmoid"


def test_score_one_uses_calibrated_probability_for_flag() -> None:
    scorer = FraudScorer(
        model=StubModel(fraud_probability=0.90),
        calibrator=StubCalibrator(calibrated_probability=0.02),
    )

    result = scorer.score_one(valid_features())

    assert result.raw_probability == pytest.approx(0.90)
    assert result.calibrated_probability == pytest.approx(0.02)
    assert result.is_high_risk is False


def test_score_one_enforces_feature_order() -> None:
    model = StubModel(fraud_probability=0.50)

    scorer = FraudScorer(
        model=model,
        calibrator=StubCalibrator(calibrated_probability=0.04),
    )

    reversed_features = dict(
        reversed(list(valid_features().items()))
    )

    scorer.score_one(reversed_features)

    assert model.last_columns == list(MODEL_FEATURE_COLUMNS)


def test_score_one_accepts_series() -> None:
    scorer = FraudScorer(
        model=StubModel(fraud_probability=0.50),
        calibrator=StubCalibrator(calibrated_probability=0.05),
    )

    result = scorer.score_one(pd.Series(valid_features()))

    assert result.is_high_risk is True


def test_score_one_rejects_missing_feature() -> None:
    features = valid_features()
    features.pop(MODEL_FEATURE_COLUMNS[0])

    scorer = FraudScorer(
        model=StubModel(0.5),
        calibrator=StubCalibrator(0.05),
    )

    with pytest.raises(
        ValueError,
        match="Missing model features",
    ):
        scorer.score_one(features)


def test_score_one_rejects_extra_feature() -> None:
    features = valid_features()
    features["LEAKAGE_COLUMN"] = 1.0

    scorer = FraudScorer(
        model=StubModel(0.5),
        calibrator=StubCalibrator(0.05),
    )

    with pytest.raises(
        ValueError,
        match="Unexpected model features",
    ):
        scorer.score_one(features)


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
        "not-a-number",
    ],
)
def test_score_one_rejects_non_finite_features(
    bad_value: object,
) -> None:
    features: dict[str, object] = valid_features()
    features[MODEL_FEATURE_COLUMNS[0]] = bad_value

    scorer = FraudScorer(
        model=StubModel(0.5),
        calibrator=StubCalibrator(0.05),
    )

    with pytest.raises(
        ValueError,
        match="finite numeric",
    ):
        scorer.score_one(features)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "threshold",
    [
        -0.01,
        1.01,
    ],
)
def test_scorer_rejects_invalid_threshold(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Threshold",
    ):
        FraudScorer(
            model=StubModel(0.5),
            calibrator=StubCalibrator(0.05),
            threshold=threshold,
        )


def test_score_one_rejects_invalid_model_probability() -> None:
    scorer = FraudScorer(
        model=StubModel(fraud_probability=1.20),
        calibrator=StubCalibrator(
            calibrated_probability=0.05
        ),
    )

    with pytest.raises(
        ValueError,
        match="Raw probability",
    ):
        scorer.score_one(valid_features())


def test_score_one_rejects_invalid_calibrated_probability() -> None:
    scorer = FraudScorer(
        model=StubModel(fraud_probability=0.50),
        calibrator=StubCalibrator(
            calibrated_probability=1.20
        ),
    )

    with pytest.raises(
        ValueError,
        match="Calibrated probability",
    ):
        scorer.score_one(valid_features())