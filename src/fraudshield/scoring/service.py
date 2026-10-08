"""Production scoring contract for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS

FROZEN_MODEL_VERSION = "day-03-hgb-balanced-v1"
FROZEN_CALIBRATION_METHOD = "sigmoid"
FROZEN_THRESHOLD = 0.048


class ProbabilityModel(Protocol):
    """Minimal model interface required by the scorer."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""


class ProbabilityCalibrator(Protocol):
    """Minimal calibration interface required by the scorer."""

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated fraud probabilities."""


@dataclass(frozen=True)
class FraudScore:
    """Stable output contract for one scored transaction."""

    raw_probability: float
    calibrated_probability: float
    threshold: float
    is_high_risk: bool
    model_version: str
    calibration_method: str


class FraudScorer:
    """Score transactions using a frozen model and calibrator."""

    def __init__(
        self,
        *,
        model: ProbabilityModel,
        calibrator: ProbabilityCalibrator,
        threshold: float = FROZEN_THRESHOLD,
        model_version: str = FROZEN_MODEL_VERSION,
        calibration_method: str = FROZEN_CALIBRATION_METHOD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must lie in [0, 1].")

        if not model_version.strip():
            raise ValueError("Model version must not be empty.")

        if not calibration_method.strip():
            raise ValueError("Calibration method must not be empty.")

        self._model = model
        self._calibrator = calibrator
        self._threshold = float(threshold)
        self._model_version = model_version
        self._calibration_method = calibration_method

    @property
    def threshold(self) -> float:
        """Return the frozen operating threshold."""
        return self._threshold

    @property
    def model_version(self) -> str:
        """Return the scorer model version."""
        return self._model_version

    def score_one(
        self,
        features: pd.Series | dict[str, float | int],
    ) -> FraudScore:
        """Score exactly one transaction."""
        frame = self._prepare_single_row(features)

        raw_probability = self._predict_raw_probability(frame)

        calibrated = self._calibrator.predict(
            np.array([raw_probability], dtype=float)
        )
        calibrated_array = np.asarray(calibrated, dtype=float)

        if calibrated_array.shape != (1,):
            raise ValueError(
                "Calibrator must return exactly one probability "
                "for one transaction."
            )

        calibrated_probability = float(calibrated_array[0])

        self._validate_probability(
            calibrated_probability,
            name="Calibrated probability",
        )

        return FraudScore(
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            threshold=self._threshold,
            is_high_risk=calibrated_probability >= self._threshold,
            model_version=self._model_version,
            calibration_method=self._calibration_method,
        )

    def _prepare_single_row(
        self,
        features: pd.Series | dict[str, float | int],
    ) -> pd.DataFrame:
        if isinstance(features, pd.Series):
            supplied = features.to_dict()
        elif isinstance(features, dict):
            supplied = features.copy()
        else:
            raise TypeError(
                "features must be a pandas Series or dictionary."
            )

        required = set(MODEL_FEATURE_COLUMNS)
        supplied_columns = set(supplied)

        missing = required - supplied_columns
        extra = supplied_columns - required

        if missing:
            raise ValueError(
                f"Missing model features: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unexpected model features: {sorted(extra)}"
            )

        frame = pd.DataFrame(
            [
                {
                    column: supplied[column]
                    for column in MODEL_FEATURE_COLUMNS
                }
            ]
        )

        numeric = frame.apply(pd.to_numeric, errors="coerce")

        if numeric.isna().any().any():
            raise ValueError(
                "Model features must contain finite numeric values."
            )

        values = numeric.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Model features must contain finite numeric values."
            )

        return numeric

    def _predict_raw_probability(
        self,
        frame: pd.DataFrame,
    ) -> float:
        probabilities = np.asarray(
            self._model.predict_proba(frame),
            dtype=float,
        )

        if probabilities.ndim != 2 or probabilities.shape != (1, 2):
            raise ValueError(
                "Model predict_proba must return shape (1, 2) "
                "for one transaction."
            )

        raw_probability = float(probabilities[0, 1])

        self._validate_probability(
            raw_probability,
            name="Raw probability",
        )

        return raw_probability

    @staticmethod
    def _validate_probability(
        probability: float,
        *,
        name: str,
    ) -> None:
        if not np.isfinite(probability):
            raise ValueError(f"{name} must be finite.")

        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")