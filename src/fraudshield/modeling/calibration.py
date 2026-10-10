"""Probability calibration utilities for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class CalibrationMetrics:
    """Probability-quality metrics."""

    brier_score: float
    log_loss: float
    mean_probability: float
    observed_prevalence: float


@dataclass(frozen=True)
class SigmoidCalibrator:
    """Platt-style sigmoid probability calibrator."""

    model: LogisticRegression

    def predict(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """Transform raw scores into calibrated probabilities."""
        scores = _validate_probabilities(
            probabilities
        )

        calibrated = self.model.predict_proba(
            scores.reshape(-1, 1)
        )[:, 1]

        if not np.isfinite(calibrated).all():
            raise ValueError(
                "Calibrator produced non-finite probabilities."
            )

        return calibrated


def _validate_binary_target(
    y_true: np.ndarray,
) -> np.ndarray:
    target = np.asarray(y_true)

    if target.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    if target.size == 0:
        raise ValueError(
            "Target must not be empty."
        )

    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError(
            "Target must contain both classes."
        )

    return target.astype(int)


def _validate_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(
        probabilities,
        dtype=float,
    )

    if scores.ndim != 1:
        raise ValueError(
            "Probabilities must be one-dimensional."
        )

    if scores.size == 0:
        raise ValueError(
            "Probabilities must not be empty."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Probabilities contain non-finite values."
        )

    if (
        (scores < 0.0)
        | (scores > 1.0)
    ).any():
        raise ValueError(
            "Probabilities must lie in [0, 1]."
        )

    return scores


def fit_sigmoid_calibrator(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> SigmoidCalibrator:
    """Fit a sigmoid calibrator on held-out predictions."""
    target = _validate_binary_target(
        y_true
    )

    scores = _validate_probabilities(
        probabilities
    )

    if len(target) != len(scores):
        raise ValueError(
            "Target and probabilities must have equal length."
        )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        scores.reshape(-1, 1),
        target,
    )

    return SigmoidCalibrator(
        model=model
    )


def evaluate_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> CalibrationMetrics:
    """Evaluate probability calibration quality."""
    target = _validate_binary_target(
        y_true
    )

    scores = _validate_probabilities(
        probabilities
    )

    if len(target) != len(scores):
        raise ValueError(
            "Target and probabilities must have equal length."
        )

    return CalibrationMetrics(
        brier_score=float(
            brier_score_loss(
                target,
                scores,
            )
        ),
        log_loss=float(
            log_loss(
                target,
                scores,
                labels=[0, 1],
            )
        ),
        mean_probability=float(
            scores.mean()
        ),
        observed_prevalence=float(
            target.mean()
        ),
    )