"""Tests for FraudShield probability calibration."""

import numpy as np
import pytest

from fraudshield.modeling.calibration import (
    evaluate_calibration,
    fit_sigmoid_calibrator,
)


def make_calibration_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Create deterministic synthetic calibration data."""
    y_true = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.05,
            0.10,
            0.20,
            0.25,
            0.30,
            0.40,
            0.55,
            0.60,
            0.70,
            0.75,
            0.80,
            0.90,
        ]
    )

    return y_true, probabilities


def test_sigmoid_calibrator_returns_valid_probabilities() -> None:
    y_true, probabilities = (
        make_calibration_data()
    )

    calibrator = fit_sigmoid_calibrator(
        y_true,
        probabilities,
    )

    calibrated = calibrator.predict(
        probabilities
    )

    assert len(calibrated) == len(
        probabilities
    )
    assert np.isfinite(calibrated).all()
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)


def test_sigmoid_calibrator_is_monotonic() -> None:
    y_true, probabilities = (
        make_calibration_data()
    )

    calibrator = fit_sigmoid_calibrator(
        y_true,
        probabilities,
    )

    calibrated = calibrator.predict(
        probabilities
    )

    assert np.all(
        np.diff(calibrated) >= 0.0
    )


def test_calibration_metrics_are_finite() -> None:
    y_true, probabilities = (
        make_calibration_data()
    )

    metrics = evaluate_calibration(
        y_true,
        probabilities,
    )

    assert np.isfinite(
        metrics.brier_score
    )
    assert np.isfinite(
        metrics.log_loss
    )
    assert np.isfinite(
        metrics.mean_probability
    )
    assert np.isfinite(
        metrics.observed_prevalence
    )


def test_observed_prevalence_matches_target() -> None:
    y_true, probabilities = (
        make_calibration_data()
    )

    metrics = evaluate_calibration(
        y_true,
        probabilities,
    )

    assert (
        metrics.observed_prevalence
        == pytest.approx(
            y_true.mean()
        )
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([]),
        np.array([-0.1, 0.5]),
        np.array([0.5, 1.1]),
        np.array([0.5, np.nan]),
    ],
)
def test_calibration_rejects_invalid_probabilities(
    probabilities: np.ndarray,
) -> None:
    y_true = np.array(
        [0, 1]
    )

    with pytest.raises(ValueError):
        evaluate_calibration(
            y_true,
            probabilities,
        )


def test_calibration_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        evaluate_calibration(
            np.array([0, 1]),
            np.array(
                [0.1, 0.5, 0.9]
            ),
        )


def test_calibrator_rejects_single_class_target() -> None:
    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        fit_sigmoid_calibrator(
            np.array(
                [0, 0, 0]
            ),
            np.array(
                [0.1, 0.2, 0.3]
            ),
        )