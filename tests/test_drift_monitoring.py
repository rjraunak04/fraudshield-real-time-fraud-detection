"""Tests for FraudShield distribution-drift monitoring."""

from __future__ import annotations

import numpy as np
import pytest

from fraudshield.monitoring import (
    DriftStatus,
    classify_psi,
    evaluate_drift,
    population_stability_index,
)


def test_identical_distribution_has_zero_psi() -> None:
    reference = np.arange(
        1,
        101,
        dtype=float,
    )

    psi = population_stability_index(
        reference,
        reference.copy(),
    )

    assert psi == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_similar_distribution_is_stable() -> None:
    reference = np.arange(
        1,
        1001,
        dtype=float,
    )

    current = reference + 0.01

    result = evaluate_drift(
        name="TX_AMOUNT",
        reference=reference,
        current=current,
    )

    assert result.status is DriftStatus.STABLE
    assert result.reference_count == 1000
    assert result.current_count == 1000


def test_large_shift_detects_drift() -> None:
    reference = np.arange(
        0,
        1000,
        dtype=float,
    )

    current = np.arange(
        1000,
        2000,
        dtype=float,
    )

    result = evaluate_drift(
        name="TX_AMOUNT",
        reference=reference,
        current=current,
    )

    assert result.psi >= 0.25
    assert result.status is DriftStatus.DRIFT


@pytest.mark.parametrize(
    ("psi", "expected"),
    [
        (0.0, DriftStatus.STABLE),
        (0.099, DriftStatus.STABLE),
        (0.10, DriftStatus.WARNING),
        (0.249, DriftStatus.WARNING),
        (0.25, DriftStatus.DRIFT),
        (1.0, DriftStatus.DRIFT),
    ],
)
def test_psi_classification_boundaries(
    psi: float,
    expected: DriftStatus,
) -> None:
    assert classify_psi(psi) is expected


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([]),
        np.asarray([1.0, np.nan]),
        np.asarray([1.0, np.inf]),
    ],
)
def test_invalid_reference_rejected(
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        population_stability_index(
            values,
            np.asarray([1.0, 2.0]),
        )


def test_invalid_current_rejected() -> None:
    with pytest.raises(ValueError):
        population_stability_index(
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, np.nan]),
        )


def test_multidimensional_input_rejected() -> None:
    with pytest.raises(ValueError):
        population_stability_index(
            np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            np.asarray([1.0, 2.0]),
        )


def test_invalid_bin_count_rejected() -> None:
    with pytest.raises(ValueError):
        population_stability_index(
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 2.0]),
            bins=1,
        )


def test_constant_reference_is_supported() -> None:
    reference = np.ones(
        100,
        dtype=float,
    )
    current = np.ones(
        100,
        dtype=float,
    )

    psi = population_stability_index(
        reference,
        current,
    )

    assert psi == pytest.approx(0.0)


def test_custom_thresholds_are_supported() -> None:
    assert (
        classify_psi(
            0.15,
            warning_threshold=0.05,
            drift_threshold=0.20,
        )
        is DriftStatus.WARNING
    )


@pytest.mark.parametrize(
    (
        "psi",
        "warning_threshold",
        "drift_threshold",
    ),
    [
        (-0.1, 0.1, 0.25),
        (0.1, -0.1, 0.25),
        (0.1, 0.25, 0.25),
        (0.1, 0.30, 0.25),
    ],
)
def test_invalid_classification_configuration(
    psi: float,
    warning_threshold: float,
    drift_threshold: float,
) -> None:
    with pytest.raises(ValueError):
        classify_psi(
            psi,
            warning_threshold=warning_threshold,
            drift_threshold=drift_threshold,
        )


def test_prediction_probability_drift() -> None:
    reference = np.linspace(
        0.001,
        0.10,
        1000,
    )

    current = np.linspace(
        0.30,
        0.80,
        1000,
    )

    result = evaluate_drift(
        name="calibrated_probability",
        reference=reference,
        current=current,
    )

    assert result.status is DriftStatus.DRIFT


def test_result_preserves_variable_name() -> None:
    result = evaluate_drift(
        name="CUSTOMER_TX_COUNT_24H",
        reference=np.arange(
            100,
            dtype=float,
        ),
        current=np.arange(
            100,
            dtype=float,
        ),
    )

    assert (
        result.name
        == "CUSTOMER_TX_COUNT_24H"
    )