"""Distribution-drift monitoring for FraudShield."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

DEFAULT_WARNING_THRESHOLD = 0.10
DEFAULT_DRIFT_THRESHOLD = 0.25
DEFAULT_BINS = 10
_EPSILON = 1e-6


class DriftStatus(StrEnum):
    """Operational drift severity."""

    STABLE = "STABLE"
    WARNING = "WARNING"
    DRIFT = "DRIFT"


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Distribution drift result for one variable."""

    name: str
    psi: float
    status: DriftStatus
    reference_count: int
    current_count: int


def classify_psi(
    psi: float,
    *,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> DriftStatus:
    """Convert a PSI value into an operational status."""
    if not np.isfinite(psi):
        raise ValueError("PSI must be finite.")

    if psi < 0:
        raise ValueError("PSI must be non-negative.")

    if warning_threshold < 0:
        raise ValueError(
            "warning_threshold must be non-negative."
        )

    if drift_threshold <= warning_threshold:
        raise ValueError(
            "drift_threshold must be greater than "
            "warning_threshold."
        )

    if psi >= drift_threshold:
        return DriftStatus.DRIFT

    if psi >= warning_threshold:
        return DriftStatus.WARNING

    return DriftStatus.STABLE


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
) -> float:
    """Calculate PSI using reference-derived quantile bins."""
    reference_values = _validated_values(
        reference,
        name="reference",
    )
    current_values = _validated_values(
        current,
        name="current",
    )

    if bins < 2:
        raise ValueError("bins must be at least 2.")

    edges = _reference_bin_edges(
        reference_values,
        bins=bins,
    )

    reference_counts, _ = np.histogram(
        reference_values,
        bins=edges,
    )
    current_counts, _ = np.histogram(
        current_values,
        bins=edges,
    )

    reference_distribution = (
        reference_counts.astype(float)
        / reference_values.size
    )
    current_distribution = (
        current_counts.astype(float)
        / current_values.size
    )

    reference_distribution = np.clip(
        reference_distribution,
        _EPSILON,
        None,
    )
    current_distribution = np.clip(
        current_distribution,
        _EPSILON,
        None,
    )

    psi_components = (
        current_distribution - reference_distribution
    ) * np.log(
        current_distribution / reference_distribution
    )

    return float(np.sum(psi_components))


def evaluate_drift(
    *,
    name: str,
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = DEFAULT_BINS,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> DriftResult:
    """Evaluate PSI and operational drift status."""
    reference_values = _validated_values(
        reference,
        name="reference",
    )
    current_values = _validated_values(
        current,
        name="current",
    )

    psi = population_stability_index(
        reference_values,
        current_values,
        bins=bins,
    )

    status = classify_psi(
        psi,
        warning_threshold=warning_threshold,
        drift_threshold=drift_threshold,
    )

    return DriftResult(
        name=name,
        psi=psi,
        status=status,
        reference_count=int(
            reference_values.size
        ),
        current_count=int(
            current_values.size
        ),
    )


def _validated_values(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return validated one-dimensional finite values."""
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def _reference_bin_edges(
    reference: np.ndarray,
    *,
    bins: int,
) -> np.ndarray:
    """Build stable quantile bins from reference data."""
    quantiles = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    edges = np.quantile(
        reference,
        quantiles,
    )

    edges = np.unique(edges)

    if edges.size < 2:
        value = float(reference[0])

        delta = max(
            abs(value) * 0.01,
            1.0,
        )

        edges = np.asarray(
            [
                value - delta,
                value + delta,
            ],
            dtype=float,
        )
    else:
        edges = edges.astype(
            float,
            copy=True,
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges