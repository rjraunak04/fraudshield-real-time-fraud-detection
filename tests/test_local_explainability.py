"""Tests for transaction-level FraudShield explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudshield.explainability.local import (
    build_reference_values,
    explain_transaction,
)
from fraudshield.modeling.dataset import MODEL_FEATURE_COLUMNS


class DeterministicProbabilityModel:
    """Simple probability model for explanation tests."""

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        first = X[
            MODEL_FEATURE_COLUMNS[0]
        ].to_numpy(dtype=float)

        second = X[
            MODEL_FEATURE_COLUMNS[1]
        ].to_numpy(dtype=float)

        logits = (
            1.5 * first
            - 0.5 * second
        )

        positive = 1.0 / (
            1.0 + np.exp(-logits)
        )

        return np.column_stack(
            [
                1.0 - positive,
                positive,
            ]
        )


def make_row(
    value: float = 0.0,
) -> pd.Series:
    """Create one valid frozen-schema row."""
    return pd.Series(
        {
            feature: value
            for feature in MODEL_FEATURE_COLUMNS
        },
        dtype=float,
    )


def make_reference_frame() -> pd.DataFrame:
    """Create deterministic reference data."""
    rows = []

    for value in (-1.0, 0.0, 1.0):
        rows.append(
            {
                feature: value
                for feature in MODEL_FEATURE_COLUMNS
            }
        )

    return pd.DataFrame(
        rows,
        columns=MODEL_FEATURE_COLUMNS,
    )


def test_build_reference_values_uses_median() -> None:
    X = make_reference_frame()

    reference = build_reference_values(X)

    assert list(reference.index) == list(
        MODEL_FEATURE_COLUMNS
    )

    assert np.allclose(
        reference.to_numpy(),
        0.0,
    )


def test_explanation_returns_requested_top_n() -> None:
    model = DeterministicProbabilityModel()
    row = make_row()
    reference = make_row()

    row.loc[
        MODEL_FEATURE_COLUMNS[0]
    ] = 2.0

    result = explain_transaction(
        model,
        row,
        reference,
        top_n=5,
    )

    assert len(result.contributions) == 5
    assert [
        contribution.rank
        for contribution
        in result.contributions
    ] == [1, 2, 3, 4, 5]


def test_primary_signal_increases_risk() -> None:
    model = DeterministicProbabilityModel()
    row = make_row()
    reference = make_row()

    signal_feature = MODEL_FEATURE_COLUMNS[0]
    row.loc[signal_feature] = 2.0

    result = explain_transaction(
        model,
        row,
        reference,
        top_n=1,
    )

    contribution = result.contributions[0]

    assert contribution.feature == signal_feature
    assert contribution.probability_change > 0.0
    assert (
        contribution.direction
        == "increases_risk"
    )


def test_negative_signal_decreases_risk() -> None:
    model = DeterministicProbabilityModel()
    row = make_row()
    reference = make_row()

    signal_feature = MODEL_FEATURE_COLUMNS[0]
    row.loc[signal_feature] = -2.0

    result = explain_transaction(
        model,
        row,
        reference,
        top_n=1,
    )

    contribution = result.contributions[0]

    assert contribution.feature == signal_feature
    assert contribution.probability_change < 0.0
    assert (
        contribution.direction
        == "decreases_risk"
    )


def test_explanation_is_deterministic() -> None:
    model = DeterministicProbabilityModel()
    row = make_row()
    reference = make_row()

    row.loc[
        MODEL_FEATURE_COLUMNS[0]
    ] = 1.5

    first = explain_transaction(
        model,
        row,
        reference,
        top_n=5,
    )
    second = explain_transaction(
        model,
        row,
        reference,
        top_n=5,
    )

    assert first == second


def test_probability_is_valid() -> None:
    model = DeterministicProbabilityModel()

    result = explain_transaction(
        model,
        make_row(),
        make_row(),
    )

    assert (
        0.0
        <= result.predicted_probability
        <= 1.0
    )


def test_dict_inputs_are_supported() -> None:
    model = DeterministicProbabilityModel()

    row = make_row().to_dict()
    reference = make_row().to_dict()

    result = explain_transaction(
        model,
        row,
        reference,
        top_n=3,
    )

    assert len(result.contributions) == 3


def test_rejects_missing_transaction_feature() -> None:
    model = DeterministicProbabilityModel()

    row = make_row().drop(
        MODEL_FEATURE_COLUMNS[-1]
    )

    with pytest.raises(
        ValueError,
        match="frozen model feature schema",
    ):
        explain_transaction(
            model,
            row,
            make_row(),
        )


def test_rejects_extra_transaction_feature() -> None:
    model = DeterministicProbabilityModel()

    row = make_row()
    row.loc["UNKNOWN_FEATURE"] = 1.0

    with pytest.raises(
        ValueError,
        match="frozen model feature schema",
    ):
        explain_transaction(
            model,
            row,
            make_row(),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rejects_invalid_transaction_values(
    invalid_value: float,
) -> None:
    model = DeterministicProbabilityModel()

    row = make_row()
    row.loc[
        MODEL_FEATURE_COLUMNS[0]
    ] = invalid_value

    with pytest.raises(ValueError):
        explain_transaction(
            model,
            row,
            make_row(),
        )


def test_rejects_invalid_top_n() -> None:
    model = DeterministicProbabilityModel()

    with pytest.raises(
        ValueError,
        match="top_n",
    ):
        explain_transaction(
            model,
            make_row(),
            make_row(),
            top_n=0,
        )


def test_top_n_cannot_exceed_feature_count() -> None:
    model = DeterministicProbabilityModel()

    result = explain_transaction(
        model,
        make_row(),
        make_row(),
        top_n=100,
    )

    assert len(result.contributions) == len(
        MODEL_FEATURE_COLUMNS
    )