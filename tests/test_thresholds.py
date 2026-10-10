"""Tests for FraudShield threshold optimization."""

import numpy as np
import pandas as pd
import pytest

from fraudshield.modeling.thresholds import (
    BusinessCostAssumptions,
    build_threshold_table,
    business_cost,
    select_cost_optimal_threshold,
)


def make_threshold_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    y_true = np.array(
        [0, 0, 1, 0, 1, 1]
    )

    probabilities = np.array(
        [0.05, 0.20, 0.35, 0.55, 0.70, 0.90]
    )

    amounts = np.array(
        [100.0, 80.0, 200.0, 60.0, 500.0, 300.0]
    )

    return (
        y_true,
        probabilities,
        amounts,
    )


def test_business_cost_matches_manual_calculation() -> None:
    y_true, probabilities, amounts = (
        make_threshold_data()
    )

    assumptions = BusinessCostAssumptions(
        false_positive_review_cost=50.0,
        detected_fraud_review_cost=10.0,
    )

    cost = business_cost(
        y_true,
        probabilities,
        amounts,
        threshold=0.50,
        assumptions=assumptions,
    )

    # At 0.50:
    # FP: score 0.55 -> 50
    # FN: fraud score 0.35, amount 200 -> 200
    # TP: two detected frauds -> 2 * 10
    assert cost == pytest.approx(
        270.0
    )


def test_lower_threshold_can_capture_more_fraud() -> None:
    y_true, probabilities, amounts = (
        make_threshold_data()
    )

    table = build_threshold_table(
        y_true,
        probabilities,
        amounts,
        thresholds=np.array(
            [0.30, 0.80]
        ),
        assumptions=BusinessCostAssumptions(),
    )

    lower = table.iloc[0]
    higher = table.iloc[1]

    assert (
        lower[
            "fraud_transaction_capture_rate"
        ]
        >= higher[
            "fraud_transaction_capture_rate"
        ]
    )


def test_threshold_table_contains_expected_columns() -> None:
    y_true, probabilities, amounts = (
        make_threshold_data()
    )

    table = build_threshold_table(
        y_true,
        probabilities,
        amounts,
        thresholds=np.array(
            [0.2, 0.5, 0.8]
        ),
        assumptions=BusinessCostAssumptions(),
    )

    expected = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "predicted_positive_rate",
        "fraud_transaction_capture_rate",
        "fraud_amount_capture_rate",
        "false_positives",
        "false_negatives",
        "true_positives",
        "business_cost",
    }

    assert expected.issubset(
        table.columns
    )


def test_select_cost_optimal_threshold_uses_minimum_cost() -> None:
    table = pd.DataFrame(
        {
            "threshold": [
                0.10,
                0.20,
                0.30,
            ],
            "business_cost": [
                500.0,
                200.0,
                300.0,
            ],
            "recall": [
                0.9,
                0.8,
                0.7,
            ],
            "false_positive_rate": [
                0.5,
                0.2,
                0.1,
            ],
        }
    )

    selected = (
        select_cost_optimal_threshold(
            table
        )
    )

    assert selected[
        "threshold"
    ] == pytest.approx(
        0.20
    )


def test_threshold_selection_breaks_cost_tie_with_recall() -> None:
    table = pd.DataFrame(
        {
            "threshold": [
                0.10,
                0.20,
            ],
            "business_cost": [
                200.0,
                200.0,
            ],
            "recall": [
                0.90,
                0.80,
            ],
            "false_positive_rate": [
                0.40,
                0.20,
            ],
        }
    )

    selected = (
        select_cost_optimal_threshold(
            table
        )
    )

    assert selected[
        "threshold"
    ] == pytest.approx(
        0.10
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_business_cost_rejects_invalid_threshold(
    threshold: float,
) -> None:
    y_true, probabilities, amounts = (
        make_threshold_data()
    )

    with pytest.raises(
        ValueError,
        match="Threshold",
    ):
        business_cost(
            y_true,
            probabilities,
            amounts,
            threshold=threshold,
            assumptions=BusinessCostAssumptions(),
        )


def test_threshold_table_rejects_empty_grid() -> None:
    y_true, probabilities, amounts = (
        make_threshold_data()
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_threshold_table(
            y_true,
            probabilities,
            amounts,
            thresholds=np.array([]),
            assumptions=BusinessCostAssumptions(),
        )


def test_business_cost_rejects_negative_amount() -> None:
    y_true, probabilities, amounts = (
        make_threshold_data()
    )

    amounts[0] = -1.0

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        business_cost(
            y_true,
            probabilities,
            amounts,
            threshold=0.5,
            assumptions=BusinessCostAssumptions(),
        )