"""Tests for the complete FraudShield feature pipeline."""

import pandas as pd
import pytest

from fraudshield.features.build import (
    ENGINEERED_FEATURE_COLUMNS,
    MODEL_EXCLUDED_COLUMNS,
    build_features,
    get_model_feature_columns,
    validate_feature_dataset,
)


def make_transactions() -> pd.DataFrame:
    """Create deterministic transactions for pipeline tests."""
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 2, 3, 4],
            "TX_DATETIME": pd.to_datetime(
                [
                    "2018-04-01 09:00:00",
                    "2018-04-01 10:00:00",
                    "2018-04-01 11:00:00",
                    "2018-04-02 12:00:00",
                ]
            ),
            "CUSTOMER_ID": [
                100,
                100,
                100,
                200,
            ],
            "TERMINAL_ID": [
                10,
                10,
                20,
                10,
            ],
            "TX_AMOUNT": [
                10.0,
                20.0,
                60.0,
                40.0,
            ],
            "TX_TIME_SECONDS": [
                0,
                3600,
                7200,
                97200,
            ],
            "TX_TIME_DAYS": [
                0,
                0,
                0,
                1,
            ],
            "TX_FRAUD": [
                0,
                0,
                1,
                0,
            ],
            "TX_FRAUD_SCENARIO": [
                0,
                0,
                2,
                0,
            ],
        }
    )


def test_pipeline_creates_all_engineered_features() -> None:
    """Unified pipeline should create every configured feature."""
    transactions = make_transactions()

    result = build_features(transactions)

    for column in ENGINEERED_FEATURE_COLUMNS:
        assert column in result.columns


def test_pipeline_preserves_rows() -> None:
    """Feature pipeline must not create or remove transactions."""
    transactions = make_transactions()

    result = build_features(transactions)

    assert len(result) == len(transactions)


def test_pipeline_preserves_transaction_order() -> None:
    """Pipeline should preserve input transaction ordering."""
    transactions = make_transactions().iloc[
        [3, 1, 0, 2]
    ].reset_index(drop=True)

    expected = transactions[
        "TRANSACTION_ID"
    ].tolist()

    result = build_features(transactions)

    assert result["TRANSACTION_ID"].tolist() == expected


def test_pipeline_does_not_modify_original_dataframe() -> None:
    """Unified feature engineering must not mutate its input."""
    transactions = make_transactions()

    original = transactions.copy(deep=True)

    build_features(transactions)

    pd.testing.assert_frame_equal(
        transactions,
        original,
    )


def test_pipeline_engineered_features_have_no_missing_values() -> None:
    """Final engineered features should be fully populated."""
    result = build_features(
        make_transactions()
    )

    assert (
        result[
            ENGINEERED_FEATURE_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
        == 0
    )


def test_target_is_not_a_model_feature() -> None:
    """Fraud target must never enter the model feature matrix."""
    result = build_features(
        make_transactions()
    )

    model_columns = get_model_feature_columns(
        result
    )

    assert "TX_FRAUD" not in model_columns


def test_scenario_is_not_a_model_feature() -> None:
    """Simulator fraud scenario must never enter model inputs."""
    result = build_features(
        make_transactions()
    )

    model_columns = get_model_feature_columns(
        result
    )

    assert "TX_FRAUD_SCENARIO" not in model_columns


def test_metadata_is_not_used_as_model_features() -> None:
    """Identifiers and timestamps should stay outside model input."""
    result = build_features(
        make_transactions()
    )

    model_columns = get_model_feature_columns(
        result
    )

    for column in MODEL_EXCLUDED_COLUMNS:
        assert column not in model_columns


def test_pipeline_can_build_features_without_target_columns() -> None:
    """Feature generation itself should not require fraud labels."""
    transactions = make_transactions().drop(
        columns=[
            "TX_FRAUD",
            "TX_FRAUD_SCENARIO",
        ]
    )

    result = build_features(transactions)

    for column in ENGINEERED_FEATURE_COLUMNS:
        assert column in result.columns


def test_feature_validation_accepts_valid_dataset() -> None:
    """Valid final feature datasets should pass QA."""
    transactions = make_transactions()

    result = build_features(transactions)

    validate_feature_dataset(
        transactions,
        result,
    )


def test_feature_validation_detects_missing_feature() -> None:
    """QA should fail if an engineered feature disappears."""
    transactions = make_transactions()

    result = build_features(
        transactions
    ).drop(
        columns=[
            ENGINEERED_FEATURE_COLUMNS[0]
        ]
    )

    with pytest.raises(
        ValueError,
        match="Missing engineered features",
    ):
        validate_feature_dataset(
            transactions,
            result,
        )


def test_feature_validation_detects_row_loss() -> None:
    """QA should detect dropped transactions."""
    transactions = make_transactions()

    result = build_features(
        transactions
    ).iloc[:-1]

    with pytest.raises(
        ValueError,
        match="changed the number of rows",
    ):
        validate_feature_dataset(
            transactions,
            result,
        )