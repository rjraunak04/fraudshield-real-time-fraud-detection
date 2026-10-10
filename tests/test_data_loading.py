"""Tests for FraudShield dataset loading and profiling."""

from pathlib import Path

import pandas as pd
import pytest

from fraudshield.data.load import load_transactions
from fraudshield.data.profile import build_dataset_profile


def make_valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid transaction dataset."""
    return pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 0],
            "TX_DATETIME": [
                "2018-04-01 11:00:00",
                "2018-04-01 10:00:00",
            ],
            "CUSTOMER_ID": [101, 100],
            "TERMINAL_ID": [11, 10],
            "TX_AMOUNT": [250.0, 25.5],
            "TX_TIME_SECONDS": [39600, 36000],
            "TX_TIME_DAYS": [0, 0],
            "TX_FRAUD": [1, 0],
            "TX_FRAUD_SCENARIO": [1, 0],
        }
    )


def test_loader_reads_and_sorts_transactions(
    tmp_path: Path,
) -> None:
    """Loader should return validated chronological transactions."""
    source = tmp_path / "transactions.csv"
    make_valid_dataframe().to_csv(source, index=False)

    result = load_transactions(source)

    assert len(result) == 2
    assert result["TRANSACTION_ID"].tolist() == [0, 1]
    assert pd.api.types.is_datetime64_any_dtype(
        result["TX_DATETIME"]
    )


def test_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Loader should fail clearly when the dataset does not exist."""
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_transactions(missing)


def test_loader_enforces_data_contract(
    tmp_path: Path,
) -> None:
    """Loader should reject datasets violating the data contract."""
    source = tmp_path / "invalid.csv"

    invalid = make_valid_dataframe().drop(
        columns=["TX_AMOUNT"]
    )
    invalid.to_csv(source, index=False)

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        load_transactions(source)


def test_profile_reports_core_statistics() -> None:
    """Profiler should calculate expected dataset statistics."""
    transactions = make_valid_dataframe()

    transactions["TX_DATETIME"] = pd.to_datetime(
        transactions["TX_DATETIME"]
    )

    profile = build_dataset_profile(transactions)

    assert profile["n_transactions"] == 2
    assert profile["n_columns"] == 9
    assert profile["n_customers"] == 2
    assert profile["n_terminals"] == 2
    assert profile["fraud_count"] == 1
    assert profile["legitimate_count"] == 1
    assert profile["fraud_rate"] == pytest.approx(0.5)
    assert profile["missing_values"] == 0
    assert profile["duplicate_transaction_ids"] == 0


def test_profile_rejects_empty_dataframe() -> None:
    """Profiler should reject an empty dataset."""
    with pytest.raises(
        ValueError,
        match="empty transaction dataset",
    ):
        build_dataset_profile(pd.DataFrame())