"""Dataset loading utilities for FraudShield."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraudshield.data.validate import validate_transactions


def load_transactions(
    path: str | Path,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load a FraudShield transaction dataset from CSV.

    Parameters
    ----------
    path:
        Location of the transaction CSV.
    validate:
        Whether to enforce the FraudShield data contract after loading.

    Returns
    -------
    pandas.DataFrame
        Transactions ordered chronologically.

    Raises
    ------
    FileNotFoundError
        If the requested dataset does not exist.
    ValueError
        If the dataset violates the FraudShield data contract.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {dataset_path}"
        )

    transactions = pd.read_csv(
        dataset_path,
        parse_dates=["TX_DATETIME"],
    )

    if validate:
        validate_transactions(transactions)

    return (
        transactions.sort_values(
            ["TX_DATETIME", "TRANSACTION_ID"]
        )
        .reset_index(drop=True)
    )