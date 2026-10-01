"""Synthetic transaction data generation for FraudShield.

This module creates a reproducible payment-transaction dataset inspired by
the Fraud Detection Handbook transaction simulator.

The generated data is intended for learning, experimentation, and portfolio
demonstration. It does not represent the full complexity or economics of a
real payment network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_customer_profiles(
    n_customers: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible synthetic customer behavioural profiles."""
    if n_customers <= 0:
        raise ValueError("n_customers must be greater than zero.")

    rng = np.random.default_rng(random_seed)

    customer_profiles = pd.DataFrame(
        {
            "CUSTOMER_ID": np.arange(n_customers, dtype=int),
            "x_customer_id": rng.uniform(0, 100, size=n_customers),
            "y_customer_id": rng.uniform(0, 100, size=n_customers),
            "mean_amount": rng.uniform(5, 100, size=n_customers),
            "mean_nb_tx_per_day": rng.uniform(0, 4, size=n_customers),
        }
    )

    return customer_profiles.set_index("CUSTOMER_ID")


def generate_terminal_profiles(
    n_terminals: int,
    random_seed: int = 43,
) -> pd.DataFrame:
    """Generate reproducible synthetic payment-terminal profiles."""
    if n_terminals <= 0:
        raise ValueError("n_terminals must be greater than zero.")

    rng = np.random.default_rng(random_seed)

    terminal_profiles = pd.DataFrame(
        {
            "TERMINAL_ID": np.arange(n_terminals, dtype=int),
            "x_terminal_id": rng.uniform(0, 100, size=n_terminals),
            "y_terminal_id": rng.uniform(0, 100, size=n_terminals),
        }
    )

    return terminal_profiles.set_index("TERMINAL_ID")


def associate_customer_terminals(
    customer_profiles: pd.DataFrame,
    terminal_profiles: pd.DataFrame,
    radius: float = 5.0,
) -> dict[int, list[int]]:
    """Associate customers with terminals inside a distance radius."""
    if radius <= 0:
        raise ValueError("radius must be greater than zero.")

    associations: dict[int, list[int]] = {}

    terminal_x = terminal_profiles["x_terminal_id"].to_numpy()
    terminal_y = terminal_profiles["y_terminal_id"].to_numpy()

    for customer_id, customer in customer_profiles.iterrows():
        distances = np.sqrt(
            (terminal_x - customer["x_customer_id"]) ** 2
            + (terminal_y - customer["y_customer_id"]) ** 2
        )

        nearby_mask = distances <= radius

        associations[int(customer_id)] = (
            terminal_profiles.index[nearby_mask].astype(int).tolist()
        )

    return associations


def generate_transactions(
    customer_profiles: pd.DataFrame,
    customer_terminals: dict[int, list[int]],
    start_date: str,
    n_days: int,
    random_seed: int = 44,
) -> pd.DataFrame:
    """Generate normal transactions from customer behavioural profiles."""
    if n_days <= 0:
        raise ValueError("n_days must be greater than zero.")

    rng = np.random.default_rng(random_seed)
    start_timestamp = pd.Timestamp(start_date)

    transactions: list[dict[str, object]] = []

    for customer_id, customer in customer_profiles.iterrows():
        available_terminals = customer_terminals[int(customer_id)]

        if not available_terminals:
            continue

        for day in range(n_days):
            n_transactions = rng.poisson(
                customer["mean_nb_tx_per_day"]
            )

            for _ in range(n_transactions):
                seconds_into_day = int(
                    rng.integers(0, 24 * 60 * 60)
                )

                tx_datetime = (
                    start_timestamp
                    + pd.Timedelta(days=day)
                    + pd.Timedelta(seconds=seconds_into_day)
                )

                amount = rng.normal(
                    loc=customer["mean_amount"],
                    scale=max(customer["mean_amount"] / 2, 1.0),
                )

                amount = round(
                    max(float(amount), 0.01),
                    2,
                )

                terminal_id = int(
                    rng.choice(available_terminals)
                )

                transactions.append(
                    {
                        "TX_DATETIME": tx_datetime,
                        "CUSTOMER_ID": int(customer_id),
                        "TERMINAL_ID": terminal_id,
                        "TX_AMOUNT": amount,
                    }
                )

    transaction_df = pd.DataFrame(transactions)

    if transaction_df.empty:
        return transaction_df

    transaction_df = (
        transaction_df.sort_values("TX_DATETIME")
        .reset_index(drop=True)
    )

    transaction_df.insert(
        0,
        "TRANSACTION_ID",
        np.arange(len(transaction_df), dtype=int),
    )

    return transaction_df


def inject_fraud_scenarios(
    transactions: pd.DataFrame,
    random_seed: int = 45,
    high_amount_threshold: float = 220.0,
    compromised_terminal_fraction: float = 0.02,
    compromised_customer_fraction: float = 0.01,
) -> pd.DataFrame:
    """Inject reproducible fraud scenarios into transaction data.

    Scenario 1:
        Transactions with unusually high amounts.

    Scenario 2:
        Transactions occurring at a randomly selected subset of
        compromised terminals.

    Scenario 3:
        Transactions from a randomly selected subset of compromised
        customers with unusually large amounts relative to their own
        observed transaction history.

    TX_FRAUD is the machine-learning target.

    TX_FRAUD_SCENARIO is retained only for simulation diagnostics and must
    never be supplied to the fraud prediction model as an input feature.
    """
    required_columns = {
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_AMOUNT",
    }

    missing_columns = required_columns.difference(transactions.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Transactions are missing required columns: {missing}"
        )

    if transactions.empty:
        raise ValueError("transactions must not be empty.")

    if high_amount_threshold <= 0:
        raise ValueError(
            "high_amount_threshold must be greater than zero."
        )

    if not 0 <= compromised_terminal_fraction <= 1:
        raise ValueError(
            "compromised_terminal_fraction must be between 0 and 1."
        )

    if not 0 <= compromised_customer_fraction <= 1:
        raise ValueError(
            "compromised_customer_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(random_seed)

    fraud_df = transactions.copy()

    fraud_df["TX_FRAUD"] = 0
    fraud_df["TX_FRAUD_SCENARIO"] = 0

    # Scenario 1: unusually high-value transactions.
    scenario_1_mask = fraud_df["TX_AMOUNT"] > high_amount_threshold

    fraud_df.loc[
        scenario_1_mask,
        ["TX_FRAUD", "TX_FRAUD_SCENARIO"],
    ] = [1, 1]

    # Scenario 2: compromised payment terminals.
    terminal_ids = fraud_df["TERMINAL_ID"].unique()

    n_compromised_terminals = int(
        len(terminal_ids) * compromised_terminal_fraction
    )

    if compromised_terminal_fraction > 0 and n_compromised_terminals == 0:
        n_compromised_terminals = 1

    if n_compromised_terminals > 0:
        compromised_terminals = rng.choice(
            terminal_ids,
            size=n_compromised_terminals,
            replace=False,
        )

        scenario_2_mask = fraud_df["TERMINAL_ID"].isin(
            compromised_terminals
        )

        fraud_df.loc[
            scenario_2_mask,
            ["TX_FRAUD", "TX_FRAUD_SCENARIO"],
        ] = [1, 2]

    # Scenario 3: compromised customers making unusually large purchases.
    customer_ids = fraud_df["CUSTOMER_ID"].unique()

    n_compromised_customers = int(
        len(customer_ids) * compromised_customer_fraction
    )

    if compromised_customer_fraction > 0 and n_compromised_customers == 0:
        n_compromised_customers = 1

    if n_compromised_customers > 0:
        compromised_customers = rng.choice(
            customer_ids,
            size=n_compromised_customers,
            replace=False,
        )

        customer_mean_amount = fraud_df.groupby(
            "CUSTOMER_ID"
        )["TX_AMOUNT"].transform("mean")

        scenario_3_mask = (
            fraud_df["CUSTOMER_ID"].isin(compromised_customers)
            & (fraud_df["TX_AMOUNT"] > 2 * customer_mean_amount)
        )

        fraud_df.loc[
            scenario_3_mask,
            ["TX_FRAUD", "TX_FRAUD_SCENARIO"],
        ] = [1, 3]

    fraud_df["TX_FRAUD"] = fraud_df["TX_FRAUD"].astype(int)
    fraud_df["TX_FRAUD_SCENARIO"] = (
        fraud_df["TX_FRAUD_SCENARIO"].astype(int)
    )

    return fraud_df


def add_time_features(
    transactions: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    """Add elapsed transaction time features used by the dataset contract."""
    result = transactions.copy()

    start_timestamp = pd.Timestamp(start_date)

    elapsed_seconds = (
        result["TX_DATETIME"] - start_timestamp
    ).dt.total_seconds()

    result["TX_TIME_SECONDS"] = elapsed_seconds.astype(int)

    result["TX_TIME_DAYS"] = (
        result["TX_TIME_SECONDS"] // (24 * 60 * 60)
    ).astype(int)

    return result


def build_fraud_dataset(
    n_customers: int = 5000,
    n_terminals: int = 10000,
    n_days: int = 183,
    start_date: str = "2018-04-01",
    terminal_radius: float = 5.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build the complete reproducible FraudShield transaction dataset."""
    if n_customers <= 0:
        raise ValueError("n_customers must be greater than zero.")

    if n_terminals <= 0:
        raise ValueError("n_terminals must be greater than zero.")

    if n_days <= 0:
        raise ValueError("n_days must be greater than zero.")

    customer_profiles = generate_customer_profiles(
        n_customers=n_customers,
        random_seed=random_seed,
    )

    terminal_profiles = generate_terminal_profiles(
        n_terminals=n_terminals,
        random_seed=random_seed + 1,
    )

    customer_terminals = associate_customer_terminals(
        customer_profiles=customer_profiles,
        terminal_profiles=terminal_profiles,
        radius=terminal_radius,
    )

    transactions = generate_transactions(
        customer_profiles=customer_profiles,
        customer_terminals=customer_terminals,
        start_date=start_date,
        n_days=n_days,
        random_seed=random_seed + 2,
    )

    if transactions.empty:
        raise RuntimeError(
            "Simulation generated no transactions. "
            "Check the simulation configuration."
        )

    transactions = add_time_features(
        transactions=transactions,
        start_date=start_date,
    )

    transactions = inject_fraud_scenarios(
        transactions=transactions,
        random_seed=random_seed + 3,
    )

    ordered_columns = [
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_AMOUNT",
        "TX_TIME_SECONDS",
        "TX_TIME_DAYS",
        "TX_FRAUD",
        "TX_FRAUD_SCENARIO",
    ]

    return (
        transactions[ordered_columns]
        .sort_values("TX_DATETIME")
        .reset_index(drop=True)
    )


def save_dataset(
    transactions: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save the generated transaction dataset to local raw storage."""
    if transactions.empty:
        raise ValueError("Cannot save an empty transaction dataset.")

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions.to_csv(
        output,
        index=False,
    )

    return output