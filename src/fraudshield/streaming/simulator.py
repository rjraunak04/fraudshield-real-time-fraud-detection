"""Chronological transaction replay for the FraudShield online pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from fraudshield.streaming.events import (
    SimulatedTransaction,
    TransactionEvent,
)

REQUIRED_STREAM_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
)

OPTIONAL_GROUND_TRUTH_COLUMNS = (
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)


class TransactionStream:
    """Replay historical transactions as chronological online events."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = self._prepare_frame(frame)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
    ) -> TransactionStream:
        """Load a historical CSV and construct a replay stream."""
        csv_path = Path(path)

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Transaction dataset not found: {csv_path}"
            )

        frame = pd.read_csv(csv_path)
        return cls(frame)

    @staticmethod
    def _prepare_frame(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and chronologically order replay data."""
        missing = [
            column
            for column in REQUIRED_STREAM_COLUMNS
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                "Missing required streaming columns: "
                + ", ".join(missing)
            )

        prepared = frame.copy()

        prepared["TX_DATETIME"] = pd.to_datetime(
            prepared["TX_DATETIME"],
            errors="raise",
        )

        required = list(REQUIRED_STREAM_COLUMNS)

        if prepared[required].isna().any().any():
            raise ValueError(
                "Required streaming columns contain missing values."
            )

        numeric_columns = (
            "TRANSACTION_ID",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
        )

        for column in numeric_columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="raise",
            )

        for column in (
            "TRANSACTION_ID",
            "CUSTOMER_ID",
            "TERMINAL_ID",
        ):
            values = prepared[column]

            if not (values % 1 == 0).all():
                raise ValueError(
                    f"{column} must contain integer values."
                )

            prepared[column] = values.astype("int64")

        prepared["TX_AMOUNT"] = prepared[
            "TX_AMOUNT"
        ].astype(float)

        if (
            prepared[
                [
                    "TRANSACTION_ID",
                    "CUSTOMER_ID",
                    "TERMINAL_ID",
                    "TX_AMOUNT",
                ]
            ]
            < 0
        ).any().any():
            raise ValueError(
                "Streaming identifiers and amounts "
                "must be non-negative."
            )

        if prepared["TRANSACTION_ID"].duplicated().any():
            raise ValueError(
                "TRANSACTION_ID must be unique."
            )

        prepared = prepared.sort_values(
            ["TX_DATETIME", "TRANSACTION_ID"],
            kind="stable",
        ).reset_index(drop=True)

        return prepared

    def __len__(self) -> int:
        """Return number of replayable transactions."""
        return len(self._frame)

    def iter_events(
        self,
        limit: int | None = None,
    ) -> Iterator[SimulatedTransaction]:
        """Yield transactions one at a time in chronological order."""
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None.")

            if limit < 0:
                raise ValueError("limit must be non-negative.")

        frame = (
            self._frame
            if limit is None
            else self._frame.iloc[:limit]
        )

        has_label = "TX_FRAUD" in frame.columns
        has_scenario = "TX_FRAUD_SCENARIO" in frame.columns

        for row in frame.itertuples(index=False):
            event = TransactionEvent(
                transaction_id=int(row.TRANSACTION_ID),
                event_time=row.TX_DATETIME.to_pydatetime(),
                customer_id=int(row.CUSTOMER_ID),
                terminal_id=int(row.TERMINAL_ID),
                amount=float(row.TX_AMOUNT),
            )

            fraud_label = (
                int(row.TX_FRAUD)
                if has_label
                else None
            )

            fraud_scenario = (
                int(row.TX_FRAUD_SCENARIO)
                if has_scenario
                else None
            )

            yield SimulatedTransaction(
                event=event,
                fraud_label=fraud_label,
                fraud_scenario=fraud_scenario,
            )