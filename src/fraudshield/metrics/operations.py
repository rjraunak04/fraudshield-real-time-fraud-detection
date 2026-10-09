"""Operational metrics for FraudShield real-time decisions."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import case, func, select

from fraudshield.persistence.models import FraudDecisionRecord
from fraudshield.persistence.repository import FraudDecisionRepository


@dataclass(frozen=True, slots=True)
class OperationalMetrics:
    """Aggregated operational decision metrics."""

    total_transactions: int

    approve_count: int
    review_count: int
    block_count: int

    approve_rate: float
    review_rate: float
    block_rate: float

    average_calibrated_probability: float
    high_risk_rate: float

    total_transaction_amount: float
    average_transaction_amount: float


def _safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    """Return a zero-safe rate."""
    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_operational_metrics(
    repository: FraudDecisionRepository,
) -> OperationalMetrics:
    """Aggregate operational metrics directly in SQL."""
    session = repository.session

    statement = select(
        func.count(FraudDecisionRecord.id),
        func.coalesce(
            func.sum(
                case(
                    (
                        FraudDecisionRecord.decision
                        == "APPROVE",
                        1,
                    ),
                    else_=0,
                )
            ),
            0,
        ),
        func.coalesce(
            func.sum(
                case(
                    (
                        FraudDecisionRecord.decision
                        == "REVIEW",
                        1,
                    ),
                    else_=0,
                )
            ),
            0,
        ),
        func.coalesce(
            func.sum(
                case(
                    (
                        FraudDecisionRecord.decision
                        == "BLOCK",
                        1,
                    ),
                    else_=0,
                )
            ),
            0,
        ),
        func.coalesce(
            func.avg(
                FraudDecisionRecord.calibrated_probability
            ),
            0.0,
        ),
        func.coalesce(
            func.sum(FraudDecisionRecord.amount),
            0.0,
        ),
        func.coalesce(
            func.avg(FraudDecisionRecord.amount),
            0.0,
        ),
    )

    row = session.execute(statement).one()

    total = int(row[0] or 0)
    approve_count = int(row[1] or 0)
    review_count = int(row[2] or 0)
    block_count = int(row[3] or 0)

    average_probability = float(row[4] or 0.0)
    total_amount = float(row[5] or 0.0)
    average_amount = float(row[6] or 0.0)

    high_risk_count = (
        review_count + block_count
    )

    return OperationalMetrics(
        total_transactions=total,
        approve_count=approve_count,
        review_count=review_count,
        block_count=block_count,
        approve_rate=_safe_rate(
            approve_count,
            total,
        ),
        review_rate=_safe_rate(
            review_count,
            total,
        ),
        block_rate=_safe_rate(
            block_count,
            total,
        ),
        average_calibrated_probability=(
            average_probability
        ),
        high_risk_rate=_safe_rate(
            high_risk_count,
            total,
        ),
        total_transaction_amount=total_amount,
        average_transaction_amount=average_amount,
    )