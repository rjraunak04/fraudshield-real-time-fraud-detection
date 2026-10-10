"""Tests for FraudShield human-readable reason codes."""

from __future__ import annotations

import pytest

from fraudshield.explainability.local import (
    LocalFeatureContribution,
    TransactionExplanation,
)
from fraudshield.explainability.reason_codes import (
    build_reason_codes,
)


def contribution(
    *,
    feature: str,
    direction: str,
    probability_change: float,
    rank: int,
) -> LocalFeatureContribution:
    """Build a deterministic local contribution."""
    return LocalFeatureContribution(
        feature=feature,
        feature_value=100.0,
        reference_value=50.0,
        probability_change=probability_change,
        direction=direction,
        rank=rank,
    )


def explanation(
    *items: LocalFeatureContribution,
) -> TransactionExplanation:
    """Build a deterministic transaction explanation."""
    return TransactionExplanation(
        predicted_probability=0.42,
        contributions=tuple(items),
    )


def test_known_risk_feature_maps_to_stable_code() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="TX_AMOUNT",
                direction="increases_risk",
                probability_change=0.12,
                rank=1,
            )
        )
    )

    assert len(result) == 1
    assert (
        result[0].code
        == "HIGH_TRANSACTION_AMOUNT_RISK"
    )
    assert result[0].feature == "TX_AMOUNT"


def test_customer_amount_deviation_maps_correctly() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="CUSTOMER_AMOUNT_TO_HIST_MEAN",
                direction="increases_risk",
                probability_change=0.08,
                rank=1,
            )
        )
    )

    assert (
        result[0].code
        == "CUSTOMER_AMOUNT_DEVIATION_RISK"
    )


def test_risk_only_filters_decreasing_contributions() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="TX_HOUR",
                direction="decreases_risk",
                probability_change=-0.10,
                rank=1,
            ),
            contribution(
                feature="TX_AMOUNT",
                direction="increases_risk",
                probability_change=0.05,
                rank=2,
            ),
        ),
        risk_only=True,
    )

    assert len(result) == 1
    assert result[0].feature == "TX_AMOUNT"


def test_risk_only_false_includes_reassuring_reason() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="TX_AMOUNT",
                direction="decreases_risk",
                probability_change=-0.05,
                rank=1,
            )
        ),
        risk_only=False,
    )

    assert len(result) == 1
    assert (
        result[0].code
        == "TRANSACTION_AMOUNT_REASSURING"
    )


def test_neutral_contribution_is_always_skipped() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="TX_AMOUNT",
                direction="neutral",
                probability_change=0.0,
                rank=1,
            )
        ),
        risk_only=False,
    )

    assert result == ()


def test_max_reasons_is_respected() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="TX_AMOUNT",
                direction="increases_risk",
                probability_change=0.20,
                rank=1,
            ),
            contribution(
                feature="TX_HOUR",
                direction="increases_risk",
                probability_change=0.10,
                rank=2,
            ),
            contribution(
                feature="TX_IS_NIGHT",
                direction="increases_risk",
                probability_change=0.05,
                rank=3,
            ),
        ),
        max_reasons=2,
    )

    assert len(result) == 2
    assert result[0].rank == 1
    assert result[1].rank == 2


def test_original_contribution_order_is_preserved() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="TX_HOUR",
                direction="increases_risk",
                probability_change=0.10,
                rank=1,
            ),
            contribution(
                feature="TX_AMOUNT",
                direction="increases_risk",
                probability_change=0.20,
                rank=2,
            ),
        )
    )

    assert result[0].feature == "TX_HOUR"
    assert result[1].feature == "TX_AMOUNT"


def test_unknown_risk_feature_gets_fallback() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="CUSTOM_UNKNOWN_SIGNAL",
                direction="increases_risk",
                probability_change=0.05,
                rank=1,
            )
        )
    )

    assert (
        result[0].code
        == "CUSTOM_UNKNOWN_SIGNAL_RISK"
    )
    assert "elevated model risk" in (
        result[0].message
    )


def test_unknown_reassuring_feature_gets_fallback() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="CUSTOM_UNKNOWN_SIGNAL",
                direction="decreases_risk",
                probability_change=-0.05,
                rank=1,
            )
        ),
        risk_only=False,
    )

    assert (
        result[0].code
        == "CUSTOM_UNKNOWN_SIGNAL_REASSURING"
    )


def test_reason_retains_probability_change() -> None:
    result = build_reason_codes(
        explanation(
            contribution(
                feature="TX_AMOUNT",
                direction="increases_risk",
                probability_change=0.123,
                rank=1,
            )
        )
    )

    assert result[0].probability_change == pytest.approx(
        0.123
    )


@pytest.mark.parametrize(
    "max_reasons",
    [0, -1],
)
def test_invalid_max_reasons_rejected(
    max_reasons: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_reasons",
    ):
        build_reason_codes(
            explanation(),
            max_reasons=max_reasons,
        )


def test_empty_explanation_returns_no_reasons() -> None:
    result = build_reason_codes(
        explanation()
    )

    assert result == ()