"""Human-readable fraud-risk reason codes."""

from __future__ import annotations

from dataclasses import dataclass

from fraudshield.explainability.local import (
    LocalFeatureContribution,
    TransactionExplanation,
)


@dataclass(frozen=True)
class ReasonCode:
    """Stable reviewer-facing explanation."""

    code: str
    message: str
    feature: str
    direction: str
    probability_change: float
    rank: int


_REASON_MAP: dict[
    tuple[str, str],
    tuple[str, str],
] = {
    (
        "TX_AMOUNT",
        "increases_risk",
    ): (
        "HIGH_TRANSACTION_AMOUNT_RISK",
        "Transaction amount contributes to elevated model risk.",
    ),
    (
        "TX_AMOUNT",
        "decreases_risk",
    ): (
        "TRANSACTION_AMOUNT_REASSURING",
        "Transaction amount reduces the model-estimated risk.",
    ),
    (
        "CUSTOMER_AMOUNT_TO_HIST_MEAN",
        "increases_risk",
    ): (
        "CUSTOMER_AMOUNT_DEVIATION_RISK",
        "Transaction amount is unusual relative to the customer's "
        "historical average.",
    ),
    (
        "CUSTOMER_AMOUNT_TO_HIST_MEAN",
        "decreases_risk",
    ): (
        "CUSTOMER_AMOUNT_PATTERN_REASSURING",
        "Transaction amount is consistent with the customer's "
        "historical average.",
    ),
    (
        "CUSTOMER_HIST_MEAN_AMOUNT",
        "increases_risk",
    ): (
        "CUSTOMER_SPENDING_PROFILE_RISK",
        "Customer historical spending profile contributes to "
        "elevated model risk.",
    ),
    (
        "CUSTOMER_HIST_MEAN_AMOUNT",
        "decreases_risk",
    ): (
        "CUSTOMER_SPENDING_PROFILE_REASSURING",
        "Customer historical spending profile reduces "
        "model-estimated risk.",
    ),
    (
        "CUSTOMER_HIST_MAX_AMOUNT",
        "increases_risk",
    ): (
        "CUSTOMER_HISTORICAL_MAX_RISK",
        "Customer historical maximum transaction pattern contributes "
        "to elevated model risk.",
    ),
    (
        "CUSTOMER_HIST_MAX_AMOUNT",
        "decreases_risk",
    ): (
        "CUSTOMER_HISTORICAL_MAX_REASSURING",
        "Customer historical maximum transaction pattern reduces "
        "model-estimated risk.",
    ),
    (
        "CUSTOMER_TX_COUNT_1H",
        "increases_risk",
    ): (
        "CUSTOMER_1H_VELOCITY_RISK",
        "Recent customer transaction frequency contributes to "
        "elevated model risk.",
    ),
    (
        "CUSTOMER_TX_COUNT_24H",
        "increases_risk",
    ): (
        "CUSTOMER_24H_VELOCITY_RISK",
        "Customer transaction frequency over the last 24 hours "
        "contributes to elevated model risk.",
    ),
    (
        "CUSTOMER_TX_COUNT_7D",
        "increases_risk",
    ): (
        "CUSTOMER_7D_VELOCITY_RISK",
        "Customer transaction frequency over the last 7 days "
        "contributes to elevated model risk.",
    ),
    (
        "CUSTOMER_TX_AMOUNT_1H",
        "increases_risk",
    ): (
        "CUSTOMER_1H_SPEND_RISK",
        "Recent customer transaction amount contributes to "
        "elevated model risk.",
    ),
    (
        "CUSTOMER_TX_AMOUNT_24H",
        "increases_risk",
    ): (
        "CUSTOMER_24H_SPEND_RISK",
        "Customer transaction amount over the last 24 hours "
        "contributes to elevated model risk.",
    ),
    (
        "CUSTOMER_TX_AMOUNT_7D",
        "increases_risk",
    ): (
        "CUSTOMER_7D_SPEND_RISK",
        "Customer transaction amount over the last 7 days "
        "contributes to elevated model risk.",
    ),
    (
        "TERMINAL_AMOUNT_TO_HIST_MEAN",
        "increases_risk",
    ): (
        "TERMINAL_AMOUNT_DEVIATION_RISK",
        "Transaction amount is unusual relative to the terminal's "
        "historical transaction pattern.",
    ),
    (
        "TERMINAL_HIST_MEAN_AMOUNT",
        "increases_risk",
    ): (
        "TERMINAL_SPENDING_PROFILE_RISK",
        "Terminal historical amount profile contributes to "
        "elevated model risk.",
    ),
    (
        "TERMINAL_HIST_TX_COUNT",
        "increases_risk",
    ): (
        "TERMINAL_ACTIVITY_RISK",
        "Terminal historical activity contributes to elevated "
        "model risk.",
    ),
    (
        "TX_HOUR",
        "increases_risk",
    ): (
        "TRANSACTION_TIME_RISK",
        "Transaction time contributes to elevated model risk.",
    ),
    (
        "TX_IS_NIGHT",
        "increases_risk",
    ): (
        "NIGHT_TRANSACTION_RISK",
        "Night-time transaction pattern contributes to elevated "
        "model risk.",
    ),
    (
        "TX_IS_WEEKEND",
        "increases_risk",
    ): (
        "WEEKEND_TRANSACTION_RISK",
        "Weekend timing contributes to elevated model risk.",
    ),
}


def build_reason_codes(
    explanation: TransactionExplanation,
    *,
    max_reasons: int = 3,
    risk_only: bool = True,
) -> tuple[ReasonCode, ...]:
    """Convert local contributions into stable reason codes."""
    if max_reasons <= 0:
        raise ValueError(
            "max_reasons must be greater than zero."
        )

    reasons: list[ReasonCode] = []

    for contribution in explanation.contributions:
        if (
            risk_only
            and contribution.direction
            != "increases_risk"
        ):
            continue

        if contribution.direction == "neutral":
            continue

        reason = _build_reason(
            contribution
        )

        reasons.append(reason)

        if len(reasons) >= max_reasons:
            break

    return tuple(reasons)


def _build_reason(
    contribution: LocalFeatureContribution,
) -> ReasonCode:
    """Translate one model contribution."""
    key = (
        contribution.feature,
        contribution.direction,
    )

    mapped = _REASON_MAP.get(key)

    if mapped is None:
        code, message = _fallback_reason(
            contribution
        )
    else:
        code, message = mapped

    return ReasonCode(
        code=code,
        message=message,
        feature=contribution.feature,
        direction=contribution.direction,
        probability_change=(
            contribution.probability_change
        ),
        rank=contribution.rank,
    )


def _fallback_reason(
    contribution: LocalFeatureContribution,
) -> tuple[str, str]:
    """Create a safe fallback for unmapped features."""
    normalized_feature = (
        contribution.feature.upper()
    )

    if contribution.direction == "increases_risk":
        return (
            f"{normalized_feature}_RISK",
            (
                f"{contribution.feature} contributes to "
                "elevated model risk."
            ),
        )

    return (
        f"{normalized_feature}_REASSURING",
        (
            f"{contribution.feature} reduces the "
            "model-estimated risk."
        ),
    )