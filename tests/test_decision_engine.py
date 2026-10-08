"""Tests for the FraudShield risk decision engine."""

from __future__ import annotations

import pytest

from fraudshield.decision.engine import (
    DECISION_POLICY_VERSION,
    DecisionAction,
    DecisionPolicy,
    RiskDecisionEngine,
)
from fraudshield.explainability.reason_codes import ReasonCode
from fraudshield.scoring.service import FraudScore


def make_score(
    probability: float,
) -> FraudScore:
    """Build a deterministic FraudScore."""
    return FraudScore(
        raw_probability=probability,
        calibrated_probability=probability,
        threshold=0.048,
        is_high_risk=probability >= 0.048,
        model_version="test-model-v1",
        calibration_method="sigmoid",
    )


def make_reason() -> ReasonCode:
    """Build one deterministic reason code."""
    return ReasonCode(
        code="HIGH_TRANSACTION_AMOUNT_RISK",
        message=(
            "Transaction amount contributes to "
            "elevated model risk."
        ),
        feature="TX_AMOUNT",
        direction="increases_risk",
        probability_change=0.10,
        rank=1,
    )


def test_low_probability_is_approved() -> None:
    engine = RiskDecisionEngine()

    decision = engine.decide(
        make_score(0.020)
    )

    assert decision.action == DecisionAction.APPROVE


def test_review_probability_is_reviewed() -> None:
    engine = RiskDecisionEngine()

    decision = engine.decide(
        make_score(0.060)
    )

    assert decision.action == DecisionAction.REVIEW


def test_high_probability_is_blocked() -> None:
    engine = RiskDecisionEngine()

    decision = engine.decide(
        make_score(0.150)
    )

    assert decision.action == DecisionAction.BLOCK


def test_review_threshold_is_inclusive() -> None:
    policy = DecisionPolicy(
        review_threshold=0.048,
        block_threshold=0.100,
    )

    decision = RiskDecisionEngine(
        policy
    ).decide(
        make_score(0.048)
    )

    assert decision.action == DecisionAction.REVIEW


def test_block_threshold_is_inclusive() -> None:
    policy = DecisionPolicy(
        review_threshold=0.048,
        block_threshold=0.100,
    )

    decision = RiskDecisionEngine(
        policy
    ).decide(
        make_score(0.100)
    )

    assert decision.action == DecisionAction.BLOCK


def test_probability_just_below_review_is_approved() -> None:
    engine = RiskDecisionEngine()

    decision = engine.decide(
        make_score(0.047999)
    )

    assert decision.action == DecisionAction.APPROVE


def test_probability_just_below_block_is_reviewed() -> None:
    engine = RiskDecisionEngine()

    decision = engine.decide(
        make_score(0.099999)
    )

    assert decision.action == DecisionAction.REVIEW


def test_reason_codes_are_preserved() -> None:
    engine = RiskDecisionEngine()
    reason = make_reason()

    decision = engine.decide(
        make_score(0.150),
        reason_codes=(reason,),
    )

    assert decision.reason_codes == (reason,)


def test_metadata_is_preserved() -> None:
    engine = RiskDecisionEngine()

    decision = engine.decide(
        make_score(0.060)
    )

    assert decision.model_version == "test-model-v1"
    assert decision.calibration_method == "sigmoid"
    assert (
        decision.policy_version
        == DECISION_POLICY_VERSION
    )


def test_custom_policy_is_supported() -> None:
    policy = DecisionPolicy(
        review_threshold=0.20,
        block_threshold=0.40,
        version="custom-v1",
    )

    engine = RiskDecisionEngine(policy)

    assert (
        engine.decide(
            make_score(0.10)
        ).action
        == DecisionAction.APPROVE
    )

    assert (
        engine.decide(
            make_score(0.30)
        ).action
        == DecisionAction.REVIEW
    )

    assert (
        engine.decide(
            make_score(0.50)
        ).action
        == DecisionAction.BLOCK
    )


@pytest.mark.parametrize(
    (
        "review_threshold",
        "block_threshold",
    ),
    [
        (0.10, 0.10),
        (0.20, 0.10),
    ],
)
def test_invalid_threshold_order_rejected(
    review_threshold: float,
    block_threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="strictly less",
    ):
        DecisionPolicy(
            review_threshold=review_threshold,
            block_threshold=block_threshold,
        )


@pytest.mark.parametrize(
    "review_threshold",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_review_threshold_rejected(
    review_threshold: float,
) -> None:
    with pytest.raises(ValueError):
        DecisionPolicy(
            review_threshold=review_threshold,
            block_threshold=1.0,
        )


@pytest.mark.parametrize(
    "block_threshold",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_block_threshold_rejected(
    block_threshold: float,
) -> None:
    with pytest.raises(ValueError):
        DecisionPolicy(
            review_threshold=0.0,
            block_threshold=block_threshold,
        )


def test_empty_policy_version_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="version",
    ):
        DecisionPolicy(
            version="   ",
        )


def test_reason_codes_must_be_tuple() -> None:
    engine = RiskDecisionEngine()

    with pytest.raises(
        TypeError,
        match="tuple",
    ):
        engine.decide(
            make_score(0.10),
            reason_codes=[make_reason()],  # type: ignore[arg-type]
        )


def test_reason_codes_must_contain_reason_objects() -> None:
    engine = RiskDecisionEngine()

    with pytest.raises(
        TypeError,
        match="ReasonCode",
    ):
        engine.decide(
            make_score(0.10),
            reason_codes=("bad",),  # type: ignore[arg-type]
        )