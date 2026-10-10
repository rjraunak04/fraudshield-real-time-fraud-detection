# FraudShield Day-05: Real-Time Fraud Platform

## 1. Objective

Day-05 extends FraudShield from an engineered-feature inference service
into an end-to-end real-time fraud decision platform.

The platform accepts raw transaction facts, reconstructs leakage-safe
online features from historical state, applies the frozen fraud scoring
contract, routes the transaction through the risk decision policy,
persists the decision for auditability, updates online state, and
exposes operational metrics for dashboard consumption.

Day-05 does not retrain the model, retune thresholds, or reopen the
locked Day-03 test set.

---

## 2. Real-Time Architecture

```text
Raw Transaction
      |
      v
TransactionEvent
      |
      v
OnlineFeaturePipeline
      |
      +---- READ historical state ----> Redis-compatible State
      |
      v
21-Feature Frozen Model Contract
      |
      v
FraudScorer
      |
      v
Calibrated Fraud Probability
      |
      v
RiskDecisionEngine
      |
      +---- APPROVE
      +---- REVIEW
      +---- BLOCK
      |
      v
Durable Decision Persistence
(PostgreSQL-compatible SQLAlchemy layer)
      |
      v
Online State Commit
      |
      v
Operational Metrics / Dashboard API