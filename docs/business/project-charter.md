# FraudShield — Project Charter

## Project Overview

FraudShield is a production-style machine-learning project for detecting
potentially fraudulent payment transactions and converting model predictions
into explainable risk decisions.

The project is designed around a realistic engineering question:

> How can an imbalanced fraud-classification problem be transformed into an
> observable real-time risk-scoring and decision service?

FraudShield is not intended to reproduce the infrastructure or economics of a
real financial institution. It is an engineering and machine-learning
simulation built for reproducible experimentation.

## Business Problem

Payment systems must identify suspicious transactions while minimizing
unnecessary disruption to legitimate customers.

Two types of mistakes are particularly important:

### False Negative

A fraudulent transaction is predicted as legitimate.

Potential consequences include:

- financial loss;
- fraud exposure;
- investigation cost;
- customer trust impact.

### False Positive

A legitimate transaction is predicted as fraudulent.

Potential consequences include:

- unnecessary transaction blocking;
- manual-review workload;
- customer friction;
- potential loss of legitimate business.

FraudShield therefore treats fraud detection as both a machine-learning problem
and a risk-decision problem.

## Machine-Learning Objective

For each incoming transaction, the model will eventually estimate:

`P(TX_FRAUD = 1 | information available at transaction time)`

The model output will be a fraud probability rather than a direct business
decision.

Conceptually:

`Transaction -> ML model -> fraud probability`

The probability will later be consumed by a separate decision layer.

## Decision Objective

A future FraudShield decision engine will convert model risk into operational
actions such as:

| Risk Band | Potential Action |
|---|---|
| Low | APPROVE |
| Medium | APPROVE / MONITOR |
| High | REVIEW |
| Critical | BLOCK |

These actions are architectural targets, not finalized Day-1 rules.

Probability thresholds will not be selected arbitrarily.

They will be determined later using validation data, model calibration,
precision-recall behaviour, operational constraints, and explicit cost
assumptions.

## Why Accuracy Is Not the Primary Metric

Fraud detection is an imbalanced classification problem.

A model can achieve apparently high accuracy by predicting most transactions as
legitimate while failing to identify meaningful fraud.

FraudShield will therefore evaluate models using metrics more appropriate for
imbalanced classification.

Planned model metrics include:

- Precision;
- Recall;
- F1 score;
- Precision-Recall AUC;
- ROC-AUC;
- false-positive rate;
- probability calibration.

Threshold-dependent metrics will be evaluated only after an operating threshold
has been defined.

## Business-Oriented Evaluation

Later stages of the project will connect statistical model performance with
decision consequences.

Planned measures include:

- fraud capture rate;
- legitimate transaction false-positive rate;
- manual-review rate;
- fraud amount captured;
- fraud amount missed;
- estimated decision cost.

Any monetary cost assumptions introduced later will be explicitly documented as
illustrative unless supported by an appropriate source.

## Data Leakage Policy

A model may use only information that could reasonably be available at or before
the transaction scoring timestamp.

The following columns are never model features:

- `TX_FRAUD` — prediction target;
- `TX_FRAUD_SCENARIO` — simulator-only information describing how fraud was
  generated.

Future behavioural features must also respect transaction chronology.

For example, a transaction at time `t` must not use customer behaviour observed
after time `t`.

## Temporal Evaluation Principle

FraudShield will treat transactions as time-dependent observations.

The modelling workflow will follow the principle:

`historical data -> training -> later data -> validation/test`

Random splitting will not automatically be assumed appropriate.

This is intended to better approximate the situation in which a deployed model
learns from historical transactions and scores future transactions.

## Planned System Flow

The target system architecture is:

`Transaction`
`-> validation`
`-> feature engineering`
`-> fraud model`
`-> probability calibration`
`-> risk score`
`-> decision engine`
`-> APPROVE / REVIEW / BLOCK`
`-> reason codes`
`-> logging`
`-> monitoring`

Not all components are implemented on Day 1.

The architecture represents the planned end state of FraudShield.

## Day-1 Scope

Day 1 establishes the project and data foundation.

Implemented or targeted Day-1 capabilities include:

- reproducible Python project structure;
- Git feature-branch workflow;
- synthetic customer profiles;
- synthetic terminal profiles;
- customer-terminal associations;
- temporal transaction generation;
- fraud scenario injection;
- reproducible raw dataset generation;
- explicit data contract;
- automated data validation;
- automated validation tests;
- dataset documentation;
- leakage rules.

Model training is intentionally outside the Day-1 scope.

## Success Criteria

FraudShield will be considered successful as a portfolio project when it can
demonstrate the complete path from transaction data to a reproducible,
explainable and observable fraud-risk decision.

The project should demonstrate:

1. statistically appropriate treatment of class imbalance;
2. leakage-safe temporal feature engineering;
3. reproducible model development;
4. calibrated fraud probabilities;
5. evidence-based threshold selection;
6. explainable predictions;
7. API-based real-time scoring;
8. transaction replay or streaming simulation;
9. prediction and decision logging;
10. drift and performance monitoring;
11. automated testing and CI/CD;
12. reproducible deployment and documentation.

## Current Status

**Day 1 — Project and Data Foundation**

No model-performance, latency, throughput, fraud-loss reduction, or
production-scale claims are made at this stage.