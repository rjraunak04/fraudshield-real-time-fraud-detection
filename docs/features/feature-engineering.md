# FraudShield Feature Engineering

## Objective

FraudShield transforms raw transaction events into behavioural features
that can be calculated using information available at transaction
scoring time.

The Day-02 feature pipeline is designed around three principles:

1. preserve temporal ordering;
2. prevent target and future-information leakage;
3. produce features suitable for reproducible model development.

## Feature Families

### Transaction-Time Features

Features derived directly from the transaction timestamp:

- `TX_HOUR`
- `TX_DAY_OF_WEEK`
- `TX_DAY_OF_MONTH`
- `TX_MONTH`
- `TX_IS_WEEKEND`
- `TX_IS_NIGHT`
- `TX_IS_BUSINESS_HOURS`

These features require no historical or target information.

### Customer Velocity Features

Velocity features summarize recent customer activity before the current
transaction:

- `CUSTOMER_TX_COUNT_1H`
- `CUSTOMER_TX_AMOUNT_1H`
- `CUSTOMER_TX_COUNT_24H`
- `CUSTOMER_TX_AMOUNT_24H`
- `CUSTOMER_TX_COUNT_7D`
- `CUSTOMER_TX_AMOUNT_7D`

The current transaction is excluded from its own velocity features.
Transactions for the same customer at the same timestamp do not observe
one another.

### Historical Behaviour Features

Historical features describe customer and terminal behaviour before the
current transaction:

- `CUSTOMER_HIST_TX_COUNT`
- `CUSTOMER_HIST_MEAN_AMOUNT`
- `CUSTOMER_HIST_MAX_AMOUNT`
- `CUSTOMER_AMOUNT_TO_HIST_MEAN`
- `TERMINAL_HIST_TX_COUNT`
- `TERMINAL_HIST_MEAN_AMOUNT`
- `TERMINAL_AMOUNT_TO_HIST_MEAN`

Cold-start entities receive zero-valued historical statistics until
prior observations exist.

Transactions sharing an entity and timestamp receive the same historical
state so arbitrary transaction identifier ordering cannot create
historical information.

## Leakage Controls

The following columns are explicitly excluded from model inputs:

- `TRANSACTION_ID`
- `TX_DATETIME`
- `TX_FRAUD`
- `TX_FRAUD_SCENARIO`

`TX_FRAUD` is the supervised learning target and therefore cannot be a
predictor.

`TX_FRAUD_SCENARIO` is simulator-only diagnostic information and is not
available as a legitimate production predictor.

No engineered Day-02 feature requires either fraud column.

Historical and velocity features use only information available before
the transaction being scored.

## Temporal Evaluation

FraudShield does not randomly shuffle transactions for model
evaluation.

Transactions are ordered chronologically and divided into:

- 70% training
- 15% validation
- 15% locked test

The locked test partition is reserved for final unbiased evaluation
after model and decision-threshold selection.

## Day-02 Output

The final Day-02 pipeline produces 20 engineered features across
transaction-time, velocity, and historical-behaviour families.

Day-03 consumes this feature pipeline for model development, imbalance
handling, probability calibration, threshold selection, and
business-oriented evaluation.