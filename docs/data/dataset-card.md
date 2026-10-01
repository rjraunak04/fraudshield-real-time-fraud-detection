# FraudShield Dataset Card

## Overview

FraudShield uses a reproducible synthetic payment-transaction dataset inspired
by the transaction simulation concepts presented in the Fraud Detection
Handbook.

The dataset is designed to support development of an end-to-end fraud detection
system involving temporal modelling, behavioural feature engineering,
probability estimation, risk decisions, explainability, and monitoring.

This dataset is synthetic and must not be interpreted as representative of the
full behaviour, economics, fraud prevalence, or operational complexity of a
real payment network.

## Generation Configuration

| Parameter | Value |
|---|---:|
| Customers | 5,000 |
| Terminals | 10,000 |
| Simulation period | 183 days |
| Start date | 2018-04-01 |
| Terminal radius | 5.0 |
| Project random seed | 42 |

## Observed Dataset Summary

| Property | Observed Value |
|---|---:|
| Transactions | 1,845,459 |
| Columns | 9 |
| Legitimate transactions | 1,807,114 |
| Fraudulent transactions | 38,345 |
| Fraud prevalence | 2.078% |
| Duplicate transaction IDs | 0 |
| Missing values | 0 |
| First transaction date | 2018-04-01 |
| Last transaction date | 2018-09-30 |

These values were measured from the generated Day-1 dataset rather than assumed
in advance.

## Schema

| Column | Description |
|---|---|
| `TRANSACTION_ID` | Unique transaction identifier |
| `TX_DATETIME` | Transaction timestamp |
| `CUSTOMER_ID` | Synthetic customer identifier |
| `TERMINAL_ID` | Synthetic payment-terminal identifier |
| `TX_AMOUNT` | Transaction amount |
| `TX_TIME_SECONDS` | Seconds elapsed since simulation start |
| `TX_TIME_DAYS` | Days elapsed since simulation start |
| `TX_FRAUD` | Binary fraud target: 0 = legitimate, 1 = fraud |
| `TX_FRAUD_SCENARIO` | Simulator-only fraud scenario identifier |

## Target

The machine-learning target is:

`TX_FRAUD`

Observed class distribution:

| Class | Transactions |
|---|---:|
| Legitimate (`0`) | 1,807,114 |
| Fraud (`1`) | 38,345 |

The observed fraud prevalence is approximately 2.078%.

Because the target is imbalanced, accuracy will not be treated as the primary
model-selection metric.

## Fraud Scenarios

The simulator records the mechanism responsible for injected fraud using
`TX_FRAUD_SCENARIO`.

Observed distribution:

| Scenario | Transactions |
|---|---:|
| 0 — legitimate | 1,807,114 |
| 1 — high-amount scenario | 1,016 |
| 2 — compromised-terminal scenario | 36,936 |
| 3 — compromised-customer scenario | 393 |

The scenario distribution is a consequence of the current simulation rules and
must not be interpreted as a real-world fraud distribution.

## Leakage Policy

`TX_FRAUD_SCENARIO` is strictly excluded from model features.

It describes the simulator mechanism used to create the fraud label and would
not be available to a real-time fraud model when an unseen transaction arrives.

Using this column during model training would therefore create an unrealistic
information advantage and a serious leakage risk.

`TX_FRAUD` is the prediction target and is also excluded from input features.

All future engineered features must use only information that could reasonably
exist at or before the transaction scoring timestamp.

## Temporal Structure

Transactions span:

`2018-04-01` through `2018-09-30`

The dataset is explicitly time ordered.

Future model development will therefore use temporal splitting rather than
blindly relying on random train/test splitting.

The intended modelling principle is:

`past data -> model development -> future data -> evaluation`

## Data Quality

The generated Day-1 dataset passed the FraudShield data contract.

Observed checks:

- duplicate `TRANSACTION_ID`: 0
- missing values: 0
- negative transaction amounts: 0
- fraud labels restricted to 0 and 1
- both target classes present
- transaction timestamps parseable
- elapsed-time fields non-negative

These checks are enforced programmatically by
`src/fraudshield/data/validate.py`.

## Storage Policy

Generated raw transaction data is stored locally under:

`data/raw/`

Raw transaction files are intentionally excluded from Git version control.

Only source code, configuration, tests, documentation, and lightweight
reproducibility artifacts belong in the repository.

## Reproducibility

Dataset generation is controlled through explicit configuration and
deterministic pseudo-random number generators.

The generation pipeline is implemented in:

`src/fraudshield/data/simulate.py`

The current project configuration is stored in:

`configs/data.yaml`

This allows the dataset to be regenerated without committing the raw CSV to
GitHub.

## Known Limitations

This dataset is synthetic.

Important limitations include:

- simplified customer behaviour;
- simplified merchant/terminal behaviour;
- artificial fraud-generation mechanisms;
- synthetic spatial coordinates rather than real geography;
- no real cardholder, merchant, device, network, or banking information;
- fraud prevalence determined by simulation rules;
- fraud scenarios do not represent the complete diversity of financial fraud;
- transaction costs and financial loss are not currently modelled.

Results obtained from this dataset should therefore be interpreted as
engineering and machine-learning evidence within the FraudShield simulation,
not as estimates of real banking-system performance.