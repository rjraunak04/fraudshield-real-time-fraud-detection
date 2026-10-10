# FraudShield Day-03 — Model Development

## Objective

Day-03 converts the leakage-safe FraudShield feature pipeline into a
validated fraud-scoring workflow.

The modeling workflow uses chronological partitions and keeps the final
test period locked until the model, calibration strategy, and operating
threshold are frozen.

## Modeling Dataset

The model uses an explicit allowlist of 21 features.

Raw identifiers, target information, simulator fraud-scenario metadata,
and simulator progression fields are excluded from the model input.

Excluded fields include:

- `TRANSACTION_ID`
- `TX_DATETIME`
- `TX_FRAUD`
- `TX_FRAUD_SCENARIO`
- `CUSTOMER_ID`
- `TERMINAL_ID`
- `TX_TIME_SECONDS`
- `TX_TIME_DAYS`

The feature engineering pipeline is past-only and chronological.

## Evaluation Strategy

Accuracy is not treated as the primary fraud metric because the target
is highly imbalanced.

Evaluation includes:

- Average precision / PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- False-positive rate
- Predicted-positive rate
- Fraud transaction capture
- Fraud amount capture
- Probability calibration metrics
- Illustrative business cost

## Baselines

Two naive baselines were evaluated before machine-learning models:

1. Constant training-prevalence probability.
2. Transaction-amount percentile score.

The amount-only baseline demonstrated why fraud amount capture alone is
not sufficient: a strategy can capture substantial fraud value while
generating an operationally unacceptable false-positive rate.

## Candidate Models

Day-03 evaluated:

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

Unweighted and cost-sensitive configurations were compared.

No SMOTE or synthetic resampling is used in the Day-03 modeling
workflow.

## Imbalance Strategy

Cost-sensitive learning is used instead of ordinary oversampling.

Balanced class/sample weights are derived from the training target only.

The selected candidate was the balanced Histogram Gradient Boosting
model based on validation average precision.

## Probability Calibration

Class weighting improves sensitivity to the minority class but changes
the interpretation of raw model probabilities.

A sigmoid calibrator is therefore fitted on the earlier half of the
validation period.

Calibration is evaluated on the later validation half.

The calibration stage substantially improved probability-quality
metrics while preserving ranking in the observed validation evaluation.

## Threshold Selection

The final operating threshold is:

`0.048`

The threshold was selected on the later validation window after model
selection and calibration.

The threshold optimization uses illustrative portfolio assumptions:

- False-positive review cost: 50 currency units.
- Missed-fraud cost: transaction amount.
- Detected-fraud review cost: 10 currency units.

These values are project assumptions and must not be interpreted as
real banking or payment-industry costs.

Under these assumptions, the selected validation operating point is
highly conservative: it has very low false-positive rate and high
precision, but low fraud recall.

This is an important modeling result rather than something to hide.
Fraud operating thresholds depend strongly on institution-specific
losses, review capacity, customer-friction costs, and risk appetite.

## Locked-Test Policy

The locked test partition is not used for:

- model selection,
- hyperparameter selection,
- class-weight strategy selection,
- calibration fitting, or
- threshold selection.

It is opened only after the complete Day-03 pipeline is frozen.

Test-set results must not be used to retune the Day-03 model.

## Important Methodological Limitation

Model-family selection in earlier checkpoints used aggregate validation
performance before the validation period was subsequently divided into
calibration and threshold-evaluation windows.

Therefore, the later validation half should not be described as a
completely untouched model-selection holdout.

The chronological locked test remains the final out-of-sample
evaluation period.

## Synthetic-Data Limitation

FraudShield currently uses a synthetic fraud dataset inspired by the
Fraud Detection Handbook.

The simulator is useful for reproducible engineering and modeling
experiments, but synthetic fraud patterns are not evidence of
production performance.

Reported metrics must therefore be described as project benchmark
results rather than real-world fraud-detection performance.

## Day-03 Frozen Pipeline

The final Day-03 workflow is:

Transaction history

→ leakage-safe past-only feature engineering

→ explicit 21-feature model schema

→ balanced Histogram Gradient Boosting

→ sigmoid probability calibration

→ calibrated fraud probability

→ operating threshold 0.048

→ fraud classification and evaluation

The next project stage will convert this scoring pipeline into an
explainable risk-decision service.