# FraudShield

### Real-Time Fraud Detection & Risk Decision Platform

FraudShield is a production-style machine-learning system that turns a raw transaction into a leakage-safe feature vector, calibrated fraud probability, operational risk decision, durable audit record, online state update, and observable service response.

It was built to answer a practical ML-engineering question:

> **How do you move an imbalanced fraud classifier from a notebook into a stateful, testable, observable real-time decision system without leaking future information or hiding model limitations?**

**Python 3.12 · scikit-learn · FastAPI · PostgreSQL · Redis · Prometheus · Docker · GitHub Actions**

> **Status:** seven-day end-to-end build complete. The system architecture and engineering workflow are production-style; this repository does not claim a deployed banking product or production-grade predictive performance.

---

## At a glance

| Area | Implementation |
| --- | --- |
| Problem | Real-time fraud risk decisioning on an imbalanced transaction stream |
| Data | 1,845,459 synthetic transactions; observed fraud rate ~2.08% |
| Leakage control | Chronological splits + features computed only from prior history |
| Model | Balanced HistGradientBoosting |
| Probability layer | Sigmoid calibration |
| Decision policy | APPROVE / REVIEW / BLOCK |
| Online state | Redis customer + terminal history |
| Durable audit | PostgreSQL + SQLAlchemy |
| Serving | FastAPI raw-transaction and engineered-feature APIs |
| Observability | Runtime, data-quality and decision telemetry + Prometheus |
| Drift | Batch/scheduled PSI evaluation |
| Reliability | Duplicate protection, failure-ordering tests, recovery runbook |
| Delivery | Docker, production Compose, GitHub Actions CI |

## The problem

Fraud detection is not only a binary-classification task. A useful transaction system must deal with several problems simultaneously:

1. **Fraud is rare.** Accuracy can look excellent while the model misses the events that matter.
2. **Transactions are temporal.** Features built with future or current-row information create leakage and unrealistic offline results.
3. **A score is not an action.** The system needs an explicit operating policy for approval, review and blocking.
4. **Online features require state.** Customer and terminal history must be available at inference time without contaminating future requests.
5. **Decisions need an audit trail.** A risk decision should be durable and traceable to its model and policy version.
6. **Production behavior must be observable.** Data quality, request failures, decision mix and drift need explicit monitoring.
7. **Failures cross system boundaries.** Database and online-state updates must have documented ordering and recovery behavior.

FraudShield treats those as one end-to-end system rather than presenting only a trained model.

## Solution architecture

~~~text
                           FRAUDSHIELD

Raw Transaction
      |
      v
Schema Validation + Data-Quality Observation
      |
      v
Past-Only Online Feature Engineering <---------- Redis
      |                                      customer / terminal history
      v
21-Feature Serving Contract
      |
      v
Balanced HistGradientBoosting
      |
      v
Sigmoid Probability Calibration
      |
      v
Calibrated Fraud Probability
      |
      v
Risk Decision Engine
      |
      +----------------+----------------+
      |                |                |
   APPROVE           REVIEW           BLOCK
      |                |                |
      +----------------+----------------+
                       |
                       v
             PostgreSQL Decision Audit
                       |
                       v
                 Redis State Commit
                       |
                       v
        Runtime / Data Quality / Decision Metrics
                       |
                       v
                Prometheus /metrics
~~~

The transaction path intentionally persists the durable decision **before** committing the event to online state. If the database write fails, the transaction is rolled back and Redis state is not intentionally advanced.

PostgreSQL and Redis are separate systems, so this is **not** a distributed atomic transaction. A Redis failure after a successful database commit remains an explicit recovery boundary documented in the runbook.

## From raw event to decision

A raw inference event contains transaction ID, event time, customer ID, terminal ID and amount. Fraud labels and fraud-scenario identifiers never enter the serving request.

~~~text
Transaction
  -> validate event
  -> read history strictly before event time
  -> build customer/terminal temporal features
  -> enforce 21-feature model contract
  -> generate model score
  -> calibrate probability
  -> apply three-way risk policy
  -> persist decision + model/policy metadata
  -> commit online state
  -> emit monitoring telemetry
~~~

Duplicate transaction IDs are rejected instead of silently creating a second durable decision.

## Leakage-safe ML design

The source stream is synthetic data inspired by the Fraud Detection Handbook. The generated dataset contains **1,845,459 transactions** with an observed fraud rate of approximately **2.08%**.

The core evaluation rule is temporal integrity:

- development, validation and locked-test partitions are chronological;
- customer and terminal aggregates use only history strictly before the incoming transaction;
- the current transaction cannot contribute to its own historical features;
- `TX_FRAUD_SCENARIO` is diagnostic metadata and is never a model feature;
- the locked test partition was opened only after model, calibration and threshold choices were frozen.

This prevents the common failure mode where offline fraud metrics benefit from information that would not exist at real inference time.

## Frozen model and decision contract

| Component | Frozen value |
| --- | --- |
| Model | Balanced HistGradientBoosting |
| Model version | `day-03-hgb-balanced-v1` |
| Model features | 21 |
| Calibration | Sigmoid |
| Review threshold | `0.048` |
| Block threshold | `0.100` — provisional |
| Policy version | `day-04-three-way-policy-v1` |

Decision semantics:

~~~text
probability < 0.048          -> APPROVE
0.048 <= probability < 0.100 -> REVIEW
probability >= 0.100         -> BLOCK
~~~

The `0.100` block threshold is deliberately labelled provisional; it is not presented as a validated business optimum.

## Locked-test evaluation

The repository keeps the model results visible rather than optimizing the story around a single attractive metric.

| Metric | Locked-test result |
| --- | ---: |
| Rows | 276,819 |
| Fraud prevalence | 2.0992% |
| Average Precision | 0.047255 |
| ROC-AUC | 0.524296 |
| Precision | 0.966942 |
| Recall | 0.020134 |
| F1 | 0.039447 |
| False-positive rate | 0.000015 |
| Fraud transactions captured | 117 / 5,811 |
| Fraud amount capture rate | 8.1241% |
| Brier score | 0.020566 |
| Log loss | 0.103113 |

**Confusion matrix:** TN 271,004 · FP 4 · FN 5,694 · TP 117.

### What the result means

The frozen operating point is extremely conservative: it produces very few false positives and therefore high precision, but captures only about **2% of fraud transactions**. ROC-AUC around **0.524** also shows weak ranking performance.

That limitation is important. FraudShield demonstrates a disciplined ML-system design, not a claim that this model should block real payments. A stronger model should be developed as a new version using a fresh untouched holdout rather than tuning against this locked test set.

## Real-time service

The main serving endpoint accepts a raw transaction:

`POST /v1/transactions/score`

The service builds the state-aware features, scores and calibrates the transaction, applies the risk policy, persists the decision and then advances online state.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/transactions/score` | End-to-end raw transaction scoring |
| `POST /v1/score` | Score an already engineered 21-feature request |
| `GET /health` | Process liveness |
| `GET /ready` | Engineered-feature scorer readiness |
| `GET /ready/realtime` | Real-time pipeline readiness |
| `GET /metrics` | Prometheus-compatible monitoring |
| `GET /v1/metrics/operations` | Persisted operational decision metrics |

## State, persistence and failure ordering

**Redis** supplies low-latency customer and terminal history for online feature construction. Reads are restricted to observations before the event being scored.

**PostgreSQL** is the durable decision/audit store. Each persisted decision retains the model/policy context required to understand how it was produced.

The critical transaction ordering is:

~~~text
transform -> score -> decide -> PostgreSQL commit -> Redis state commit
~~~

Failure behavior is explicit:

| Failure | Expected behavior |
| --- | --- |
| Model/calibrator unavailable at startup | Fail fast |
| Scoring failure | No decision/state mutation |
| PostgreSQL write failure | Roll back; do not intentionally advance Redis |
| Duplicate transaction | Reject duplicate decision attempt |
| Redis failure after DB commit | Durable decision may exist without state update; inspect/reconcile |
| Monitoring failure | Monitoring is fail-open; transaction truth remains in durable storage |

See `docs/operations-runbook.md` for incident handling and recovery boundaries.

## Monitoring and drift

FraudShield observes more than HTTP uptime. The monitoring layer tracks:

- request successes, failures and processing time;
- duplicate attempts;
- data-quality violations and event-time issues;
- prediction and APPROVE/REVIEW/BLOCK volumes;
- calibrated probability summaries;
- high-risk rate and threshold proximity.

Prometheus-compatible telemetry is exposed at `/metrics`.

Population Stability Index is implemented as a **batch/scheduled** drift evaluator:

| PSI | Operational interpretation |
| ---: | --- |
| < 0.10 | STABLE |
| 0.10–0.25 | WARNING |
| >= 0.25 | DRIFT |

A drift signal means investigate. It does **not** automatically retrain the model or change thresholds. The system also does not claim live precision/recall because fraud labels are not assumed to arrive synchronously.

## Production-style deployment

The runtime uses a Python 3.12 slim image and runs as a non-root user. `compose.production.yaml` defines:

~~~text
API
 |-- PostgreSQL  -> durable decisions
 |-- Redis       -> online feature state
 '-- artifacts   -> read-only trusted model/calibrator files
~~~

Runtime configuration is environment-driven and documented in `.env.example`. Secrets and generated model artifacts are not baked into the image.

Quick start, once the trusted frozen artifacts are available:

~~~powershell
$env:FRAUDSHIELD_POSTGRES_PASSWORD = "<strong-local-password>"
docker compose -f compose.production.yaml up --build -d
docker compose -f compose.production.yaml ps

curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready/realtime
~~~

For the short recruiter-facing walkthrough, use **`docs/demo-guide.md`**.

## CI/CD and verification

GitHub Actions validates the repository in a clean Ubuntu/Python 3.12 environment.

~~~text
Install package
      |
Import verification
      |
Ruff
      |
Full pytest suite
      |
Docker build
      |
Container health
      |
Liveness / readiness semantics
      |
Non-root runtime verification
      |
Diagnostics + cleanup
~~~

The automated suite also covers production configuration, deployment contracts, deterministic end-to-end flow, and failure-ordering behavior.

## Repository structure

~~~text
fraudshield-real-time-fraud-detection/
|
|-- src/fraudshield/
|   |-- api/             FastAPI routes, schemas and observability
|   |-- data/            simulation, validation and temporal splitting
|   |-- decision/        APPROVE / REVIEW / BLOCK policy
|   |-- explainability/  reason-code support
|   |-- features/        offline + online feature engineering
|   |-- modeling/        leakage-safe modeling dataset
|   |-- monitoring/      runtime, quality, drift, decision, Prometheus
|   |-- persistence/     SQLAlchemy durable decision storage
|   |-- production/      settings, bootstrap and ASGI entrypoint
|   |-- realtime/        transaction orchestration
|   |-- scoring/         frozen scoring contract
|   |-- state/           Redis and in-memory state stores
|   '-- streaming/       raw transaction event contract
|
|-- tests/               unit, integration, failure and deployment coverage
|-- docs/                architecture, deployment, monitoring and runbooks
|-- configs/             project/data configuration
|-- .github/workflows/   CI quality + container gates
|-- Dockerfile
|-- compose.yaml
|-- compose.production.yaml
|-- pyproject.toml
'-- .env.example
~~~

## Key engineering decisions

| Decision | Why it matters |
| --- | --- |
| Chronological split | Prevents future information from contaminating evaluation |
| Past-only features | Makes offline feature logic consistent with real inference constraints |
| Probability calibration | Gives the decision layer an explicit probability interface |
| Three-way policy | Separates elevated risk from automatic blocking |
| Redis + PostgreSQL | Separates low-latency online state from durable audit storage |
| Persist before state mutation | Avoids advancing feature state when the durable write fails |
| Explicit model/policy versions | Makes decisions traceable |
| Fail-open monitoring | Telemetry failures do not become payment-decision failures |
| Locked-test discipline | Prevents repeated tuning against the final evaluation set |
| Honest weak-model reporting | Separates ML-system quality from predictive-model quality |

## What is intentionally not claimed

- The current model is **not** presented as production-grade fraud detection.
- The `0.100` BLOCK threshold is **not** presented as business-validated.
- PSI is **not** presented as live automatic retraining.
- PostgreSQL + Redis are **not** presented as one atomic transaction.
- Offline and online features are **not** claimed to have proven exact numerical parity.
- The repository is **not** claimed to be currently hosted in a public cloud.
- Synthetic data is **not** presented as equivalent to a real payment network.

## Next engineering iteration

The next version should be treated as a new model/system release rather than another pass over the existing locked test set:

1. create a fresh untouched evaluation holdout;
2. investigate richer temporal and behavioral fraud signals and stronger ranking models;
3. validate decision thresholds against explicit business costs;
4. add delayed-label model-performance monitoring;
5. replace startup schema creation with migrations;
6. add idempotent PostgreSQL/Redis reconciliation for partial cross-store failures.

---

### Project completion

The seven-day FraudShield build is complete across data foundation, leakage-safe ML, calibrated decisioning, real-time stateful serving, durable persistence, observability, failure handling, Docker packaging, CI/CD and production configuration.

**Demo:** `docs/demo-guide.md`  
**Operations:** `docs/operations-runbook.md`  
**Production deployment:** `docs/day-07-production-deployment.md`  
**Final audit:** `docs/day-07-final-audit.md`
