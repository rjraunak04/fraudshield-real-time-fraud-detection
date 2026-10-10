# FraudShield — Real-Time Fraud Detection & Risk Decision Platform

FraudShield is an end-to-end machine-learning system for converting raw financial transactions into calibrated fraud risk, operational decisions, durable audit records, online feature state, and production monitoring.

The project focuses on the engineering problems around fraud detection: severe class imbalance, temporal leakage, probability calibration, threshold-based decision policy, stateful online features, persistence, observability, failure handling, containerization, and CI/CD.

> The current model is intentionally reported with its limitations. FraudShield is a production-style portfolio system, not a claim of a deployed banking fraud product.

## System architecture

```text
Raw Transaction
      |
      v
Validation / Data Quality
      |
      v
Past-Only Online Features <------ Redis state
      |
      v
Balanced HistGradientBoosting
      |
      v
Sigmoid Probability Calibration
      |
      v
Risk Decision Engine
  |       |       |
APPROVE  REVIEW  BLOCK
      |
      +------> PostgreSQL decision audit
      |
      +------> Redis state commit
      |
      +------> Runtime / model-decision monitoring
                         |
                         v
                   Prometheus /metrics
```

The durable database write occurs before the online state commit. PostgreSQL and Redis are separate systems, so this ordering reduces one failure mode but is not a distributed atomic transaction.

## ML methodology

The source dataset is a large synthetic transaction stream inspired by the Fraud Detection Handbook. The generated dataset contains 1,845,459 transactions with an observed fraud rate of about 2.08%.

FraudShield uses chronological train/validation/locked-test partitions rather than random splitting. Online and offline features are designed around past-only customer and terminal history, with the current transaction excluded from its own historical aggregates.

The frozen serving contract is:

| Component | Frozen value |
| --- | --- |
| Model | Balanced HistGradientBoosting |
| Model version | `day-03-hgb-balanced-v1` |
| Model features | 21 |
| Calibration | Sigmoid |
| Review threshold | `0.048` |
| Provisional block threshold | `0.100` |
| Policy version | `day-04-three-way-policy-v1` |

The block threshold is provisional; it was not presented as an optimized or validated business threshold.

## Locked-test results

The locked test partition was opened once after model, calibration, and threshold selection were frozen.

| Metric | Result |
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
| Fraud transaction capture rate | 2.0134% |
| Fraud amount capture rate | 8.1241% |
| Brier score | 0.020566 |
| Log loss | 0.103113 |

Confusion matrix: TN 271,004, FP 4, FN 5,694, TP 117.

These results show a deliberately conservative operating point but weak ranking performance. The high precision must not be read in isolation: recall is only about 2%. Improving the predictive model would require a new experimental cycle and a fresh holdout/versioning strategy rather than tuning against this locked test set.

## Real-time decision service

The primary raw-transaction route is `POST /v1/transactions/score`. A successful request flows through state-aware feature engineering, calibrated scoring, three-way decisioning, durable persistence, and then online-state mutation.

Additional service surfaces include:

- `POST /v1/score` — score an already engineered 21-feature transaction.
- `GET /health` — process liveness.
- `GET /ready` — engineered-feature scorer readiness.
- `GET /ready/realtime` — real-time pipeline readiness.
- `GET /metrics` — Prometheus-compatible runtime/data-quality/decision telemetry.
- `GET /v1/metrics/operations` — dashboard-oriented operational decision metrics.

Duplicate transaction identifiers are rejected rather than silently producing a second durable decision.

## Stateful features and persistence

Redis stores online customer and terminal transaction history used by the feature service. Reads are restricted to history strictly before the incoming event time to protect against current-row leakage.

PostgreSQL stores durable fraud decisions and their model/policy metadata. SQLAlchemy isolates the persistence layer from the API. SQLite and the in-memory state backend are used for deterministic automated integration tests; they are not advertised as the production stores.

## Monitoring and drift

FraudShield tracks request outcomes, processing time, duplicate attempts, data-quality violations, prediction volume, APPROVE/REVIEW/BLOCK mix, calibrated probability summaries, high-risk rate, and threshold proximity.

Prometheus exposition is available at `/metrics`. Population Stability Index (PSI) is implemented as a batch/scheduled drift evaluator with operational bands:

- PSI < 0.10: STABLE
- 0.10 <= PSI < 0.25: WARNING
- PSI >= 0.25: DRIFT

Drift is an investigation signal. It does not automatically retrain the model or change decision thresholds. The monitoring layer also does not claim live precision/recall because ground-truth fraud labels are not assumed to arrive synchronously.

## Production packaging

The runtime is packaged in a Python 3.12 slim container and runs as a non-root user. `compose.production.yaml` defines the API, PostgreSQL, and Redis topology with dependency health checks and persistent service volumes.

Required runtime inputs are documented in `.env.example`. Model and calibrator artifacts are mounted read-only and are intentionally excluded from the Docker image. Only trusted pickle artifacts should be loaded.

Example local production-style start:

```powershell
$env:FRAUDSHIELD_POSTGRES_PASSWORD = "<strong-local-password>"
docker compose -f compose.production.yaml up --build -d
docker compose -f compose.production.yaml ps
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready/realtime
```

This repository packages a production-style topology; it does not claim a currently hosted cloud deployment.

## CI/CD quality gates

GitHub Actions validates the project on a clean Ubuntu runner using Python 3.12. The pipeline installs the package, verifies imports, runs Ruff, executes the full pytest suite, builds the Docker image, starts it, waits for container health, verifies liveness and readiness semantics, confirms non-root execution, captures diagnostics, and stops the container.

The production deployment contract and end-to-end smoke path are also covered by automated tests.

## Failure and recovery design

The service fails fast when required model/calibrator artifacts or startup dependencies are unavailable. Scoring failures occur before persistence/state mutation. Database write failures roll back and do not intentionally advance online state.

A Redis failure after the database commit is a documented consistency boundary: a durable decision may exist while online state is missing the event. The operational runbook describes inspection and reconciliation rather than pretending PostgreSQL and Redis are transactionally atomic.

See `docs/operations-runbook.md` for incident triage and recovery guidance.

## Repository map

```text
src/fraudshield/
  api/             FastAPI schemas, routes, observability
  data/            simulation, validation, temporal splitting
  decision/        three-way risk policy
  explainability/  reviewer-facing reason-code support
  features/        offline and online feature engineering
  modeling/        leakage-safe modeling dataset
  monitoring/      runtime, quality, drift, decision, Prometheus
  persistence/     SQLAlchemy decision storage
  production/      environment settings and dependency bootstrap
  realtime/        transaction orchestration
  scoring/         frozen scoring contract
  state/           in-memory and Redis state stores
  streaming/       raw transaction event contract

tests/             unit, integration, failure and deployment tests
docs/              architecture, monitoring, deployment and runbooks
.github/workflows/ clean CI quality and container gates
```

## Engineering decisions worth discussing

**Why temporal splitting?** Fraud systems operate forward in time. Random splits can mix future behavior into model development and produce misleading evaluation.

**Why calibration?** Decision thresholds consume probabilities. Calibration separates probability quality from ranking quality and makes the risk-policy interface explicit.

**Why Redis plus PostgreSQL?** Redis serves low-latency historical state for online features; PostgreSQL provides durable decision/audit storage. Their different roles are explicit, as is their consistency boundary.

**Why three decisions instead of binary fraud/not-fraud?** Operational systems often need a middle path for manual review rather than treating every elevated score as an automatic block.

**Why keep weak model results?** The goal is to demonstrate correct evaluation and system design without hiding inconvenient metrics. The locked-test ROC-AUC of about 0.524 and low recall make the current model limitation visible and create a clear next-model-version problem.

## Known limitations

The synthetic fraud scenarios do not reproduce the full complexity of real payment networks. The current model has weak locked-test ranking performance. The block threshold remains provisional. PostgreSQL schema creation currently occurs during bootstrap rather than through a migration framework. PostgreSQL and Redis commits are not distributed-atomic. Live ground-truth performance monitoring and automated cross-store reconciliation are not implemented. The repository also does not claim a public cloud deployment.

## Tech stack

Python 3.12 · pandas · NumPy · scikit-learn · FastAPI · Pydantic · SQLAlchemy · PostgreSQL · Redis · Prometheus · Docker · Docker Compose · pytest · Ruff · GitHub Actions

## Project status

Day 01–06 established the data, leakage-safe ML, calibrated decision policy, real-time stateful service, persistence, monitoring, Docker, and CI/CD layers. Day 07 adds production configuration, service composition, deployment packaging, end-to-end smoke testing, failure/recovery controls, and recruiter-facing documentation.

The final release audit and demo/interview packaging are handled as separate Day-07 checkpoints.
