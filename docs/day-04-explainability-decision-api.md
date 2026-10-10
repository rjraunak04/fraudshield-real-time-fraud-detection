# FraudShield Day-04 — Explainability, Decisioning, and API

## Objective

Day-04 converts the frozen Day-03 fraud model into an interpretable
and versioned serving boundary.

The goal is not to improve model performance. The goal is to expose
the existing model through a controlled scoring, explanation,
decision, API, and observability architecture.

## Architecture

The Day-04 serving path is:

21 engineered model features
→ frozen fraud model
→ probability calibration
→ FraudScore
→ risk decision policy
→ APPROVE / REVIEW / BLOCK
→ FastAPI response
→ request correlation and latency telemetry

## Frozen model contract

Day-04 preserves the Day-03 model contract.

- Model version: `day-03-hgb-balanced-v1`
- Calibration: `sigmoid`
- Model features: 21
- Review threshold: `0.048`
- Locked test is not reused for tuning.

The scoring service validates the exact model feature schema and
rejects missing, unexpected, non-numeric, NaN, and infinite values.

## Explainability

Two complementary explanation layers are implemented.

### Global explanation

Permutation importance is computed on validation data using average
precision as the scoring metric.

This is a model-behaviour diagnostic and does not imply causality.

### Transaction-level explanation

For a transaction, each feature is replaced independently by a
validation-derived reference value while other features remain fixed.

The change in raw model probability is used as a local contribution.

Positive contributions indicate increased model-estimated risk.
Negative contributions indicate decreased model-estimated risk.

These contributions are not causal effects.

## Human-readable reason codes

Technical local contributions are converted into stable
reviewer-facing reason codes.

Reason codes describe model behaviour and must not be interpreted as
proof that a transaction is fraudulent.

## Decision policy

The operational policy separates risk estimation from business
action.

- `APPROVE`: calibrated probability below `0.048`
- `REVIEW`: calibrated probability from `0.048` to below `0.100`
- `BLOCK`: calibrated probability at or above `0.100`

Policy version:

`day-04-three-way-policy-v1`

The `0.100` block threshold is a provisional operational boundary.
It was not optimized using the locked test.

During the validation-window CP-05 verification, the provisional
block boundary produced zero BLOCK decisions. Therefore it must not
be described as empirically validated or production-optimal.

## FastAPI inference contract

The API exposes:

- `GET /health`
- `GET /ready`
- `POST /v1/score`
- `/openapi.json`

`POST /v1/score` currently accepts the 21 already-engineered model
features.

Raw transaction ingestion plus online historical/velocity feature
state is intentionally deferred to Day-05.

The serving process does not retrain the model for each request.

## Health and readiness

`/health` represents process liveness.

`/ready` represents whether the scoring dependency is available for
inference.

Separating these concepts supports container and orchestration
environments where a process may be alive but temporarily unable to
serve model predictions.

## Observability

Each HTTP request receives:

- `X-Request-ID`
- `X-Process-Time-Ms`

A caller-provided request ID is preserved.

The API emits structured request lifecycle events containing request
metadata, status, and latency without logging the full transaction
feature payload.

This supports request correlation while reducing the risk of
unnecessarily logging transaction-level data.

## Failure behaviour

Invalid request schemas return HTTP 422.

Known scoring failures are converted to safe HTTP 500 responses
without exposing internal model errors to the client.

Unexpected middleware-level failures are converted to a generic
internal-service response and are logged server-side.

## Day-03 locked-test boundary

Day-04 does not reopen the Day-03 locked test.

The frozen Day-03 locked-test results remain the final evaluation for
that model version.

No Day-04 explainability, API, decision-policy, or observability work
is used to retune the model using the locked test.

## Known limitations

The Day-03 model has weak ranking performance and should not be
described as a production-ready fraud detector based on predictive
performance alone.

The review threshold is inherited from the Day-03 illustrative
business-cost analysis.

The block threshold is provisional.

The API currently expects pre-engineered features rather than raw
transactions.

The current TestClient dependency stack emits a Starlette/httpx
deprecation warning. It does not affect current test correctness but
should be resolved during dependency hardening.

Online state management, persistence, streaming simulation, and
operational monitoring are deferred to later project stages.

## Day-04 completion criteria

Day-04 is complete when:

- scoring contract is frozen and validated;
- global explainability is reproducible;
- local explanations are available;
- human-readable reason codes are available;
- the three-way decision policy is versioned;
- the inference API validates requests;
- health and readiness endpoints work;
- request IDs and latency telemetry are exposed;
- end-to-end integration tests pass;
- the full repository regression suite passes;
- the locked test remains untouched.