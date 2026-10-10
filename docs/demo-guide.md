# FraudShield Demo Guide

A short recruiter-facing demo for the frozen FraudShield release. This demonstrates the system flow without making claims beyond what the repository verifies.

## 1. Start the production-style stack

Place the trusted frozen model and calibrator artifacts under `artifacts/`, provide the PostgreSQL password outside source control, and start the services.

~~~powershell
$env:FRAUDSHIELD_POSTGRES_PASSWORD = "<strong-local-password>"
docker compose -f compose.production.yaml up --build -d
docker compose -f compose.production.yaml ps
~~~

The production Compose stack contains the FraudShield API, PostgreSQL, and Redis.

## 2. Verify health and readiness

~~~powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready
curl.exe http://localhost:8000/ready/realtime
~~~

`/health` checks process liveness. The readiness endpoints represent whether the configured scoring paths are available.

## 3. Score a raw transaction

~~~powershell
$body = @{
  transaction_id = 900001
  event_time = "2018-06-15T14:30:00"
  customer_id = 101
  terminal_id = 501
  amount = 125.50
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/transactions/score" -ContentType "application/json" -Body $body
~~~

The request flows through validation, past-only online feature engineering, calibrated fraud scoring, the three-way risk policy, PostgreSQL persistence, Redis state update, and monitoring.

Do not expect a hard-coded APPROVE, REVIEW, or BLOCK result from this example. The result depends on the configured frozen artifacts and the available online history.

## 4. Show operational observability

~~~powershell
curl.exe http://localhost:8000/metrics
curl.exe http://localhost:8000/v1/metrics/operations
~~~

The Prometheus endpoint exposes runtime, data-quality, and model-decision telemetry. The operational endpoint summarizes persisted decisions.

PSI drift evaluation is batch/scheduled rather than a live metric and does not automatically retrain the model.

## 5. Demonstrate duplicate protection

Run the same transaction request again with transaction ID `900001`. The service should reject an already processed transaction instead of silently creating another durable decision.

## 6. Show failure and recovery design

Use `docs/operations-runbook.md` to demonstrate the documented behavior for model/artifact, PostgreSQL, Redis, duplicate, and monitoring failures.

The processing order is:

~~~text
Raw transaction
      |
Past-only features from Redis
      |
Calibrated fraud score
      |
APPROVE / REVIEW / BLOCK
      |
PostgreSQL durable decision
      |
Redis state commit
      |
Monitoring / Prometheus
~~~

A database write failure rolls back before online state is intentionally advanced. PostgreSQL and Redis are not a distributed atomic transaction, so a Redis failure after the database commit remains an explicit recovery boundary.

## 7. Show CI/CD

Open the latest successful GitHub Actions run. The workflow verifies package installation/import, Ruff, the full pytest suite, Docker image construction, container health, liveness/readiness semantics, and non-root runtime behavior.

## Demo boundaries

This demo proves the implemented application and infrastructure flow; it is not evidence that the current predictive model is suitable for blocking real financial transactions.

The frozen model has weak locked-test ranking performance, the `0.100` BLOCK threshold remains provisional, the dataset is synthetic, and the repository does not claim a currently hosted cloud deployment.
