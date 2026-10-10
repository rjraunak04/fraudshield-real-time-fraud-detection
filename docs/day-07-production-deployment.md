# Day 07 CP-03 — Production Deployment Packaging

FraudShield now has an explicit production deployment stack separate from the Day-06 image smoke-test configuration.

## Service topology

`compose.production.yaml` starts three services:

- PostgreSQL for durable fraud decisions.
- Redis for online customer and terminal feature state.
- FraudShield API using the production dependency bootstrap.

The API waits for healthy PostgreSQL and Redis services before starting. Its container health check uses `/ready/realtime`, while the lightweight `/health` endpoint remains process liveness.

## Required runtime inputs

Set `FRAUDSHIELD_POSTGRES_PASSWORD` outside source control. Place the trusted frozen model and calibrator artifacts at:

- `artifacts/fraud_model.pkl`
- `artifacts/fraud_calibrator.pkl`

The artifacts are mounted read-only and are intentionally excluded from the Docker build context. Never load untrusted pickle artifacts.

## Start

```powershell
$env:FRAUDSHIELD_POSTGRES_PASSWORD = "<strong-local-password>"
docker compose -f compose.production.yaml up --build -d
docker compose -f compose.production.yaml ps
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready
curl.exe http://localhost:8000/ready/realtime
```

## Stop

```powershell
docker compose -f compose.production.yaml down
```

Add `-v` only when you intentionally want to remove the PostgreSQL and Redis named volumes.

## Deployment boundaries

This checkpoint packages the service topology and production entrypoint; it does not create or publish model artifacts and does not claim a hosted cloud deployment. PostgreSQL schema creation still happens during application bootstrap rather than through a migration system. Redis and PostgreSQL are separate systems, so the existing durable-database-write-before-state ordering is not a distributed atomic transaction.

The frozen Day-03 model version, sigmoid calibration, review threshold 0.048, and provisional block threshold 0.100 remain unchanged. No locked-test data is reopened or used for tuning.
