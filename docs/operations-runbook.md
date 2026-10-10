# FraudShield Operational Runbook

This runbook covers failure detection and recovery for the Day-07 production service.

## Fast triage

Check `/health` first. A 200 response means the API process is alive; it does not prove scoring dependencies are ready. Check `/ready` and `/ready/realtime` next, then inspect container logs and `/metrics`.

With Compose:

```powershell
docker compose -f compose.production.yaml ps
docker compose -f compose.production.yaml logs --tail=200 api
docker compose -f compose.production.yaml logs --tail=100 postgres
docker compose -f compose.production.yaml logs --tail=100 redis
```

## Startup and artifact failure

The production bootstrap is fail-fast. Missing or unreadable model/calibrator artifacts prevent the API from starting. Verify the read-only artifact mount and the two configured artifact paths. Only trusted pickle artifacts may be loaded. Do not silently replace a missing artifact or retrain during incident recovery.

## PostgreSQL failure

A database failure makes production readiness fail. During transaction processing, persistence exceptions trigger a SQLAlchemy rollback and are re-raised. Because the online feature-state commit occurs only after the durable database commit, a database write failure does not intentionally advance Redis state.

Recovery: restore PostgreSQL, confirm its container health, confirm readiness, then retry the transaction using the same transaction identifier. Do not manually insert Redis history to compensate for a failed database write.

## Redis failure

Redis is checked during production bootstrap and readiness. A Redis read failure prevents online feature construction. If Redis fails after the database commit but before the state commit, the durable fraud decision can exist while online state is missing that transaction.

This is a known consistency boundary: PostgreSQL and Redis are not a distributed atomic transaction. Before retrying an uncertain transaction, inspect the durable decision by transaction ID. Do not assume a retry is safe merely because Redis lacks state.

Recovery: restore Redis, verify `PING` and application readiness, inspect the transaction's database record, then reconcile state deliberately. Automated cross-store reconciliation is outside the current project scope.

## Model or calibration failure

Scoring exceptions fail the request before a decision is persisted or feature state is committed. Confirm the configured model and calibrator artifacts match the serving contract. Do not alter the frozen review threshold 0.048 or provisional block threshold 0.100 as an incident workaround.

## Duplicate transaction

Duplicate transaction identifiers are rejected by the service contract. Treat a duplicate as an idempotency/retry signal and inspect the existing durable decision rather than generating a second decision.

## Monitoring failure

Application monitoring is intentionally fail-open: telemetry exceptions are logged without changing the business scoring result. A missing metric therefore does not prove a transaction failed. Use the durable decision store for transaction truth.

## Recovery verification

After dependency recovery, require all of the following before returning the service to traffic: containers healthy, `/health` 200, readiness endpoints successful, a controlled smoke transaction completes, its durable decision is queryable, online state advances only after success, and Prometheus counters update.

Do not use the locked model test set for incident checks, threshold changes, or retraining.
