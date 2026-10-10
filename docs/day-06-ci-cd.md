# Day 06 CP-07 — GitHub Actions CI Hardening

FraudShield uses GitHub Actions as the reproducible cloud verification gate for Python quality and container packaging.

## Trigger policy

The workflow runs on pushes to `main`, `develop`, and every `feature/**` branch. Pull requests targeting `main` or `develop` are also checked, and `workflow_dispatch` keeps a manual verification path available.

Concurrency cancellation prevents obsolete runs for the same branch/ref from consuming CI time after a newer commit arrives.

## Python quality gate

The quality job uses an explicit Ubuntu 24.04 runner and Python 3.12. It installs the package with the development extra from `pyproject.toml`, verifies that `fraudshield` imports successfully, runs Ruff, and then executes the full pytest suite.

The development extra explicitly declares the Starlette TestClient dependency (`httpx2`) so a clean CI environment does not depend on packages that happen to exist in a developer machine.

## Container gate

The Docker job runs only after the Python quality job succeeds. It builds the production image, starts the API, waits for Docker's health check, verifies `/health`, verifies that the intentionally unconfigured `/ready/realtime` endpoint returns HTTP 503, and confirms that the running application process is non-root.

Diagnostics and container cleanup execute even when an earlier container check fails.

## What CI proves

A green workflow demonstrates that a clean Linux runner can install FraudShield, pass static/test gates, build the Docker image, start the service, satisfy its liveness contract, preserve the explicit readiness boundary, and run the image without root privileges.

It does not prove production scoring readiness. Redis, PostgreSQL, model artifacts, secrets, and production dependency injection remain deployment concerns.

No CI step retrains the model, refits calibration, retunes the frozen 0.048 review threshold, changes the provisional 0.100 block threshold, or uses the locked test set for tuning.
