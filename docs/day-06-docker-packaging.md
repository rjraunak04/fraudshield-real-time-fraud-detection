# Day 06 CP-06 — Docker Production Packaging

FraudShield is packaged as a small Python 3.12 container for the FastAPI service.

## Container contract

- Base image: `python:3.12-slim`.
- Package installation uses the repository's `pyproject.toml`.
- The API runs as the unprivileged `fraudshield` user.
- Python bytecode and pip caches are disabled in the runtime image.
- Uvicorn binds to `0.0.0.0` and reads `PORT`, defaulting to `8000`.
- The image exposes port 8000.
- Docker health checks call the lightweight `/health` liveness endpoint.
- The default image contains source code required for serving, but excludes tests, notebooks, local environments, raw data, local databases, secrets, and generated model artifacts through `.dockerignore`.

## Build and verify

```powershell
docker build -t fraudshield:day06 .
docker run --rm -d --name fraudshield-day06 -p 8000:8000 fraudshield:day06
docker inspect --format='{{.State.Health.Status}}' fraudshield-day06
curl.exe http://localhost:8000/health
docker logs fraudshield-day06
docker stop fraudshield-day06
```

The default application intentionally reports 503 from readiness endpoints whose production dependencies have not been injected. Liveness at `/health` is therefore used for the image health check; a healthy process must not be confused with configured scoring readiness.

## Compose smoke test

```powershell
docker compose up --build -d
docker compose ps
curl.exe http://localhost:8000/health
docker compose down
```

## Production boundary

CP-06 validates reproducible container packaging and process liveness. It does not claim that Redis, PostgreSQL, trained model artifacts, or production dependency wiring are bundled into this image. Those runtime dependencies remain explicit deployment concerns rather than being hidden behind a misleading container health check.

The frozen Day-03 model, sigmoid calibration, review threshold 0.048, and provisional block threshold 0.100 are unchanged. The locked test set is not reopened by container packaging.
