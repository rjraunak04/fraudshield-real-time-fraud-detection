# Day 06 — Monitoring, Drift, Docker, and CI/CD

Day 06 adds the operational controls around the frozen FraudShield scoring system without retraining or retuning it.

## Delivered checkpoints

- **CP-01 — Runtime monitoring:** thread-safe request, outcome, duplicate-attempt, and processing-time snapshots.
- **CP-02 — Data-quality monitoring:** invalid identifiers/amounts, duplicates, future timestamps, and event-time regressions.
- **CP-03 — Drift monitoring:** PSI-based batch evaluation with STABLE, WARNING, and DRIFT operational bands. Drift is an investigation signal, not an automatic retraining trigger.
- **CP-04 — Model/decision monitoring:** prediction volume, APPROVE/REVIEW/BLOCK mix, high-risk rate, calibrated-probability summaries, and threshold-proximity counts.
- **CP-05 — Prometheus exposure:** live runtime, data-quality, and model-decision telemetry at `/metrics`. PSI remains a batch/scheduled evaluator rather than a falsely advertised live metric.
- **CP-06 — Docker packaging:** Python 3.12 slim image, non-root application user, liveness health check, Compose smoke-test definition, and explicit readiness boundary.
- **CP-07 — GitHub Actions:** clean Linux install, package import, Ruff, full pytest suite, Docker build/start/health/liveness/readiness/non-root gates.

## Verified cloud gate

The Day-06 branch was verified by GitHub Actions on a clean Ubuntu runner. Both the Python quality job and Docker build/smoke-test job completed successfully. The Docker gate confirmed that the image builds, starts, becomes healthy, answers `/health`, preserves the intentionally unconfigured readiness response, and runs as a non-root user.

## Frozen model contract

Day 06 does not alter the Day-03 model contract:

- model version: `day-03-hgb-balanced-v1`
- calibration: sigmoid
- review threshold: `0.048`
- provisional block threshold: `0.100`
- policy: `day-04-three-way-policy-v1`

The locked test set is not used for Day-06 tuning. Monitoring does not silently retrain, recalibrate, or change decision thresholds.

## Operational boundaries

A green CI workflow proves reproducible package installation, code-quality/test gates, container construction, process liveness, readiness semantics, and non-root execution. It does **not** claim a fully configured production scoring deployment. Redis, PostgreSQL, trained model artifacts, secrets, and production dependency injection remain explicit deployment responsibilities.

Likewise, the current monitoring layer should not be described as ground-truth fraud-performance monitoring: live labels are not assumed to be available. Decision telemetry and drift signals support observation and investigation; they do not establish live precision, recall, or business-loss performance.

## Day-06 exit criteria

Day 06 is complete when the monitoring modules, Prometheus endpoint, Docker packaging, CI workflow, documentation, and successful clean-run verification are all present on the branch. The next phase is Day 07 deployment, recruiter-facing documentation, and final project packaging.
