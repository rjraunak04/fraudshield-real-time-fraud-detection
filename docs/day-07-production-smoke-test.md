# Day 07 CP-04 — End-to-End Production Smoke Test

This checkpoint verifies the complete transaction-processing contract without reopening the locked test set or reporting deterministic smoke-test values as model performance.

The automated smoke path exercises:

`raw event → past-only online features → scorer → calibration → three-way decision → durable decision persistence → online state commit → monitoring → Prometheus exposition`

The test uses isolated infrastructure substitutes for deterministic CI execution: SQLite stands in for the SQL persistence contract, the in-memory state store implements the same online-state interface used by Redis, and a fixed probability model plus identity calibrator exercise serving behavior without retraining or loading the frozen model artifact.

Assertions cover the REVIEW route at probability 0.06, the frozen review threshold 0.048, provisional block threshold 0.100, model/policy version persistence, post-success state mutation, and monitoring counters.

This is an infrastructure and integration smoke test. It is not evidence of predictive performance, PostgreSQL/Redis network behavior, or cloud deployment. The production Compose topology remains responsible for the real PostgreSQL and Redis services.
