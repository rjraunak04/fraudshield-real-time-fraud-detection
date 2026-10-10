"""Production dependency bootstrap for FraudShield."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from redis import Redis
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from fraudshield.api.app import create_app
from fraudshield.decision import RiskDecisionEngine
from fraudshield.features.online_pipeline import OnlineFeaturePipeline
from fraudshield.monitoring import (
    DataQualityMonitor,
    FraudMonitoringService,
    ModelDecisionMonitor,
    PrometheusExporter,
    RuntimeMonitor,
)
from fraudshield.persistence import (
    FraudDecisionRepository,
    create_database_engine,
    create_schema,
    create_session_factory,
)
from fraudshield.production.settings import ProductionSettings
from fraudshield.realtime import RealTimeFraudPipeline
from fraudshield.scoring.service import FraudScorer
from fraudshield.state import RedisStateStore


def _load_pickle(path: Path) -> object:
    """Load an explicitly configured trusted model artifact."""
    if not path.is_file():
        raise RuntimeError(f"Required artifact does not exist: {path}")
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Unable to load configured artifact: {path}") from exc


@dataclass(slots=True)
class ProductionContainer:
    """Own long-lived infrastructure and construct request-scoped services."""

    settings: ProductionSettings
    database_engine: Engine
    session_factory: sessionmaker[Session]
    redis_client: Redis
    scorer: FraudScorer
    decision_engine: RiskDecisionEngine
    monitoring: FraudMonitoringService
    prometheus: PrometheusExporter

    def repository(self) -> FraudDecisionRepository:
        """Return a repository backed by a fresh database session."""
        return FraudDecisionRepository(self.session_factory())

    def readiness_check(self) -> None:
        """Verify live database and Redis dependencies."""
        with self.database_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        self.redis_client.ping()

    def close(self) -> None:
        """Release production infrastructure resources."""
        try:
            self.redis_client.close()
        finally:
            self.database_engine.dispose()

    def realtime_service(self) -> RealTimeFraudPipeline:
        """Build a real-time service with its own database session."""
        session = self.session_factory()
        repository = FraudDecisionRepository(session)
        state = RedisStateStore(
            self.redis_client,
            namespace=self.settings.redis_namespace,
        )
        return RealTimeFraudPipeline(
            feature_pipeline=OnlineFeaturePipeline(state),
            scorer=self.scorer,
            decision_engine=self.decision_engine,
            repository=repository,
            session=session,
        )


def build_production_container(
    settings: ProductionSettings,
) -> ProductionContainer:
    """Validate dependencies and assemble the production service graph."""
    model = _load_pickle(settings.model_artifact_path)
    calibrator = _load_pickle(settings.calibrator_artifact_path)
    scorer = FraudScorer(model=model, calibrator=calibrator)

    engine = create_database_engine(settings.database_url)
    create_schema(engine)
    session_factory = create_session_factory(engine)

    redis_client = Redis.from_url(settings.redis_url)
    try:
        redis_client.ping()
    except Exception as exc:
        engine.dispose()
        raise RuntimeError("Redis readiness check failed.") from exc

    runtime = RuntimeMonitor()
    quality = DataQualityMonitor()
    decisions = ModelDecisionMonitor()
    monitoring = FraudMonitoringService(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )
    prometheus = PrometheusExporter(
        runtime_monitor=runtime,
        data_quality_monitor=quality,
        decision_monitor=decisions,
    )

    return ProductionContainer(
        settings=settings,
        database_engine=engine,
        session_factory=session_factory,
        redis_client=redis_client,
        scorer=scorer,
        decision_engine=RiskDecisionEngine(),
        monitoring=monitoring,
        prometheus=prometheus,
    )


def create_production_app(settings: ProductionSettings | None = None):
    """Create a fully wired app or fail fast on invalid production config."""
    container = build_production_container(
        settings or ProductionSettings.from_env()
    )
    app = create_app(
        scorer_dependency=lambda: container.scorer,
        decision_engine=container.decision_engine,
        realtime_dependency=container.realtime_service,
        metrics_repository_dependency=container.repository,
        prometheus_dependency=lambda: container.prometheus,
        monitoring_dependency=lambda: container.monitoring,
    )
    app.state.production_container = container

    @app.on_event("shutdown")
    def close_production_dependencies() -> None:
        container.close()

    return app
