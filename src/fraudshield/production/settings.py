"""Environment-driven production configuration for FraudShield."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProductionSettings:
    """Validated runtime settings used by the production bootstrap."""

    environment: str
    database_url: str
    redis_url: str
    model_artifact_path: Path
    calibrator_artifact_path: Path
    redis_namespace: str = "fraudshield"

    @classmethod
    def from_env(cls) -> ProductionSettings:
        """Build settings from environment variables without hidden defaults."""
        required = {
            "FRAUDSHIELD_DATABASE_URL": os.getenv("FRAUDSHIELD_DATABASE_URL"),
            "FRAUDSHIELD_REDIS_URL": os.getenv("FRAUDSHIELD_REDIS_URL"),
            "FRAUDSHIELD_MODEL_ARTIFACT": os.getenv("FRAUDSHIELD_MODEL_ARTIFACT"),
            "FRAUDSHIELD_CALIBRATOR_ARTIFACT": os.getenv(
                "FRAUDSHIELD_CALIBRATOR_ARTIFACT"
            ),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required FraudShield settings: " + ", ".join(sorted(missing))
            )

        namespace = os.getenv("FRAUDSHIELD_REDIS_NAMESPACE", "fraudshield").strip()
        if not namespace:
            raise RuntimeError("FRAUDSHIELD_REDIS_NAMESPACE must not be empty.")

        return cls(
            environment=os.getenv("FRAUDSHIELD_ENV", "production"),
            database_url=str(required["FRAUDSHIELD_DATABASE_URL"]),
            redis_url=str(required["FRAUDSHIELD_REDIS_URL"]),
            model_artifact_path=Path(str(required["FRAUDSHIELD_MODEL_ARTIFACT"])),
            calibrator_artifact_path=Path(
                str(required["FRAUDSHIELD_CALIBRATOR_ARTIFACT"])
            ),
            redis_namespace=namespace,
        )
