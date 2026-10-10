"""Observability utilities for the FraudShield API."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

LOGGER_NAME = "fraudshield.api"

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class RequestContext:
    """Observability context for one HTTP request."""

    request_id: str
    method: str
    path: str
    started_at: float


def new_request_id() -> str:
    """Return a unique request correlation identifier."""
    return uuid.uuid4().hex


def elapsed_ms(started_at: float) -> float:
    """Return elapsed monotonic time in milliseconds."""
    return (time.perf_counter() - started_at) * 1000.0


def log_event(
    event: str,
    **fields: Any,
) -> None:
    """Write one structured JSON log event."""
    payload = {
        "event": event,
        **fields,
    }

    logger.info(
        json.dumps(
            payload,
            sort_keys=True,
            default=str,
        )
    )