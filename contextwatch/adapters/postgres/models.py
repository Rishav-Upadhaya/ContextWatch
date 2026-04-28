"""Data persistence models (DTOs) for the Postgres adapter."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

@dataclass(frozen=True)
class StoredNormalLog:
    log_id: str
    protocol: str
    normalized_text: str
    embedding: list[float]
    trace_context: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass(frozen=True)
class StoredAnomaly:
    anomaly_id: str
    log_id: str
    anomaly_type: str
    score: float
    confidence: float
    explanation: str
    details: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
