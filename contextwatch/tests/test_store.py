from __future__ import annotations

from datetime import datetime, timedelta, timezone

from api.store import InMemoryStore, ProcessedLog
from core.schema import AnomalyResult, ClassificationResult, NormalizedLog


def _mk_processed(log_id: str, session_id: str, ts: datetime) -> ProcessedLog:
    normalized = NormalizedLog(
        log_id=log_id,
        session_id=session_id,
        timestamp=ts,
        protocol="MCP",
        agent_id="agent_1",
        text_for_embedding="x",
        metadata={},
    )
    anomaly = AnomalyResult(
        log_id=log_id,
        anomaly_score=0.2,
        is_anomaly=False,
        anomaly_type=None,
        confidence=0.1,
    )
    classification = ClassificationResult(
        anomaly_type=None,
        confidence=1.0,
        method="none",
        reasoning="ok",
    )
    return ProcessedLog(
        normalized=normalized,
        anomaly=anomaly,
        classification=classification,
        explanation=None,
        rca=None,
    )


def test_latest_in_session_returns_most_recent():
    store = InMemoryStore()
    t0 = datetime.now(timezone.utc)
    store.upsert(_mk_processed("l1", "s1", t0))
    store.upsert(_mk_processed("l2", "s1", t0 + timedelta(seconds=5)))
    store.upsert(_mk_processed("l3", "s2", t0 + timedelta(seconds=7)))

    latest = store.latest_in_session("s1")
    assert latest is not None
    assert latest.normalized.log_id == "l2"


def test_latest_in_session_respects_exclude_log_id():
    store = InMemoryStore()
    t0 = datetime.now(timezone.utc)
    store.upsert(_mk_processed("l1", "s1", t0))
    store.upsert(_mk_processed("l2", "s1", t0 + timedelta(seconds=5)))

    latest = store.latest_in_session("s1", exclude_log_id="l2")
    assert latest is not None
    assert latest.normalized.log_id == "l1"
