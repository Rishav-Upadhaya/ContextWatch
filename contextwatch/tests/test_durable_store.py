from __future__ import annotations

from datetime import datetime, timezone

from api.store import DurableStore, ProcessedLog
from core.schema import AnomalyResult, ClassificationResult, NormalizedLog, RCAResult


def _mk_processed(log_id: str, session_id: str, anomaly: bool = False) -> ProcessedLog:
    normalized = NormalizedLog(
        log_id=log_id,
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="agent_test",
        text_for_embedding="sample text",
        metadata={"intent": "test"},
    )
    anomaly_result = AnomalyResult(
        log_id=log_id,
        anomaly_score=0.7 if anomaly else 0.1,
        is_anomaly=anomaly,
        anomaly_type="CONTEXT_POISONING" if anomaly else None,
        confidence=0.8 if anomaly else 0.2,
    )
    classification = ClassificationResult(
        anomaly_type="CONTEXT_POISONING" if anomaly else None,
        confidence=0.8 if anomaly else 1.0,
        method="rule" if anomaly else "none",
        reasoning="test",
    )
    rca = (
        RCAResult(
            root_cause_log_id=log_id,
            causal_chain=[log_id],
            hop_count=0,
            explanation="root",
        )
        if anomaly
        else None
    )
    return ProcessedLog(
        normalized=normalized,
        anomaly=anomaly_result,
        classification=classification,
        explanation="explain" if anomaly else None,
        rca=rca,
    )


def test_durable_store_persists_and_reloads(tmp_path):
    db_path = tmp_path / "contextwatch_test.db"

    store = DurableStore(str(db_path))
    store.upsert(_mk_processed("l1", "s1", anomaly=False))
    store.upsert(_mk_processed("l2", "s1", anomaly=True))
    store.close()

    reloaded = DurableStore(str(db_path))
    assert "l1" in reloaded.all_logs
    assert "l2" in reloaded.all_logs
    assert "l2" in reloaded.anomalies
    assert reloaded.anomalies["l2"].classification.anomaly_type == "CONTEXT_POISONING"
    reloaded.close()


def test_durable_store_latest_in_session(tmp_path):
    db_path = tmp_path / "contextwatch_test_latest.db"
    store = DurableStore(str(db_path))
    store.upsert(_mk_processed("a1", "session-x", anomaly=False))
    store.upsert(_mk_processed("a2", "session-x", anomaly=True))

    latest = store.latest_in_session("session-x")
    assert latest is not None
    assert latest.normalized.log_id in {"a1", "a2"}
    store.close()
