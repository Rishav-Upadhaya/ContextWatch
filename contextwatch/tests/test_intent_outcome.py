from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.anomalies import router as anomalies_router
from api.store import InMemoryStore, ProcessedLog
from core.intent_outcome import compute_intent_outcome_gap
from core.schema import AnomalyResult, ClassificationResult, NormalizedLog


def _make_processed(log_id: str, intent: str, tool_name: str, is_anomaly: bool) -> ProcessedLog:
    normalized = NormalizedLog(
        log_id=log_id,
        session_id="sess-1",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="figma-mcp-server",
        text_for_embedding="x",
        metadata={
            "intent": intent,
            "tool_name": tool_name,
            "response_status": "success",
            "is_anomaly": is_anomaly,
        },
    )
    anomaly = AnomalyResult(
        log_id=log_id,
        anomaly_score=0.8 if is_anomaly else 0.1,
        is_anomaly=is_anomaly,
        anomaly_type="REGISTRY_OVERFLOW" if is_anomaly else None,
        confidence=0.8,
    )
    classification = ClassificationResult(
        anomaly_type="REGISTRY_OVERFLOW" if is_anomaly else None,
        confidence=0.8,
        method="rule" if is_anomaly else "none",
        reasoning="test",
    )
    return ProcessedLog(normalized=normalized, anomaly=anomaly, classification=classification, explanation=None, rca=None)


def test_compute_intent_outcome_gap_low_when_aligned():
    score = compute_intent_outcome_gap(
        {
            "intent": "please review figma design component",
            "tool_name": "get_node",
            "response_status": "success",
        }
    )
    assert score.gap_score <= 0.2
    assert score.coherence_score >= 0.8


def test_compute_intent_outcome_gap_high_when_misaligned():
    score = compute_intent_outcome_gap(
        {
            "intent": "send an email to stakeholders",
            "tool_name": "get_node",
            "response_status": "success",
            "is_anomaly": True,
        }
    )
    assert score.gap_score >= 0.8
    assert score.coherence_score <= 0.2


def test_cognitive_stats_endpoint_returns_metrics():
    app = FastAPI()
    store = InMemoryStore()
    store.upsert(_make_processed("1", "review figma design", "get_file", False))
    store.upsert(_make_processed("2", "send email update", "get_node", True))
    app.state.store = store
    app.include_router(anomalies_router)

    with TestClient(app) as client:
        response = client.get("/stats/cognitive")

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["total_logs"] == 2
    assert 0.0 <= body["avg_intent_outcome_gap"] <= 1.0
    assert 0.0 <= body["avg_thought_action_coherence"] <= 1.0
    assert body["high_gap_count"] >= 1
