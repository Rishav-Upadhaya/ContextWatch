from __future__ import annotations

from datetime import datetime, timezone

from config.settings import Settings
from core.classifier import AnomalyClassifier
from core.schema import AnomalyResult, NormalizedLog


def test_rule_based_tool_hallucination():
    classifier = AnomalyClassifier(Settings(LLM_API_KEY=""))
    log = NormalizedLog(
        log_id="x",
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="agent_finance_01",
        text_for_embedding="find customer email",
        metadata={
            "tool_name": "quantum_solve",
            "intent": "find customer email",
            "response_status": "success",
            "tool_parameters": {},
        },
    )
    anomaly = AnomalyResult(log_id="x", anomaly_score=0.8, is_anomaly=True, anomaly_type=None, confidence=0.9)
    result = classifier.classify(log, anomaly, [])
    assert result.anomaly_type == "TOOL_HALLUCINATION"


def test_rule_based_mcp_rate_limit_warning_maps_to_context_poisoning():
    classifier = AnomalyClassifier(Settings(LLM_API_KEY=""))
    log = NormalizedLog(
        log_id="x2",
        session_id="s2",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="figma-mcp-server",
        text_for_embedding="Rate limit warning | batching recommendation",
        metadata={
            "event": "FIGMA_API_RATE_LIMIT_APPROACHING",
            "level": "warning",
            "tool_name": "get_node",
            "intent": "llm_reasoning: 'Inspecting many nodes for design review'",
            "response_status": "success",
            "response_payload": {"requests_used": 58, "requests_limit": 60},
            "params": {
                "data": {
                    "meta": {"requests_used": 58, "requests_limit": 60, "truncated": True}
                }
            },
        },
    )
    anomaly = AnomalyResult(log_id="x2", anomaly_score=0.86, is_anomaly=True, anomaly_type=None, confidence=0.9)
    result = classifier.classify(log, anomaly, [])
    assert result.anomaly_type == "CONTEXT_POISONING"
