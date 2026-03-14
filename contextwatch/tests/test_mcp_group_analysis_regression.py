from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from config.settings import Settings
from core.classifier import AnomalyClassifier
from core.detector import AnomalyDetector
from core.normalizer import LogNormalizer


class DummyEmbedder:
    def compute_anomaly_score(self, log):
        return 0.65


class DetectorSettings:
    ANOMALY_THRESHOLD = 0.20
    ANOMALY_THRESHOLD_MCP = 0.20
    ANOMALY_THRESHOLD_A2A = 0.20
    MIN_BASELINE_LOGS = 1


ANOMALY_EVENTS = {
    "TOOL_HALLUCINATION": {4, 14, 23, 31, 38, 45},
    "CONTEXT_POISONING": {6, 16, 25, 32, 39, 46},
    "REGISTRY_OVERFLOW": {8, 18, 27, 34, 41, 48},
    "DELEGATION_CHAIN_FAILURE": {11, 20, 29, 36, 43},
}


def _event_for(i: int) -> tuple[str, str, str]:
    if i in ANOMALY_EVENTS["TOOL_HALLUCINATION"]:
        return "TOOL_HALLUCINATION", "error", "Unsupported or invented MCP tool usage"
    if i in ANOMALY_EVENTS["CONTEXT_POISONING"]:
        return "CONTEXT_POISONING", "warning", "Prompt or context injection signature detected"
    if i in ANOMALY_EVENTS["REGISTRY_OVERFLOW"]:
        return "REGISTRY_OVERFLOW", "error", "Registry capacity limit exceeded"
    if i in ANOMALY_EVENTS["DELEGATION_CHAIN_FAILURE"]:
        return "DELEGATION_CHAIN_FAILURE", "error", "Delegation chain broke between dependent calls"

    benign = {
        1: ("SERVER_INIT", "info", "Server ready"),
        2: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        3: ("FIGMA_API_REQUEST", "debug", "Fetching metadata"),
        5: ("TOOL_CALL_SUCCESS", "info", "Tool succeeded"),
        7: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        9: ("HEALTH_CHECK", "debug", "Cache pruned"),
        10: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        12: ("METRIC_UPDATE", "debug", "Average latency"),
        13: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        15: ("TOOL_CALL_SUCCESS", "info", "Node retrieval success"),
        17: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        19: ("TOOL_CALL_SUCCESS", "info", "Comment posted"),
        21: ("API_LOG", "debug", "API call 200 OK"),
        22: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        24: ("TOOL_CALL_SUCCESS", "info", "Schema synced"),
        26: ("RESOURCE_STAT", "debug", "Memory stable"),
        28: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        30: ("TOOL_CALL_SUCCESS", "info", "Components listed"),
        33: ("HEARTBEAT", "debug", "Connection healthy"),
        35: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        37: ("TOOL_CALL_SUCCESS", "info", "Node properties retrieved"),
        40: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        42: ("TOOL_CALL_SUCCESS", "info", "Asset exported"),
        44: ("CLEANUP", "debug", "Temp files deleted"),
        47: ("TOOL_CALL_RECEIVED", "info", "Tool call received"),
        49: ("TOOL_CALL_SUCCESS", "info", "Truncated success"),
        50: ("SERVER_SHUTDOWN", "info", "Graceful shutdown"),
    }
    return benign[i]


def _raw_mcp_log(i: int) -> dict:
    event, level, message = _event_for(i)
    ts = datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc).isoformat()
    return {
        "log_id": str(uuid4()),
        "protocol": "MCP",
        "session": {
            "id": "mcp-sess-55",
            "host": "VS Code Agent",
            "server": "figma-mcp-server v1.4.2",
            "connected_at": ts,
            "transport": "websocket",
        },
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": level,
            "logger": "figma-mcp-server",
            "data": {
                "timestamp": ts,
                "event": event,
                "message": message,
                "meta": {"tool": "get_file", "request_id": i},
            },
        },
    }


def test_mcp_group_analysis_expected_anomaly_counts_and_types():
    normalizer = LogNormalizer()
    detector = AnomalyDetector(DummyEmbedder(), DetectorSettings())
    classifier = AnomalyClassifier(Settings(LLM_API_KEY=""))

    anomalies = 0
    type_counts = {
        "TOOL_HALLUCINATION": 0,
        "CONTEXT_POISONING": 0,
        "REGISTRY_OVERFLOW": 0,
        "DELEGATION_CHAIN_FAILURE": 0,
    }

    for i in range(1, 51):
        normalized = normalizer.normalize(_raw_mcp_log(i))
        anomaly = detector.detect(normalized)
        classified = classifier.classify(normalized, anomaly, [])

        if anomaly.is_anomaly:
            anomalies += 1
            assert classified.anomaly_type is not None
            type_counts[classified.anomaly_type] += 1
        else:
            assert classified.anomaly_type is None

    assert anomalies == 23
    assert type_counts == {
        "TOOL_HALLUCINATION": 6,
        "CONTEXT_POISONING": 6,
        "REGISTRY_OVERFLOW": 6,
        "DELEGATION_CHAIN_FAILURE": 5,
    }
