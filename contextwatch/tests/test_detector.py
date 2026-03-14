from __future__ import annotations

from datetime import datetime, timezone

from core.detector import AnomalyDetector
from core.schema import NormalizedLog


class DummyEmbedder:
    def compute_anomaly_score(self, log: NormalizedLog) -> float:
        return 0.8


class DummyCollection:
    def __init__(self, count: int):
        self._count = count

    def count(self) -> int:
        return self._count


class DummyEmbedderWithCollection(DummyEmbedder):
    def __init__(self, count: int):
        self.collection = DummyCollection(count)


class DummySettings:
    ANOMALY_THRESHOLD = 0.35
    ANOMALY_THRESHOLD_MCP = None
    ANOMALY_THRESHOLD_A2A = None
    MIN_BASELINE_LOGS = 200
    DETECTOR_DECISION_MODE = "rule_first"
    DETECTOR_SHADOW_MODE = True


def test_detector_flags_anomaly():
    detector = AnomalyDetector(DummyEmbedder(), DummySettings())
    log = NormalizedLog(
        log_id="1",
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        protocol="A2A",
        agent_id="agent",
        text_for_embedding="abc",
        metadata={},
    )
    result = detector.detect(log)
    assert result.is_anomaly is True
    assert result.anomaly_score > 0.35
    assert result.embedding_lane_score is not None
    assert result.embedding_lane_threshold is not None
    assert result.arbitration_mode == "rule_first"


def test_detector_uses_protocol_specific_thresholds():
    class SettingsWithProtocolThresholds:
        ANOMALY_THRESHOLD = 0.35
        ANOMALY_THRESHOLD_MCP = 0.90
        ANOMALY_THRESHOLD_A2A = 0.30

    detector = AnomalyDetector(DummyEmbedder(), SettingsWithProtocolThresholds())

    mcp_log = NormalizedLog(
        log_id="m1",
        session_id="s1",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="agent",
        text_for_embedding="abc",
        metadata={},
    )
    a2a_log = NormalizedLog(
        log_id="a1",
        session_id="s2",
        timestamp=datetime.now(timezone.utc),
        protocol="A2A",
        agent_id="agent",
        text_for_embedding="abc",
        metadata={},
    )

    mcp_result = detector.detect(mcp_log)
    a2a_result = detector.detect(a2a_log)

    assert mcp_result.is_anomaly is False  # 0.8 <= 0.9
    assert a2a_result.is_anomaly is True  # 0.8 > 0.3


def test_detector_returns_non_anomaly_when_baseline_not_ready():
    detector = AnomalyDetector(DummyEmbedderWithCollection(count=25), DummySettings())
    log = NormalizedLog(
        log_id="boot-1",
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="agent",
        text_for_embedding="abc",
        metadata={},
    )
    result = detector.detect(log)
    assert result.is_anomaly is False
    assert result.anomaly_score == 0.0


def test_detector_mcp_benign_event_uses_stricter_threshold():
    detector = AnomalyDetector(DummyEmbedder(), DummySettings())
    log = NormalizedLog(
        log_id="mcp-ok-1",
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="figma-mcp-server",
        text_for_embedding="tool call success",
        metadata={
            "level": "info",
            "event": "TOOL_CALL_SUCCESS",
            "reasoning_step": "tool get_node completed successfully",
        },
    )
    result = detector.detect(log)
    assert result.is_anomaly is False


def test_detector_mcp_explicit_bad_event_forces_anomaly():
    detector = AnomalyDetector(DummyEmbedder(), DummySettings())
    log = NormalizedLog(
        log_id="mcp-bad-1",
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="figma-mcp-server",
        text_for_embedding="invalid tool invocation",
        metadata={
            "level": "error",
            "event": "TOOL_HALLUCINATION",
            "reasoning_step": "model attempted non-existent tool",
        },
    )
    result = detector.detect(log)
    assert result.is_anomaly is True
    assert result.rule_lane_triggered is True


def test_detector_rule_and_embedding_mode_requires_both():
    class StrictModeSettings(DummySettings):
        DETECTOR_DECISION_MODE = "rule_and_embedding"
        DETECTOR_SHADOW_MODE = False

    detector = AnomalyDetector(DummyEmbedder(), StrictModeSettings())
    log = NormalizedLog(
        log_id="mcp-and-1",
        session_id="s",
        timestamp=datetime.now(timezone.utc),
        protocol="MCP",
        agent_id="figma-mcp-server",
        text_for_embedding="generic mcp warning",
        metadata={
            "level": "warning",
            "event": "RATE_LIMIT_WARN",
            "reasoning_step": "approaching rate threshold",
        },
    )
    result = detector.detect(log)
    assert result.rule_lane_triggered is False
    assert result.is_anomaly is False
