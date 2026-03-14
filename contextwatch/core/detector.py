from __future__ import annotations

import math

from config.settings import Settings
from core.embedder import LogEmbedder
from core.schema import AnomalyResult, NormalizedLog


class AnomalyDetector:
    def __init__(self, embedder: LogEmbedder, settings: Settings):
        self.embedder = embedder
        self.threshold = settings.ANOMALY_THRESHOLD
        self.threshold_mcp = getattr(settings, "ANOMALY_THRESHOLD_MCP", None)
        self.threshold_a2a = getattr(settings, "ANOMALY_THRESHOLD_A2A", None)
        self.min_baseline_logs = int(getattr(settings, "MIN_BASELINE_LOGS", 200))
        self.decision_mode = str(getattr(settings, "DETECTOR_DECISION_MODE", "rule_first")).lower()
        self.shadow_mode = bool(getattr(settings, "DETECTOR_SHADOW_MODE", True))

    def _baseline_ready(self) -> bool:
        try:
            collection = getattr(self.embedder, "collection", None)
            if collection is None:
                return True
            return int(collection.count()) >= self.min_baseline_logs
        except Exception:
            return True

    def _get_threshold(self, log: NormalizedLog) -> float:
        if log.protocol == "MCP" and self.threshold_mcp is not None:
            return float(self.threshold_mcp)
        if log.protocol == "A2A" and self.threshold_a2a is not None:
            return float(self.threshold_a2a)
        return float(self.threshold)

    def _mcp_behavior(self, log: NormalizedLog, base_threshold: float, score: float) -> tuple[float, bool, bool]:
        metadata = log.metadata if isinstance(log.metadata, dict) else {}
        level = str(metadata.get("level", "")).lower()
        event = str(metadata.get("event", "")).upper()
        message = str(metadata.get("reasoning_step", "")).lower()

        benign_events = {
            "SERVER_INIT",
            "TOOL_CALL_RECEIVED",
            "FIGMA_API_REQUEST",
            "FIGMA_API_RESPONSE",
            "TOOL_CALL_SUCCESS",
            "HEALTH_CHECK",
            "LATENCY_REPORT",
            "METRIC_UPDATE",
            "API_LOG",
            "RESOURCE_STAT",
            "HEARTBEAT",
            "CLEANUP",
            "SERVER_SHUTDOWN",
        }
        explicit_anomaly_events = {
            "TOOL_HALLUCINATION",
            "CONTEXT_POISONING",
            "REGISTRY_OVERFLOW",
            "DELEGATION_CHAIN_FAILURE",
        }

        has_explicit_bad_event = event in explicit_anomaly_events
        has_explicit_bad_text = any(
            token in message
            for token in [
                "tool_hallucination",
                "context_poisoning",
                "registry_overflow",
                "delegation_chain_failure",
                "non-existent tool",
                "hallucinated",
                "injection",
                "overflow",
                "chain failure",
            ]
        )

        if has_explicit_bad_event or has_explicit_bad_text:
            return base_threshold, True, False

        if event in benign_events:
            return max(base_threshold, 0.99), False, True

        if level in {"info", "debug"}:
            return max(base_threshold, 0.98), False, False

        if level == "warning":
            return max(base_threshold, 0.99), False, False

        return max(base_threshold, 0.95), False, False

    def detect(self, log: NormalizedLog) -> AnomalyResult:
        if not self._baseline_ready():
            return AnomalyResult(
                log_id=log.log_id,
                anomaly_score=0.0,
                is_anomaly=False,
                anomaly_type=None,
                confidence=0.0,
                rule_lane_triggered=False,
                embedding_lane_score=0.0,
                embedding_lane_threshold=None,
                arbitration_mode=self.decision_mode,
            )
        score = self.embedder.compute_anomaly_score(log)
        threshold = self._get_threshold(log)
        force_anomaly = False
        force_normal = False
        if log.protocol == "MCP":
            threshold, force_anomaly, force_normal = self._mcp_behavior(log, threshold, score)

        rule_lane_triggered = force_anomaly
        embedding_lane_triggered = score > threshold

        if force_normal:
            is_anomaly = False
        elif self.decision_mode == "embedding_only":
            is_anomaly = embedding_lane_triggered
        elif self.decision_mode == "rule_or_embedding":
            is_anomaly = rule_lane_triggered or embedding_lane_triggered
        elif self.decision_mode == "rule_and_embedding":
            is_anomaly = rule_lane_triggered and embedding_lane_triggered
        else:
            is_anomaly = rule_lane_triggered or embedding_lane_triggered

        if self.shadow_mode and self.decision_mode == "rule_first":
            is_anomaly = rule_lane_triggered or embedding_lane_triggered

        confidence = 1.0 / (1.0 + math.exp(-((score - threshold) * 10)))
        return AnomalyResult(
            log_id=log.log_id,
            anomaly_score=float(score),
            is_anomaly=is_anomaly,
            anomaly_type=None,
            confidence=float(confidence),
            rule_lane_triggered=rule_lane_triggered,
            embedding_lane_score=float(score),
            embedding_lane_threshold=float(threshold),
            arbitration_mode=self.decision_mode,
        )

    def detect_batch(self, logs: list[NormalizedLog]) -> list[AnomalyResult]:
        return [self.detect(log) for log in logs]
