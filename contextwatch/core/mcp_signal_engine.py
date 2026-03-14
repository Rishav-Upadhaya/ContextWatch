from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from config.settings import get_settings
from core.mcp_mapping import REAL_EVENT_TO_SUBTYPE, SYNTHETIC_EVENTS, map_subtype_to_legacy
from core.mcp_ml_assist import DistilBERTSignalAssist
from core.schema import MCPFinding, MCPFindingResponse, MCPSessionSummary, NormalizedLog


@dataclass
class _RuleMatch:
    rule_id: str
    subtype: str


_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("R-01", re.compile(r"method not found|code\D*-32601", re.IGNORECASE), "unknown_tool_call"),
    ("R-02", re.compile(r"invalid params|unknown field|code\D*-32602", re.IGNORECASE), "invalid_tool_parameter"),
    ("R-03", re.compile(r"http_status\D*404|not found|deleted", re.IGNORECASE), "resource_not_found"),
    ("R-04", re.compile(r"http_status\D*429|too many requests", re.IGNORECASE), "rate_limit_breach"),
    ("R-05", re.compile(r"http_status\D*401|unauthorized|pat.*expired|revoked", re.IGNORECASE), "auth_failure"),
    ("R-06", re.compile(r"http_status\D*5\d\d", re.IGNORECASE), "upstream_server_error"),
    ("R-07", re.compile(r"elapsed_ms\D*>\s*threshold|latency threshold|figma_api_slow", re.IGNORECASE), "latency_spike"),
    ("R-08", re.compile(r"retry.*back\s*[-_]?\s*off", re.IGNORECASE), "retry_storm_indicator"),
    ("R-09", re.compile(r"ignore\s+(previous|all)\s+instructions|ignore all|forget your|you are now|act as", re.IGNORECASE), "prompt_injection_attempt"),
    ("R-10", re.compile(r"system:|process\.exit|exec\(|__proto__|<script", re.IGNORECASE), "code_injection_attempt"),
    ("R-11", re.compile(r"select\b.*\bfrom\b|drop table|insert into", re.IGNORECASE), "sql_injection_attempt"),
    ("R-12", re.compile(r"recursion_depth|depth\D*9\d", re.IGNORECASE), "input_anomaly"),
]

_ROUTINE_NOISE_EVENTS = {
    "API_CACHE_HIT",
    "API_CACHE_MISS",
    "API_CACHE_EVICT",
    "GC_EVENT",
    "MEM_SNAPSHOT",
    "HEARTBEAT",
    "METRIC_REPORT",
}

_ALWAYS_ESCALATE_SUBTYPES = {
    "prompt_injection_attempt",
    "code_injection_attempt",
    "sql_injection_attempt",
    "auth_failure",
    "unknown_tool_call",
}


class MCPSignalEngine:
    def __init__(self, settings=None, ml_assist: DistilBERTSignalAssist | None = None):
        self.settings = settings or get_settings()
        self._ml_assist = None
        ml_enabled = (
            self.settings.MCP_HYBRID_ENABLED
            and not self.settings.MCP_ML_KILLSWITCH
            and self.settings.MCP_HYBRID_PHASE >= 2
        )
        if ml_assist is not None:
            self._ml_assist = ml_assist
        elif ml_enabled:
            self._ml_assist = DistilBERTSignalAssist(self.settings.MCP_ML_MODEL_NAME)

    def _extract(self, log: NormalizedLog) -> dict:
        md = log.metadata if isinstance(log.metadata, dict) else {}
        params = md.get("params", {}) if isinstance(md.get("params"), dict) else {}
        data = params.get("data", {}) if isinstance(params.get("data"), dict) else {}
        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        return {
            "session_id": log.session_id,
            "log_id": log.log_id,
            "timestamp": log.timestamp,
            "event": str(md.get("event") or data.get("event") or "").upper(),
            "level": str(md.get("level") or params.get("level") or "info").lower(),
            "message": str(md.get("reasoning_step") or data.get("message") or ""),
            "meta": meta,
            "tool": str(md.get("tool_name") or meta.get("tool") or "").strip(),
            "request_id": str(meta.get("request_id") or ""),
        }

    def _signal_1(self, event: str) -> tuple[str | None, bool]:
        if event in SYNTHETIC_EVENTS:
            return "synthetic_test_data", True
        return REAL_EVENT_TO_SUBTYPE.get(event), False

    def _signal_3(self, event: str, message: str, meta: dict) -> list[_RuleMatch]:
        event_text = event
        message_text = message
        meta_text = json.dumps(meta, default=str)
        matches: list[_RuleMatch] = []
        for rule_id, pattern, subtype in _RULES:
            if pattern.search(message_text) or pattern.search(meta_text) or pattern.search(event_text):
                if rule_id == "R-03" and event in {"CLEANUP", "GC_EVENT", "MEM_SNAPSHOT"}:
                    continue
                if rule_id == "R-04" and event == "RATE_LIMIT_WARN":
                    continue
                if rule_id == "R-08" and event == "TOOL_CALL_RETRY_EXEC":
                    continue
                matches.append(_RuleMatch(rule_id=rule_id, subtype=subtype))
        return matches

    def _signal_4(self, tool: str, meta: dict) -> str | None:
        args = meta.get("args") if isinstance(meta.get("args"), dict) else None
        if args is None:
            args = meta.get("arguments") if isinstance(meta.get("arguments"), dict) else {}

        args_text = json.dumps(args, default=str).lower()
        file_key = str(args.get("file_key", "")).lower()
        if file_key and any(token in file_key for token in ("system", "root", "admin", "null", "undefined")):
            return "prompt_injection_attempt"

        if tool == "post_comment":
            msg = str(args.get("message", ""))
            if re.search(r"ignore all|forget your|you are now|act as|system:", msg, re.IGNORECASE):
                return "prompt_injection_attempt"

        if tool == "get_node":
            node_id = str(args.get("node_id", ""))
            if node_id and not re.match(r"^\d+:\d+$", node_id):
                return "input_anomaly"

        if any(isinstance(v, str) and len(v) > 500 for v in args.values()):
            return "input_anomaly"

        if re.search(r"(.{8,})\1\1", args_text):
            return "input_anomaly"

        if re.search(r"ignore all|forget your|you are now|act as|process\.exit|exec\(|<script", args_text, re.IGNORECASE):
            return "prompt_injection_attempt"

        return None

    def _warning_accumulated(self, level: str, tool: str, now: datetime, window: deque[dict]) -> bool:
        if level != "warning":
            return False
        if not tool:
            return False
        threshold = now - timedelta(seconds=60)
        count = 1
        for row in window:
            row_ts = row["timestamp"]
            if row_ts >= threshold and row.get("level") == "warning":
                if row.get("tool") == tool:
                    count += 1
        return count >= 2

    def _is_noise_only(self, event: str, meta: dict, level: str) -> bool:
        if event in _ROUTINE_NOISE_EVENTS:
            return True
        if event == "RATE_LIMIT_CHECK":
            used = int(meta.get("used", 0) or 0)
            limit = int(meta.get("limit", 0) or 0)
            if limit > 0 and used < int(0.8 * limit):
                return True
        if event == "FIGMA_API_SLOW":
            elapsed = int(meta.get("elapsed_ms", 0) or 0)
            if elapsed < 1500:
                return True
        if level == "debug":
            return True
        return False

    def _confidence(self, signals: set[str], level: str, warning_accumulated: bool, has_explicit_error_code: bool) -> str:
        if ("signal_1" in signals and "signal_2" in signals) or ("signal_3" in signals and has_explicit_error_code):
            return "high"
        if ("signal_2" in signals and level == "error") or warning_accumulated or (("signal_4" in signals) and ("signal_3" in signals)):
            return "medium"
        return "low"

    def _recommended_action(self, subtype: str) -> str:
        actions = {
            "unknown_tool_call": "Validate MCP tool manifest and block unregistered tool invocations.",
            "invalid_tool_parameter": "Validate tool schema and reject unsupported arguments before execution.",
            "auth_failure": "Rotate/refresh PAT and verify token scope before retrying requests.",
            "prompt_injection_attempt": "Isolate session, review prompt content, and apply stricter input sanitization.",
            "rate_limit_breach": "Throttle calls and add backoff/jitter to prevent repeated upstream rate-limit failures.",
            "tool_execution_failure": "Inspect tool dependency chain and recover or roll back failed execution state.",
            "orphaned_retry_logic_error": "Fix retry orchestration to require a preceding concrete error.",
            "synthetic_test_data": "Treat this as test data; exclude from production anomaly KPIs.",
        }
        return actions.get(subtype, "Review correlated logs and remediate root cause before replaying the request.")

    def _summary_text(self, density_pct: float, anomalies: int, total: int) -> str:
        if total == 0:
            return "No logs were analyzed."
        if anomalies == 0:
            return "Session appears healthy with no anomalies detected."
        if density_pct > 20.0:
            return "Anomaly density is elevated above healthy baseline; review thresholds and investigate repeated failure patterns."
        return "Session shows isolated anomaly patterns with moderate operational risk; targeted remediation is recommended."

    def _correlate(self, current: dict, window: deque[dict], findings_by_log: dict[str, MCPFinding]) -> tuple[str | None, list[str], list[str]]:
        event = current["event"]
        tool = current["tool"]
        now = current["timestamp"]
        req_id = current.get("request_id")
        correlated: list[str] = []
        context_rule_ids: list[str] = []
        derived_subtype: str | None = None

        if event == "TOOL_CALL_ERROR" and tool:
            latest_received = None
            latest_success = None
            for row in window:
                if row.get("tool") != tool:
                    continue
                if row.get("event") == "TOOL_CALL_RECEIVED":
                    latest_received = row
                if row.get("event") == "TOOL_CALL_SUCCESS":
                    latest_success = row
            if latest_received and (not latest_success or latest_success["timestamp"] < latest_received["timestamp"]):
                derived_subtype = derived_subtype or "tool_execution_failure"
                correlated.append(latest_received["log_id"])

        if event == "TOOL_CALL_RETRY":
            has_prior_error = any(row.get("event") in {"FIGMA_API_ERROR", "TOOL_CALL_ERROR"} for row in window)
            if not has_prior_error:
                derived_subtype = "orphaned_retry_logic_error"
                context_rule_ids.append("C-02")

        if event == "TOOL_CALL_RECEIVED" and tool:
            threshold = now - timedelta(seconds=20)
            calls = [
                row
                for row in window
                if row.get("event") == "TOOL_CALL_RECEIVED" and row.get("tool") == tool and row["timestamp"] >= threshold
            ]
            recent_503 = [
                row
                for row in window
                if row.get("event") == "FIGMA_API_ERROR"
                and int(row.get("meta", {}).get("http_status", 0) or 0) == 503
                and row["timestamp"] >= threshold
            ]
            if len(calls) >= 2 and len(recent_503) >= 2:
                derived_subtype = derived_subtype or "runaway_retry_loop"
                correlated.extend([row["log_id"] for row in calls[-2:]])
                context_rule_ids.append("C-03")

        if event == "RATE_LIMIT_WARN":
            used = int(current["meta"].get("used", 0) or 0)
            limit = int(current["meta"].get("limit", 0) or 0)
            if limit > 0 and used >= int(0.95 * limit):
                derived_subtype = derived_subtype or "rate_limit_breach"
                context_rule_ids.append("C-04")

        if event == "FIGMA_API_ERROR":
            http_status = int(current["meta"].get("http_status", 0) or 0)
            if http_status == 429:
                for row in reversed(window):
                    if now - row["timestamp"] > timedelta(seconds=10):
                        break
                    used = int(row.get("meta", {}).get("used", 0) or 0)
                    limit = int(row.get("meta", {}).get("limit", 0) or 0)
                    if row.get("level") == "warning" and row.get("event") in {"RATE_LIMIT_WARN", "FIGMA_API_WARN"} and limit > 0 and used >= int(0.8 * limit):
                        correlated.append(row["log_id"])
                        derived_subtype = derived_subtype or "rate_limit_breach"
                        if row["log_id"] in findings_by_log:
                            existing = findings_by_log[row["log_id"]]
                            if current["log_id"] not in existing.correlated_log_ids:
                                existing.correlated_log_ids.append(current["log_id"])
                        break

        if event == "TOOL_CALL_SUCCESS" and req_id:
            for row in reversed(window):
                if row.get("request_id") == req_id and row.get("event") == "SCHEMA_VALIDATION_ERROR":
                    derived_subtype = derived_subtype or "inconsistent_state"
                    correlated.append(row["log_id"])
                    break

        return derived_subtype, sorted(set(correlated)), sorted(set(context_rule_ids))

    def _apply_hybrid_policy(self, rule_hit: bool, ml_hit: bool, ml_score: float | None, subtype: str | None) -> tuple[bool, str]:
        if not self.settings.MCP_HYBRID_ENABLED:
            return (rule_hit or ml_hit) and subtype is not None, "rule_only"

        phase = self.settings.MCP_HYBRID_PHASE
        if phase <= 1:
            return rule_hit and subtype is not None, "rule_only"

        if phase == 2:
            return rule_hit and subtype is not None, "rule_only"

        if phase == 3:
            if rule_hit and ml_hit:
                return subtype is not None, "rule_ml"
            if rule_hit:
                return subtype is not None, "rule_only"
            return False, "ml_review_queue"

        ml_pass = ml_hit and ml_score is not None and ml_score >= self.settings.MCP_ML_PROMOTION_THRESHOLD
        if rule_hit and ml_hit:
            return subtype is not None, "rule_ml"
        if rule_hit:
            return subtype is not None, "rule_only"
        if ml_pass:
            return subtype is not None, "ml_only"
        return False, "ml_review_queue"

    def analyze(self, logs: list[NormalizedLog]) -> MCPFindingResponse:
        session_windows: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=10))
        findings: list[MCPFinding] = []
        findings_by_log: dict[str, MCPFinding] = {}
        policy_counts: Counter[str] = Counter()

        for log in logs:
            if log.protocol != "MCP":
                continue

            row = self._extract(log)
            event = row["event"]
            level = row["level"]
            message = row["message"]
            meta = row["meta"]
            tool = row["tool"]
            session_id = row["session_id"]
            window = session_windows[session_id]

            signals: set[str] = set()
            subtype: str | None = None
            synthetic = False
            context_rule_ids: list[str] = []
            ml_score: float | None = None
            ml_label: str | None = None

            signal1_subtype, is_synthetic = self._signal_1(event)
            if signal1_subtype:
                signals.add("signal_1")
                subtype = signal1_subtype
                synthetic = is_synthetic

            rule_matches = self._signal_3(event, message, meta)
            if rule_matches:
                signals.add("signal_3")
                if subtype is None:
                    subtype = rule_matches[0].subtype

            signal4_subtype = self._signal_4(tool, meta)
            if signal4_subtype:
                signals.add("signal_4")
                if subtype is None:
                    subtype = signal4_subtype

            if self._ml_assist is not None and not self.settings.MCP_ML_KILLSWITCH:
                ml_text = f"event={event} message={message} meta={json.dumps(meta, default=str)}"
                ml_result = self._ml_assist.score(ml_text)
                ml_score = ml_result.score
                ml_label = ml_result.label
                if ml_score >= self.settings.MCP_ML_PROMOTION_THRESHOLD and ml_label:
                    signals.add("signal_4")
                    if subtype is None:
                        subtype = ml_label

            warning_accumulated = self._warning_accumulated(level, tool, row["timestamp"], window)
            if level == "error" or warning_accumulated:
                signals.add("signal_2")

            if level == "warning" and "signal_1" not in signals and "signal_3" not in signals and "signal_4" not in signals and not warning_accumulated:
                signals.clear()

            if self._is_noise_only(event, meta, level):
                has_always_escalate = any(match.subtype in _ALWAYS_ESCALATE_SUBTYPES for match in rule_matches)
                if subtype in _ALWAYS_ESCALATE_SUBTYPES:
                    has_always_escalate = True
                if not has_always_escalate:
                    signals.clear()
                    subtype = None

            has_explicit_error_code = bool(re.search(r"code\D*-3260[12]|http_status", message + json.dumps(meta, default=str), re.IGNORECASE))

            derived_subtype, correlated, context_rule_ids = self._correlate(row, window, findings_by_log)
            if derived_subtype:
                subtype = subtype or derived_subtype
                signals.add("signal_3")
                if correlated:
                    signals.add("signal_2")

            rule_hit = any(sig in signals for sig in {"signal_1", "signal_2", "signal_3"})
            ml_hit = "signal_4" in signals
            should_report, promotion_source = self._apply_hybrid_policy(rule_hit, ml_hit, ml_score, subtype)
            if should_report:
                policy_counts["promoted"] += 1
            else:
                policy_counts[promotion_source] += 1
            if should_report:
                confidence = self._confidence(signals, level, warning_accumulated, has_explicit_error_code)
                if subtype in {"prompt_injection_attempt", "auth_failure", "unknown_tool_call"}:
                    confidence = "high"
                if promotion_source == "rule_ml" and confidence == "low":
                    confidence = "medium"
                if promotion_source == "ml_only":
                    confidence = "medium" if (ml_score or 0.0) < 0.92 else "high"

                finding = MCPFinding(
                    finding_id=str(uuid4()),
                    session_id=session_id,
                    log_id=row["log_id"],
                    timestamp=row["timestamp"],
                    anomaly_type=subtype,
                    confidence=confidence,
                    signals_triggered=sorted(list(signals)),
                    event=event,
                    level=level if level in {"debug", "info", "warning", "error"} else "info",
                    summary=f"{subtype.replace('_', ' ')} detected for event {event}.",
                    detail=f"message={message[:220]} | meta={json.dumps(meta, default=str)[:240]}",
                    recommended_action=self._recommended_action(subtype),
                    correlated_log_ids=correlated,
                    context_rule_ids=context_rule_ids,
                    legacy_anomaly_type=map_subtype_to_legacy(subtype),
                    subtype=subtype,
                    synthetic_test_data=synthetic,
                    ml_score=ml_score,
                    ml_label=ml_label,
                    promotion_source=promotion_source,
                    policy_decision="promoted",
                )
                findings.append(finding)
                findings_by_log[finding.log_id] = finding

                if subtype == "prompt_injection_attempt" and tool == "post_comment":
                    prior_ids = [x["log_id"] for x in window if x.get("tool") == "post_comment"]
                    finding.correlated_log_ids = sorted(set(finding.correlated_log_ids + prior_ids))

            window.append(row)

        if logs:
            primary_session = Counter([x.session_id for x in logs]).most_common(1)[0][0]
        else:
            primary_session = ""

        by_type = Counter([x.anomaly_type for x in findings])
        by_confidence = Counter([x.confidence for x in findings])
        by_level = Counter([str(x.metadata.get("level", "info")).lower() for x in logs if isinstance(x.metadata, dict)])
        anomalous_log_ids = {x.log_id for x in findings}
        non_anomalous_warnings = 0
        non_anomalous_errors = 0
        for x in logs:
            level = str(x.metadata.get("level", "info")).lower() if isinstance(x.metadata, dict) else "info"
            if x.log_id in anomalous_log_ids:
                continue
            if level == "warning":
                non_anomalous_warnings += 1
            if level == "error":
                non_anomalous_errors += 1

        total_logs = len(logs)
        anomaly_count = len(findings)
        density_pct = (anomaly_count / total_logs * 100.0) if total_logs else 0.0

        risk_rank = {"high": 3, "medium": 2, "low": 1}
        highest = None
        if findings:
            highest = sorted(findings, key=lambda f: (risk_rank.get(f.confidence, 0), len(f.signals_triggered)), reverse=True)[0].finding_id

        summary = MCPSessionSummary(
            session_id=primary_session,
            total_logs_analysed=total_logs,
            anomalies_found=anomaly_count,
            anomaly_density_pct=round(density_pct, 2),
            by_type=dict(by_type),
            by_confidence={"high": by_confidence.get("high", 0), "medium": by_confidence.get("medium", 0), "low": by_confidence.get("low", 0)},
            by_level={
                "error": int(by_level.get("error", 0)),
                "warning": int(by_level.get("warning", 0)),
                "info": int(by_level.get("info", 0)),
                "debug": int(by_level.get("debug", 0)),
            },
            warning_logs=int(by_level.get("warning", 0)),
            error_logs=int(by_level.get("error", 0)),
            non_anomalous_warnings=non_anomalous_warnings,
            non_anomalous_errors=non_anomalous_errors,
            policy_stats=dict(policy_counts),
            highest_risk_finding_id=highest,
            assessment=self._summary_text(density_pct, anomaly_count, total_logs),
        )

        return MCPFindingResponse(findings=findings, session_summary=summary)
