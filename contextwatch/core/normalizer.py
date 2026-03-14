from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from core.schema import A2ALog, MCPLog, NormalizedLog


class LogNormalizer:
    def _coerce_mcp_raw(self, raw: dict[str, Any]) -> dict[str, Any]:
        if {"timestamp", "session_id", "tool_name", "reasoning_step", "intent"}.issubset(raw.keys()):
            return {
                "log_id": raw.get("log_id"),
                "protocol": "MCP",
                "session": {
                    "id": str(raw.get("session_id")),
                    "host": "legacy-client",
                    "server": "legacy-mcp-adapter",
                    "connected_at": raw.get("timestamp"),
                    "transport": "stdio",
                },
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {
                    "level": "error" if str(raw.get("response_status", "")).lower() in {"error", "timeout"} else "info",
                    "logger": str(raw.get("agent_id", "mcp-legacy-agent")),
                    "data": {
                        "timestamp": raw.get("timestamp"),
                        "event": "LEGACY_MCP_EVENT",
                        "message": str(raw.get("reasoning_step", "")),
                        "meta": {
                            "tool": raw.get("tool_name"),
                            "arguments": raw.get("tool_parameters", {}),
                            "triggered_by": raw.get("intent"),
                            "payload_size_kb": round(float(raw.get("context_window_tokens", 0) or 0) * 4 / 1024, 2),
                            "latency_ms": int(raw.get("latency_ms", 0) or 0),
                            "legacy_context_summary": raw.get("context_summary", "n/a"),
                            "legacy_response_payload": raw.get("response_payload", {}),
                        },
                    },
                },
                "is_anomaly": raw.get("is_anomaly"),
                "anomaly_type": raw.get("anomaly_type"),
            }

        if "session" in raw and isinstance(raw.get("logs"), list):
            logs = raw.get("logs") or []
            if not logs:
                raise ValueError("MCP envelope includes empty logs array")
            last = dict(logs[-1])
            last["session"] = raw["session"]
            last.setdefault("protocol", "MCP")
            if "log_id" in raw and "log_id" not in last:
                last["log_id"] = raw["log_id"]
            if "is_anomaly" in raw and "is_anomaly" not in last:
                last["is_anomaly"] = raw["is_anomaly"]
            if "anomaly_type" in raw and "anomaly_type" not in last:
                last["anomaly_type"] = raw["anomaly_type"]
            return last
        return raw

    def _extract_mcp_features(self, metadata: dict[str, Any]) -> dict[str, Any]:
        params = metadata.get("params", {}) if isinstance(metadata.get("params"), dict) else {}
        data = params.get("data", {}) if isinstance(params.get("data"), dict) else {}
        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}

        event = str(data.get("event", "")).strip()
        message = str(data.get("message", "")).strip()
        tool_name = str(meta.get("tool", "")).strip()
        arguments = meta.get("arguments", {}) if isinstance(meta.get("arguments"), dict) else {}
        triggered_by = str(meta.get("triggered_by", "")).strip()
        response_status = "error" if params.get("level") == "error" else "success"
        if "status" in meta and isinstance(meta.get("status"), int) and int(meta["status"]) >= 400:
            response_status = "error"
        if "timeout_ms" in meta and event.endswith("TIMEOUT"):
            response_status = "timeout"

        latency_ms = meta.get("latency_ms")
        if not isinstance(latency_ms, int):
            latency_ms = 0

        context_summary_parts = [message]
        if triggered_by:
            context_summary_parts.append(triggered_by)
        recommendation = str(meta.get("recommendation", "")).strip()
        if recommendation:
            context_summary_parts.append(recommendation)

        context_window_tokens = 0
        payload_size_kb = meta.get("payload_size_kb")
        if isinstance(payload_size_kb, (int, float)):
            context_window_tokens = int(float(payload_size_kb) * 1024 / 4)

        return {
            "tool_name": tool_name,
            "tool_parameters": arguments,
            "reasoning_step": message,
            "intent": triggered_by or event,
            "context_window_tokens": max(context_window_tokens, 0),
            "context_summary": " | ".join([x for x in context_summary_parts if x]).strip() or "n/a",
            "response_status": response_status,
            "response_payload": meta,
            "latency_ms": max(latency_ms, 0),
            "event": event,
            "logger": params.get("logger"),
            "level": params.get("level"),
        }

    def normalize(self, raw: dict[str, Any]) -> NormalizedLog:
        cleaned = dict(raw)
        cleaned.pop("ground_truth_label", None)
        protocol = cleaned.get("protocol")
        if protocol is None and "session" in cleaned and isinstance(cleaned.get("logs"), list):
            protocol = "MCP"
        if protocol is None and {"jsonrpc", "method", "params"}.issubset(cleaned.keys()):
            protocol = "MCP"

        if protocol == "MCP":
            validated = MCPLog.model_validate(self._coerce_mcp_raw(cleaned))
            extracted = self._extract_mcp_features(validated.model_dump(mode="json"))
            text_for_embedding = " | ".join(
                [
                    str(extracted.get("reasoning_step", "")),
                    str(extracted.get("intent", "")),
                    str(extracted.get("tool_name", "")),
                    str(extracted.get("context_summary", "")),
                ]
            )
            metadata = validated.model_dump(mode="json")
            metadata.update(extracted)
            return NormalizedLog(
                log_id=str(validated.log_id),
                session_id=str(validated.session.id),
                timestamp=validated.params.data.timestamp,
                protocol=validated.protocol,
                agent_id=str(validated.params.logger),
                text_for_embedding=text_for_embedding,
                metadata=metadata,
            )

        if protocol == "A2A":
            validated = A2ALog.model_validate(cleaned)
            text_for_embedding = " | ".join(
                [validated.message_content, validated.task_intent, validated.target_agent]
            )
            metadata = validated.model_dump(mode="json")
            return NormalizedLog(
                log_id=str(validated.log_id),
                session_id=str(validated.session_id),
                timestamp=validated.timestamp,
                protocol=validated.protocol,
                agent_id=validated.source_agent,
                text_for_embedding=text_for_embedding,
                metadata=metadata,
            )

        raise ValidationError.from_exception_data(
            "NormalizedLog",
            [
                {
                    "type": "value_error",
                    "loc": ("protocol",),
                    "msg": "Unsupported or missing protocol",
                    "input": protocol,
                    "ctx": {"error": "Use MCP or A2A"},
                }
            ],
        )
