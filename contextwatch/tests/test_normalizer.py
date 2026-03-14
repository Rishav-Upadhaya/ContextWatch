from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.normalizer import LogNormalizer


def test_normalize_mcp_valid():
    raw = {
        "protocol": "MCP",
        "log_id": str(uuid4()),
        "session": {
            "id": f"mcp-session-{uuid4().hex[:8]}",
            "host": "Claude Desktop",
            "server": "figma-mcp-server v1.4.2",
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "transport": "stdio",
        },
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "logger": "figma-mcp-server",
            "data": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "TOOL_CALL_RECEIVED",
                "message": "Tool invocation received: get_file",
                "meta": {
                    "tool": "get_file",
                    "arguments": {"file_key": "abc123"},
                    "request_id": 1,
                    "triggered_by": "user_prompt: 'Review dashboard design'",
                },
            },
        },
        "is_anomaly": False,
        "anomaly_type": None,
    }
    out = LogNormalizer().normalize(raw)
    assert out.protocol == "MCP"
    assert out.metadata.get("tool_name") == "get_file"
    assert out.text_for_embedding.strip()


def test_normalize_mcp_without_labels():
    raw = {
        "protocol": "MCP",
        "session": {
            "id": f"mcp-session-{uuid4().hex[:8]}",
            "host": "Claude Desktop",
            "server": "figma-mcp-server v1.4.2",
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "transport": "stdio",
        },
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "debug",
            "logger": "figma-mcp-server",
            "data": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "FIGMA_API_RESPONSE",
                "message": "Figma API responded successfully for get_file.",
                "meta": {
                    "tool": "get_file",
                    "status": 200,
                    "latency_ms": 120,
                },
            },
        },
    }
    out = LogNormalizer().normalize(raw)
    assert out.protocol == "MCP"
    assert out.metadata.get("is_anomaly") is None
    assert out.metadata.get("anomaly_type") is None


def test_normalize_mcp_envelope_shape_uses_last_log():
    session = {
        "id": f"mcp-session-{uuid4().hex[:8]}",
        "host": "Claude Desktop",
        "server": "figma-mcp-server v1.4.2",
        "connected_at": datetime.now(timezone.utc).isoformat(),
        "transport": "stdio",
    }
    raw = {
        "session": session,
        "logs": [
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {
                    "level": "info",
                    "logger": "figma-mcp-server",
                    "data": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event": "SERVER_INIT",
                        "message": "Initialized",
                        "meta": {},
                    },
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {
                    "level": "info",
                    "logger": "figma-mcp-server",
                    "data": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event": "TOOL_CALL_RECEIVED",
                        "message": "Tool invocation received: get_node",
                        "meta": {
                            "tool": "get_node",
                            "arguments": {"node_id": "14:302"},
                            "triggered_by": "llm_reasoning: 'Inspecting node'",
                        },
                    },
                },
            },
        ],
    }
    out = LogNormalizer().normalize(raw)
    assert out.protocol == "MCP"
    assert out.metadata.get("tool_name") == "get_node"


def test_normalize_a2a_without_labels():
    raw = {
        "log_id": str(uuid4()),
        "protocol": "A2A",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": str(uuid4()),
        "source_agent": "agent_a",
        "target_agent": "agent_b",
        "message_type": "task_delegation",
        "message_content": "please fetch invoice",
        "task_intent": "fetch invoice",
        "response_status": "success",
        "latency_ms": 80,
    }
    out = LogNormalizer().normalize(raw)
    assert out.protocol == "A2A"
    assert out.metadata.get("is_anomaly") is None
    assert out.metadata.get("anomaly_type") is None
    assert isinstance(out.metadata.get("delegation_chain"), list)


def test_normalize_invalid_protocol():
    with pytest.raises(Exception):
        LogNormalizer().normalize({"protocol": "UNKNOWN"})
