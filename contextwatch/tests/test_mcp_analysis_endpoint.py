from __future__ import annotations

from datetime import datetime, timezone
import socket
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.main import app
from config.settings import get_settings


def _neo4j_available(host: str = "localhost", port: int = 7687, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not available on localhost:7687")


def _mcp_log(event: str, level: str, message: str):
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "protocol": "MCP",
        "log_id": str(uuid4()),
        "session": {
            "id": "mcp-sess-ep",
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
                "meta": {"tool": "get_file", "args": {"file_key": "abc123"}},
            },
        },
    }


def test_mcp_session_endpoint_returns_findings_and_summary():
    payload = {
        "logs": [
            _mcp_log("TOOL_CALL_RECEIVED", "info", "normal received"),
            _mcp_log("UNKNOWN_TOOL", "error", "Method not found code -32601"),
        ]
    }

    with TestClient(app) as client:
        settings = get_settings()
        headers = {"X-API-Key": settings.API_KEY} if settings.AUTH_ENABLED else {}
        response = client.post("/analyze/mcp/session", json=payload, headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        data = body["data"]
        assert "findings" in data
        assert "session_summary" in data
        assert data["session_summary"]["total_logs_analysed"] == 2
        assert data["session_summary"]["anomalies_found"] >= 1
        assert "policy_stats" in data["session_summary"]


def test_stats_overview_returns_required_fields():
    payloads = [
        _mcp_log("UNKNOWN_TOOL", "error", "Method not found code -32601"),
        _mcp_log("UNKNOWN_TOOL", "error", "Method not found code -32601"),
        _mcp_log("TOOL_CALL_RECEIVED", "info", "normal received"),
    ]

    with TestClient(app) as client:
        settings = get_settings()
        headers = {"X-API-Key": settings.API_KEY} if settings.AUTH_ENABLED else {}

        for payload in payloads:
            response = client.post("/ingest/log", json=payload, headers=headers)
            assert response.status_code == 200

        overview_response = client.get("/stats/overview")
        assert overview_response.status_code == 200
        body = overview_response.json()
        assert body["status"] == "success"

        data = body["data"]
        assert "total_logs" in data
        assert "total_anomalies" in data
        assert "latest_data" in data
        assert "recurring_anomalies" in data
        assert isinstance(data["latest_data"], list)
        assert isinstance(data["recurring_anomalies"], list)
        if data["latest_data"]:
            first = data["latest_data"][0]
            assert "rule_lane_triggered" in first
            assert "embedding_lane_score" in first
            assert "embedding_lane_threshold" in first
            assert "arbitration_mode" in first
