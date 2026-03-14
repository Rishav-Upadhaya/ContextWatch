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


def test_health_endpoint_exists():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code in (200, 500)


def test_ingest_log_without_labels():
    payload = {
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
                    "arguments": {"file_key": "hU2yV3kPzXqN8mD1oL5rT9"},
                    "request_id": 1,
                    "triggered_by": "user_prompt: 'Review dashboard design and suggest improvements'",
                },
            },
        },
    }
    with TestClient(app) as client:
        settings = get_settings()
        headers = {"X-API-Key": settings.API_KEY} if settings.AUTH_ENABLED else {}
        response = client.post("/ingest/log", json=payload, headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["log_id"] == payload["log_id"]
        assert "rule_lane_triggered" in body["data"]
        assert "embedding_lane_score" in body["data"]
        assert "embedding_lane_threshold" in body["data"]
        assert "arbitration_mode" in body["data"]
