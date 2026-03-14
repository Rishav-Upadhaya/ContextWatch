from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from faker import Faker

sys.path.append(str(Path(__file__).resolve().parents[1]))

ANOMALY_TYPES = [
    "TOOL_HALLUCINATION",
    "CONTEXT_POISONING",
    "REGISTRY_OVERFLOW",
    "DELEGATION_CHAIN_FAILURE",
]

FIGMA_TOOLS = ["get_file", "get_node", "list_components", "export_asset", "post_comment"]
FAKE_TOOLS = ["quantum_layer_scan", "ghost_component_merge", "void_asset_reconcile", "neural_canvas_patch"]
PROMPTS = [
    "Can you review my dashboard design and suggest improvements?",
    "Inspect this KPI card and check spacing and typography.",
    "Export hero illustration assets for engineering handoff.",
    "List reusable components and identify inconsistencies.",
    "Post accessibility feedback on this node.",
]


def realistic_timestamp(window_start: datetime, max_minutes: int = 8_640) -> str:
    minute_offset = random.randint(0, max_minutes)
    second_offset = random.randint(0, 59)
    ms = random.randint(0, 999)
    return (window_start + timedelta(minutes=minute_offset, seconds=second_offset, milliseconds=ms)).isoformat()


def make_session(ts: str) -> dict:
    return {
        "id": f"mcp-session-{uuid4().hex[:8]}",
        "host": random.choice(["Claude Desktop", "Cursor", "VS Code Agent"]),
        "server": "figma-mcp-server v1.4.2",
        "connected_at": ts,
        "transport": random.choice(["stdio", "websocket"]),
    }


def _file_key() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(random.choice(alphabet) for _ in range(22))


def _base_log(session: dict, ts: str, event: str, level: str, message: str, meta: dict) -> dict:
    return {
        "log_id": str(uuid4()),
        "protocol": "MCP",
        "session": session,
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": level,
            "logger": "figma-mcp-server",
            "data": {
                "timestamp": ts,
                "event": event,
                "message": message,
                "meta": meta,
            },
        },
        "is_anomaly": False,
        "anomaly_type": None,
    }


def make_normal_log(fake: Faker, session: dict, ts: str, request_id: int) -> dict:
    tool = random.choice(FIGMA_TOOLS)
    file_key = _file_key()

    patterns: list[dict] = [
        {
            "event": "SERVER_INIT",
            "level": "info",
            "message": "Figma MCP Server initialized. Capabilities declared: tools, resources, logging.",
            "meta": {
                "tools_registered": FIGMA_TOOLS,
                "figma_api_version": "v1",
                "auth_method": "personal_access_token",
            },
        },
        {
            "event": "TOOL_CALL_RECEIVED",
            "level": "info",
            "message": f"Tool invocation received: {tool}",
            "meta": {
                "tool": tool,
                "arguments": {"file_key": file_key},
                "request_id": request_id,
                "triggered_by": f"user_prompt: '{random.choice(PROMPTS)}'",
            },
        },
        {
            "event": "FIGMA_API_REQUEST",
            "level": "debug",
            "message": f"Outbound request to Figma REST API for {tool}",
            "meta": {
                "endpoint": f"https://api.figma.com/v1/files/{file_key}",
                "headers": {"X-Figma-Token": "••••••••••••fq9z"},
                "timeout_ms": random.choice([8000, 10000, 12000]),
                "tool": tool,
                "arguments": {"file_key": file_key},
                "request_id": request_id,
                "triggered_by": f"llm_reasoning: 'Inspecting design artifact for UI review before response.'",
            },
        },
        {
            "event": "FIGMA_API_RESPONSE",
            "level": "debug",
            "message": f"Figma API responded successfully for {tool}.",
            "meta": {
                "status": 200,
                "latency_ms": random.randint(120, 920),
                "file_name": fake.sentence(nb_words=4).replace(".", ""),
                "version": str(random.randint(1000000000, 1999999999)),
                "pages": ["Cover", "Components", "Dashboard — Light", "Dashboard — Dark"],
                "total_nodes": random.randint(400, 2600),
                "tool": tool,
                "arguments": {"file_key": file_key},
                "request_id": request_id,
                "triggered_by": f"llm_reasoning: 'Need full file context before design recommendations.'",
            },
        },
        {
            "event": "TOOL_CALL_SUCCESS",
            "level": "info",
            "message": f"Tool {tool} completed successfully.",
            "meta": {
                "tool": tool,
                "request_id": request_id,
                "response_content_type": random.choice(["text", "json"]),
                "payload_size_kb": round(random.uniform(8.5, 190.0), 1),
                "truncated": False,
                "latency_ms": random.randint(100, 900),
                "triggered_by": f"llm_reasoning: 'Returning structured output to model context.'",
            },
        },
    ]

    chosen = random.choices(patterns, weights=[8, 32, 20, 20, 20], k=1)[0]
    return _base_log(
        session=session,
        ts=ts,
        event=chosen["event"],
        level=chosen["level"],
        message=chosen["message"],
        meta=chosen["meta"],
    )


def inject_anomaly(base: dict, anomaly_type: str) -> dict:
    row = dict(base)
    row["is_anomaly"] = True
    row["anomaly_type"] = anomaly_type

    data = row.setdefault("params", {}).setdefault("data", {})
    meta = data.setdefault("meta", {})

    if anomaly_type == "TOOL_HALLUCINATION":
        fake_tool = random.choice(FAKE_TOOLS)
        row["params"]["level"] = "error"
        data["event"] = "TOOL_CALL_RECEIVED"
        data["message"] = f"Tool invocation received: {fake_tool}"
        meta.update(
            {
                "tool": fake_tool,
                "arguments": {"hallucinated_key": "x", "undefined_parameter": "y"},
                "triggered_by": "llm_reasoning: 'Attempting latent unregistered capability for canvas rewrite.'",
                "request_id": random.randint(1, 50),
            }
        )

    elif anomaly_type == "CONTEXT_POISONING":
        row["params"]["level"] = "warning"
        data["event"] = "FIGMA_API_RATE_LIMIT_APPROACHING"
        data["message"] = "Figma API rate limit warning: elevated request density in current window."
        meta.update(
            {
                "requests_used": random.randint(38, 60),
                "requests_limit": 60,
                "window_resets_in_seconds": random.randint(10, 50),
                "recommendation": "Batch node requests where possible to avoid ceiling.",
                "truncated": True,
                "payload_size_kb": round(random.uniform(180, 460), 1),
                "triggered_by": "llm_reasoning: 'Collecting broad context for many nodes before summary.'",
            }
        )

    elif anomaly_type == "REGISTRY_OVERFLOW":
        row["params"]["level"] = "info"
        data["event"] = "TOOL_CALL_RECEIVED"
        tool = random.choice(["get_file", "get_node"])  # design tools
        data["message"] = f"Tool invocation received: {tool}"
        meta.update(
            {
                "tool": tool,
                "arguments": {"file_key": _file_key()},
                "request_id": random.randint(1, 50),
                "triggered_by": "user_prompt: 'Please send a Slack escalation to all stakeholders now.'",
            }
        )

    elif anomaly_type == "DELEGATION_CHAIN_FAILURE":
        row["params"]["level"] = "error"
        data["event"] = "TOOL_CALL_ERROR"
        tool = random.choice(["get_node", "post_comment"])
        data["message"] = f"Tool {tool} failed due to downstream timeout."
        meta.update(
            {
                "tool": tool,
                "request_id": random.randint(1, 50),
                "error": "downstream_subagent_timeout",
                "hop": random.randint(2, 5),
                "latency_ms": random.randint(900, 2500),
                "triggered_by": "llm_reasoning: 'Retrying through delegated specialist agent chain.'",
            }
        )

    return row


def generate_connected_sessions(fake: Faker, normal_count: int, anomaly_count: int, start_date: datetime) -> tuple[list[dict], list[dict]]:
    normal_logs: list[dict] = []
    anomaly_logs: list[dict] = []

    session_total = max(250, (normal_count + anomaly_count) // 10)
    anomaly_budget = Counter({atype: anomaly_count // len(ANOMALY_TYPES) for atype in ANOMALY_TYPES})
    remainder = anomaly_count - sum(anomaly_budget.values())
    for i in range(remainder):
        anomaly_budget[ANOMALY_TYPES[i % len(ANOMALY_TYPES)]] += 1

    for _ in range(session_total):
        if len(normal_logs) >= normal_count and len(anomaly_logs) >= anomaly_count:
            break

        session = make_session(realistic_timestamp(start_date))
        session_len = random.randint(10, 22)
        request_counter = random.randint(1, 4)

        for _ in range(session_len):
            if len(normal_logs) >= normal_count and len(anomaly_logs) >= anomaly_count:
                break

            ts = realistic_timestamp(start_date)
            base = make_normal_log(fake, session=session, ts=ts, request_id=request_counter)
            request_counter += random.choice([0, 1])

            can_inject = len(anomaly_logs) < anomaly_count and random.random() < 0.13
            if can_inject:
                available = [k for k, v in anomaly_budget.items() if v > 0]
                if available:
                    atype = random.choice(available)
                    anomaly_logs.append(inject_anomaly(base, atype))
                    anomaly_budget[atype] -= 1
                    continue

            if len(normal_logs) < normal_count:
                normal_logs.append(base)

    while len(normal_logs) < normal_count:
        session = make_session(realistic_timestamp(start_date))
        normal_logs.append(make_normal_log(fake, session=session, ts=realistic_timestamp(start_date), request_id=1))

    while len(anomaly_logs) < anomaly_count:
        session = make_session(realistic_timestamp(start_date))
        base = make_normal_log(fake, session=session, ts=realistic_timestamp(start_date), request_id=1)
        atype = ANOMALY_TYPES[len(anomaly_logs) % len(ANOMALY_TYPES)]
        anomaly_logs.append(inject_anomaly(base, atype))

    return normal_logs[:normal_count], anomaly_logs[:anomaly_count]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def validate_invariants(normal_logs: list[dict], anomaly_logs: list[dict]) -> None:
    total_logs = len(normal_logs) + len(anomaly_logs)
    assert total_logs > 0
    assert len(normal_logs) > 0
    assert len(anomaly_logs) > 0

    dist = Counter(log["anomaly_type"] for log in anomaly_logs)
    assert set(dist.keys()) == set(ANOMALY_TYPES)
    assert max(dist.values()) - min(dist.values()) <= max(10, int(len(anomaly_logs) * 0.2))

    ids = [log["log_id"] for log in normal_logs + anomaly_logs]
    assert len(ids) == len(set(ids))

    for log in normal_logs + anomaly_logs:
        assert all(
            key in log
            for key in [
                "log_id",
                "protocol",
                "session",
                "jsonrpc",
                "method",
                "params",
                "is_anomaly",
                "anomaly_type",
            ]
        )
        assert log["protocol"] == "MCP"
        assert log["jsonrpc"] == "2.0"
        assert log["method"] == "notifications/message"
        datetime.fromisoformat(log["params"]["data"]["timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCP synthetic logs in real-world MCP notifications/message format")
    parser.add_argument("--normal", type=int, default=90_000)
    parser.add_argument("--anomaly", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic/mcp"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    fake = Faker()
    fake.seed_instance(args.seed)
    start_date = datetime.now(timezone.utc) - timedelta(days=30)

    normal_logs, anomaly_logs = generate_connected_sessions(
        fake=fake,
        normal_count=args.normal,
        anomaly_count=args.anomaly,
        start_date=start_date,
    )

    validate_invariants(normal_logs, anomaly_logs)

    write_jsonl(args.out_dir / "mcp_normal_logs.jsonl", normal_logs)
    write_jsonl(args.out_dir / "mcp_anomaly_logs.jsonl", anomaly_logs)

    print(f"Generated MCP normal logs: {len(normal_logs)}")
    print(f"Generated MCP anomaly logs: {len(anomaly_logs)}")
    print("Anomaly distribution:", dict(Counter(log["anomaly_type"] for log in anomaly_logs)))


if __name__ == "__main__":
    main()
