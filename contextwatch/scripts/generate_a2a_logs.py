from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from faker import Faker

sys.path.append(str(Path(__file__).resolve().parents[1]))

ANOMALY_TYPES = [
    "TOOL_HALLUCINATION",
    "CONTEXT_POISONING",
    "REGISTRY_OVERFLOW",
    "DELEGATION_CHAIN_FAILURE",
]

AGENT_PAIRS = [
    ("agent_router_01", "agent_ops_01"),
    ("agent_router_01", "agent_finance_01"),
    ("agent_router_01", "agent_risk_01"),
    ("agent_ops_01", "agent_support_01"),
    ("agent_ops_02", "agent_support_02"),
    ("agent_finance_01", "agent_legal_01"),
    ("agent_finance_02", "agent_risk_01"),
    ("agent_sales_01", "agent_marketing_01"),
    ("agent_marketing_01", "agent_sales_02"),
    ("agent_support_01", "agent_humanops_01"),
]

MESSAGE_TYPES = ["task_delegation", "result_return", "clarification", "error_propagation"]

INTENTS = [
    "fetch invoice details",
    "risk assessment",
    "route customer escalation",
    "perform compliance review",
    "draft escalation summary",
    "collect KYC evidence",
    "validate settlement anomaly",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_chain(source: str, target: str, depth: int) -> list[str]:
    chain = [source, target]
    while len(chain) < depth + 1:
        next_candidates = [dst for src, dst in AGENT_PAIRS if src == chain[-1]]
        if not next_candidates:
            next_candidates = [pair[1] for pair in AGENT_PAIRS]
        chain.append(random.choice(next_candidates))
    return chain[: depth + 1]


def realistic_timestamp(start_date: datetime) -> str:
    return (start_date + timedelta(minutes=random.randint(0, 43_200), seconds=random.randint(0, 59))).isoformat()


def make_normal_log(fake: Faker, start_date: datetime, session_id: str | None = None) -> dict:
    source, target = random.choice(AGENT_PAIRS)
    depth = random.randint(0, 3)
    chain = make_chain(source, target, depth)
    intent = random.choice(INTENTS)
    msg_type = random.choices(MESSAGE_TYPES, weights=[58, 28, 12, 2], k=1)[0]
    status = random.choices(["success", "partial", "error"], weights=[88, 8, 4], k=1)[0]
    return {
        "log_id": str(uuid4()),
        "protocol": "A2A",
        "timestamp": realistic_timestamp(start_date),
        "session_id": session_id or str(uuid4()),
        "source_agent": source,
        "target_agent": target,
        "delegation_depth": depth,
        "delegation_chain": chain,
        "message_type": msg_type,
        "message_content": f"{msg_type} for intent: {intent}. upstream={source}, downstream={target}",
        "task_intent": intent,
        "context_carried": {
            "ticket": fake.uuid4(),
            "priority": random.choice(["low", "medium", "high"]),
            "customer_tier": random.choice(["standard", "premium", "enterprise"]),
        },
        "response_status": status,
        "response_content": fake.sentence(nb_words=12) if status != "error" else "dependency unavailable",
        "latency_ms": random.randint(30, 800),
        "is_anomaly": False,
        "anomaly_type": None,
    }


def make_anomaly_log(fake: Faker, start_date: datetime, anomaly_type: str) -> dict:
    row = make_normal_log(fake, start_date)
    row["is_anomaly"] = True
    row["anomaly_type"] = anomaly_type
    if anomaly_type == "TOOL_HALLUCINATION":
        row["target_agent"] = random.choice(["agent_quantum_99", "ghost_agent", "agent_null_endpoint", "agent_void_404"])
        row["message_content"] = "delegating to non-registered specialist agent with unsupported capability"
        row["response_status"] = "error"
        row["response_content"] = "target agent missing in registry"
    elif anomaly_type == "CONTEXT_POISONING":
        row["context_carried"] = {
            "chunk": "duplicate " * 500,
            "contradiction": "approved/rejected",
            "context_tokens": 18000,
        }
        row["response_status"] = random.choice(["partial", "success"])
        row["response_content"] = ""
    elif anomaly_type == "REGISTRY_OVERFLOW":
        row["task_intent"] = "draft legal contract"
        row["target_agent"] = "agent_marketing_01"
        row["message_content"] = "Assign legal drafting to campaign optimization agent due to overloaded candidate registry"
    elif anomaly_type == "DELEGATION_CHAIN_FAILURE":
        row["delegation_depth"] = random.randint(3, 5)
        row["delegation_chain"] = ["agent_router_01"] + [f"agent_chain_{i}" for i in range(row["delegation_depth"])]
        row["message_type"] = "error_propagation"
        row["response_status"] = "error"
        row["response_content"] = "Downstream delegation failed at hop boundary; propagating error"
        row["latency_ms"] = random.randint(900, 2500)
    return row


def validate_invariants(normal_logs: list[dict], anomaly_logs: list[dict]) -> None:
    assert len(normal_logs) > 0
    assert len(anomaly_logs) > 0
    dist = Counter(item["anomaly_type"] for item in anomaly_logs)
    assert set(dist.keys()) == set(ANOMALY_TYPES)
    assert max(dist.values()) - min(dist.values()) <= max(8, int(len(anomaly_logs) * 0.2))
    ids = [x["log_id"] for x in normal_logs + anomaly_logs]
    assert len(ids) == len(set(ids))


def generate_connected_a2a(fake: Faker, start_date: datetime, normal_count: int, anomaly_count: int) -> tuple[list[dict], list[dict]]:
    normal_logs: list[dict] = []
    anomaly_logs: list[dict] = []

    session_total = max(120, (normal_count + anomaly_count) // 10)
    budget = Counter({atype: anomaly_count // len(ANOMALY_TYPES) for atype in ANOMALY_TYPES})

    for _ in range(session_total):
        if len(normal_logs) >= normal_count and len(anomaly_logs) >= anomaly_count:
            break
        session_id = str(uuid4())
        session_len = random.randint(6, 15)
        for _ in range(session_len):
            if len(normal_logs) >= normal_count and len(anomaly_logs) >= anomaly_count:
                break
            row = make_normal_log(fake, start_date, session_id=session_id)
            if len(anomaly_logs) < anomaly_count and random.random() < 0.11:
                available = [k for k, v in budget.items() if v > 0]
                if available:
                    atype = random.choice(available)
                    anomaly_logs.append(make_anomaly_log(fake, start_date, atype))
                    budget[atype] -= 1
                    continue
            if len(normal_logs) < normal_count:
                normal_logs.append(row)

    while len(normal_logs) < normal_count:
        normal_logs.append(make_normal_log(fake, start_date))
    while len(anomaly_logs) < anomaly_count:
        atype = ANOMALY_TYPES[len(anomaly_logs) % len(ANOMALY_TYPES)]
        anomaly_logs.append(make_anomaly_log(fake, start_date, atype))

    return normal_logs[:normal_count], anomaly_logs[:anomaly_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate A2A synthetic logs")
    parser.add_argument("--normal", type=int, default=9_000)
    parser.add_argument("--anomaly", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic/a2a"))
    args = parser.parse_args()

    random.seed(args.seed)
    fake = Faker()
    fake.seed_instance(args.seed)
    start_date = datetime.now(timezone.utc) - timedelta(days=30)

    normal_logs, anomaly_logs = generate_connected_a2a(
        fake=fake,
        start_date=start_date,
        normal_count=args.normal,
        anomaly_count=args.anomaly,
    )

    validate_invariants(normal_logs, anomaly_logs)

    write_jsonl(args.out_dir / "a2a_normal_logs.jsonl", normal_logs)
    write_jsonl(args.out_dir / "a2a_anomaly_logs.jsonl", anomaly_logs)
    print("Generated A2A logs:", len(normal_logs), len(anomaly_logs))


if __name__ == "__main__":
    main()
