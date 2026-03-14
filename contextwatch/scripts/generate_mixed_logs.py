from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_ts(item: dict) -> datetime:
    if item.get("protocol") == "MCP":
        params = item.get("params", {}) if isinstance(item.get("params"), dict) else {}
        data = params.get("data", {}) if isinstance(params.get("data"), dict) else {}
        return datetime.fromisoformat(data["timestamp"])
    return datetime.fromisoformat(item["timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed MCP+A2A connected dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--take-mcp-normal", type=int, default=20_000)
    parser.add_argument("--take-mcp-anomaly", type=int, default=4_000)
    parser.add_argument("--take-a2a-normal", type=int, default=6_000)
    parser.add_argument("--take-a2a-anomaly", type=int, default=2_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic/mixed"))
    args = parser.parse_args()

    random.seed(args.seed)

    mcp_normal = load_jsonl(args.data_dir / "mcp/mcp_normal_logs.jsonl")
    mcp_anomaly = load_jsonl(args.data_dir / "mcp/mcp_anomaly_logs.jsonl")
    a2a_normal = load_jsonl(args.data_dir / "a2a/a2a_normal_logs.jsonl")
    a2a_anomaly = load_jsonl(args.data_dir / "a2a/a2a_anomaly_logs.jsonl")

    mixed = (
        random.sample(mcp_normal, min(args.take_mcp_normal, len(mcp_normal)))
        + random.sample(mcp_anomaly, min(args.take_mcp_anomaly, len(mcp_anomaly)))
        + random.sample(a2a_normal, min(args.take_a2a_normal, len(a2a_normal)))
        + random.sample(a2a_anomaly, min(args.take_a2a_anomaly, len(a2a_anomaly)))
    )

    random.shuffle(mixed)
    mixed.sort(key=parse_ts)

    write_jsonl(args.out_dir / "mcp_a2a_mixed_logs.jsonl", mixed)
    print(f"Generated mixed logs: {len(mixed)}")


if __name__ == "__main__":
    main()
