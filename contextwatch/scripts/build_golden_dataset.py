from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


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


def mark(rows: list[dict], label: int) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        copied = dict(row)
        copied["ground_truth_label"] = label
        output.append(copied)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build golden dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    random.seed(args.seed)
    mcp_normal = load_jsonl(args.data_dir / "synthetic/mcp/mcp_normal_logs.jsonl")
    mcp_anomaly = load_jsonl(args.data_dir / "synthetic/mcp/mcp_anomaly_logs.jsonl")
    a2a_normal = load_jsonl(args.data_dir / "synthetic/a2a/a2a_normal_logs.jsonl")
    a2a_anomaly = load_jsonl(args.data_dir / "synthetic/a2a/a2a_anomaly_logs.jsonl")

    sampled_normal = random.sample(mcp_normal, 9_100) + random.sample(a2a_normal, 900)

    mcp_by_type: dict[str, list[dict]] = {}
    for row in mcp_anomaly:
        mcp_by_type.setdefault(row["anomaly_type"], []).append(row)
    a2a_by_type: dict[str, list[dict]] = {}
    for row in a2a_anomaly:
        a2a_by_type.setdefault(row["anomaly_type"], []).append(row)

    sampled_anomaly: list[dict] = []
    for anomaly_type in sorted(mcp_by_type):
        sampled_anomaly.extend(random.sample(mcp_by_type[anomaly_type], 227))
        sampled_anomaly.extend(random.sample(a2a_by_type[anomaly_type], 23))

    assert len(sampled_normal) == 10_000
    assert len(sampled_anomaly) == 1_000

    golden_dir = args.data_dir / "golden_dataset"
    write_jsonl(golden_dir / "golden_normal.jsonl", mark(sampled_normal, 0))
    write_jsonl(golden_dir / "golden_anomalies.jsonl", mark(sampled_anomaly, 1))
    print("Wrote golden dataset", len(sampled_normal), len(sampled_anomaly))


if __name__ == "__main__":
    main()
