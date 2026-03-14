from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate near real-time ingestion at fixed intervals")
    parser.add_argument("--file", type=Path, default=Path("data/synthetic/mixed/mcp_a2a_mixed_logs.jsonl"))
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between log submissions")
    parser.add_argument("--max-logs", type=int, default=100, help="Maximum rows to ingest")
    args = parser.parse_args()

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    sent = 0
    failed = 0
    for row in iter_jsonl(args.file):
        if sent >= args.max_logs:
            break
        try:
            resp = requests.post(f"{args.api_url}/ingest/log", headers=headers, json=row, timeout=15)
            if resp.status_code >= 400:
                failed += 1
            sent += 1
        except Exception:
            failed += 1
            sent += 1
        time.sleep(max(args.interval, 0.0))

    print(json.dumps({"sent": sent, "failed": failed, "interval_seconds": args.interval}, indent=2))


if __name__ == "__main__":
    main()
