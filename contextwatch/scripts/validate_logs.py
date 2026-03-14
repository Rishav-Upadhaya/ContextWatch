from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.schema import A2ALog, MCPLog


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def validate_file(path: Path, max_samples: int | None = None) -> tuple[int, int]:
    total = 0
    failures = 0
    for row in iter_jsonl(path):
        total += 1
        if max_samples and total > max_samples:
            break
        try:
            protocol = row.get("protocol")
            if protocol is None and isinstance(row.get("logs"), list) and "session" in row:
                protocol = "MCP"
            if protocol is None and {"jsonrpc", "method", "params"}.issubset(row.keys()):
                protocol = "MCP"
            if protocol == "MCP":
                candidate = dict(row)
                if "session" in candidate and isinstance(candidate.get("logs"), list):
                    logs = candidate.get("logs") or []
                    if not logs:
                        raise ValueError("MCP envelope has empty logs array")
                    candidate = dict(logs[-1])
                    candidate["session"] = row["session"]
                    candidate.setdefault("protocol", "MCP")
                MCPLog.model_validate(candidate)
            elif protocol == "A2A":
                A2ALog.model_validate(row)
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
        except (ValidationError, ValueError, TypeError) as exc:
            failures += 1
            print(f"[FAIL] {path.name} row {total}: {exc}")
    print(f"[OK] {path} total={total} failures={failures}")
    return total, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MCP/A2A JSONL logs")
    parser.add_argument("paths", nargs="+", help="JSONL files to validate")
    parser.add_argument("--samples", type=int, default=None, help="Optional max rows per file")
    args = parser.parse_args()

    grand_total = 0
    grand_failures = 0
    for item in args.paths:
        total, failures = validate_file(Path(item), args.samples)
        grand_total += total
        grand_failures += failures

    print(f"Validated={grand_total}, Failures={grand_failures}")
    if grand_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
