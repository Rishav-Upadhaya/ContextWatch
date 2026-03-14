from __future__ import annotations

import json
from pathlib import Path

from core.mcp_signal_engine import MCPSignalEngine
from core.normalizer import LogNormalizer


def test_mcp_benchmark_matches_answer_key():
    base = Path(__file__).resolve().parents[2] / "system_test"
    logs_path = base / "mcp_benchmark_logs.jsonl"
    answer_path = base / "mcp_benchmark_answer_key.jsonl"

    assert logs_path.exists(), f"Missing benchmark logs: {logs_path}"
    assert answer_path.exists(), f"Missing benchmark answer key: {answer_path}"

    raw_logs = [json.loads(x) for x in logs_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    answer_rows = [json.loads(x) for x in answer_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    expected_anomaly_ids = {row["log_id"] for row in answer_rows if row.get("label") == "anomaly"}

    normalizer = LogNormalizer()
    engine = MCPSignalEngine()
    normalized = [normalizer.normalize(row) for row in raw_logs]
    out = engine.analyze(normalized)

    predicted_anomaly_ids = {f.log_id for f in out.findings}

    assert predicted_anomaly_ids == expected_anomaly_ids
    assert out.session_summary.anomalies_found == len(expected_anomaly_ids)
