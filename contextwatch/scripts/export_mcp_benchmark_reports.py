from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from core.mcp_signal_engine import MCPSignalEngine
from core.normalizer import LogNormalizer


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent

    logs_path = workspace_root / "system_test" / "mcp_benchmark_logs.jsonl"
    key_path = workspace_root / "system_test" / "mcp_benchmark_answer_key.jsonl"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logs = _load_jsonl(logs_path)
    answer_rows = _load_jsonl(key_path)

    normalizer = LogNormalizer()
    engine = MCPSignalEngine()
    normalized_logs = [normalizer.normalize(row) for row in logs]
    result = engine.analyze(normalized_logs)

    predicted_by_log_id = {finding.log_id: finding for finding in result.findings}
    promotion_breakdown = Counter(f.promotion_source for f in result.findings)
    expected_anomaly_log_ids = {row["log_id"] for row in answer_rows if row.get("label") == "anomaly"}
    predicted_anomaly_log_ids = set(predicted_by_log_id)

    true_positive_ids = expected_anomaly_log_ids & predicted_anomaly_log_ids
    false_positive_ids = predicted_anomaly_log_ids - expected_anomaly_log_ids
    false_negative_ids = expected_anomaly_log_ids - predicted_anomaly_log_ids

    per_session = defaultdict(lambda: {"expected": 0, "predicted": 0, "tp": 0, "fp": 0, "fn": 0})
    for row in answer_rows:
        session_id = row.get("session_id", "unknown")
        log_id = row["log_id"]
        expected_is_anomaly = row.get("label") == "anomaly"
        predicted_is_anomaly = log_id in predicted_anomaly_log_ids

        if expected_is_anomaly:
            per_session[session_id]["expected"] += 1
        if predicted_is_anomaly:
            per_session[session_id]["predicted"] += 1
        if expected_is_anomaly and predicted_is_anomaly:
            per_session[session_id]["tp"] += 1
        elif (not expected_is_anomaly) and predicted_is_anomaly:
            per_session[session_id]["fp"] += 1
        elif expected_is_anomaly and (not predicted_is_anomaly):
            per_session[session_id]["fn"] += 1

    total_expected = len(expected_anomaly_log_ids)
    total_predicted = len(predicted_anomaly_log_ids)
    tp = len(true_positive_ids)
    fp = len(false_positive_ids)
    fn = len(false_negative_ids)
    precision = (tp / total_predicted) if total_predicted else 1.0
    recall = (tp / total_expected) if total_expected else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary = {
        "benchmark_name": "mcp_benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "logs": str(logs_path.relative_to(workspace_root)),
            "answer_key": str(key_path.relative_to(workspace_root)),
        },
        "totals": {
            "logs": len(logs),
            "expected_anomalies": total_expected,
            "predicted_anomalies": total_predicted,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "score": f"{tp}/{total_expected}",
        },
        "session_summary": {
            session_id: stats for session_id, stats in sorted(per_session.items(), key=lambda item: item[0])
        },
        "mismatch_log_ids": {
            "false_positive": sorted(false_positive_ids),
            "false_negative": sorted(false_negative_ids),
        },
        "promotion_source_breakdown": dict(sorted(promotion_breakdown.items(), key=lambda item: item[0])),
        "engine_session_summary": result.session_summary.model_dump(mode="json"),
    }

    summary_path = reports_dir / "mcp_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    session_csv_path = reports_dir / "mcp_benchmark_session_matrix.csv"
    with session_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "expected", "predicted", "tp", "fp", "fn", "match_status"])
        for session_id, stats in sorted(per_session.items(), key=lambda item: item[0]):
            is_match = stats["fp"] == 0 and stats["fn"] == 0
            writer.writerow([
                session_id,
                stats["expected"],
                stats["predicted"],
                stats["tp"],
                stats["fp"],
                stats["fn"],
                "PASS" if is_match else "FAIL",
            ])

    comparison_csv_path = reports_dir / "mcp_benchmark_log_comparison.csv"
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "log_id",
            "session_id",
            "expected_label",
            "expected_anomaly_type",
            "predicted_label",
            "predicted_anomaly_type",
            "predicted_legacy_type",
            "confidence",
            "promotion_source",
            "policy_decision",
            "ml_score",
            "ml_label",
            "context_rule_ids",
            "signals",
            "match_status",
        ])
        for row in answer_rows:
            log_id = row["log_id"]
            expected_label = row.get("label", "normal")
            expected_type = row.get("anomaly_type", "")
            finding = predicted_by_log_id.get(log_id)
            predicted_label = "anomaly" if finding else "normal"
            predicted_type = finding.anomaly_type if finding else ""
            predicted_legacy = finding.legacy_anomaly_type if finding else ""
            confidence = finding.confidence if finding else ""
            promotion_source = finding.promotion_source if finding else ""
            policy_decision = finding.policy_decision if finding else ""
            ml_score = finding.ml_score if finding else ""
            ml_label = finding.ml_label if finding else ""
            context_rule_ids = "|".join(finding.context_rule_ids) if finding else ""
            signals = "|".join(finding.signals_triggered) if finding else ""
            match_status = "PASS" if expected_label == predicted_label else "FAIL"
            writer.writerow([
                log_id,
                row.get("session_id", "unknown"),
                expected_label,
                expected_type,
                predicted_label,
                predicted_type,
                predicted_legacy,
                confidence,
                promotion_source,
                policy_decision,
                ml_score,
                ml_label,
                context_rule_ids,
                signals,
                match_status,
            ])

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {session_csv_path}")
    print(f"Wrote: {comparison_csv_path}")


if __name__ == "__main__":
    main()
