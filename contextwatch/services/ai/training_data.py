"""Utilities for building labeled LogBERT training corpora from repo datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_MAX_NORMAL_SAMPLES = 2048
DEFAULT_MAX_ANOMALY_SAMPLES = 1024
SYSTEM_DATASET_FILES = (
    DATA_ROOT / "training" / "training_train.jsonl",
    DATA_ROOT / "golden_dataset" / "golden_normal.jsonl",
    DATA_ROOT / "golden_dataset" / "golden_anomalies.jsonl",
    DATA_ROOT / "synthetic" / "mcp" / "mcp_normal_logs.jsonl",
    DATA_ROOT / "synthetic" / "mcp" / "mcp_anomaly_logs.jsonl",
    DATA_ROOT / "synthetic" / "a2a" / "a2a_normal_logs.jsonl",
    DATA_ROOT / "synthetic" / "a2a" / "a2a_anomaly_logs.jsonl",
)


@dataclass
class LabeledLogRecord:
    log_id: str
    label: int
    source: str
    anomaly_type: Optional[str]
    payload: dict[str, Any]
    text: str


@dataclass
class TrainingCorpus:
    normal_logs: List[dict[str, Any]]
    anomaly_logs: List[dict[str, Any]]
    labeled_records: List[LabeledLogRecord]

    @property
    def total_logs(self) -> int:
        return len(self.labeled_records)


def _downsample(items: List[Any], limit: Optional[int]) -> List[Any]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]

    step = (len(items) - 1) / float(limit - 1)
    indexes = sorted({min(len(items) - 1, round(i * step)) for i in range(limit)})
    return [items[idx] for idx in indexes]


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            yield cleaned
        return
    if isinstance(value, dict):
        for nested_value in value.values():
            yield from _flatten_strings(nested_value)
        return
    if isinstance(value, list):
        for item in value:
            yield from _flatten_strings(item)


def log_to_training_text(log: dict[str, Any]) -> str:
    """Extract a stable text view from a structured log payload."""
    fields: list[str] = []
    for key in ("protocol", "method", "jsonrpc", "log_id", "anomaly_type"):
        value = log.get(key)
        if isinstance(value, str) and value.strip():
            fields.append(value.strip())
    fields.extend(_flatten_strings(log.get("session")))
    fields.extend(_flatten_strings(log.get("params")))
    if not fields:
        fields.extend(_flatten_strings(log))
    if not fields:
        fields.append(json.dumps(log, sort_keys=True))
    return " ".join(fields)


def infer_log_label(log: dict[str, Any], source: str = "") -> int:
    """Infer binary label from explicit fields or dataset filename."""
    ground_truth = log.get("ground_truth_label")
    if ground_truth is not None:
        try:
            return 1 if int(ground_truth) == 1 else 0
        except (TypeError, ValueError):
            pass

    is_anomaly = log.get("is_anomaly")
    if isinstance(is_anomaly, bool):
        return 1 if is_anomaly else 0

    anomaly_type = log.get("anomaly_type")
    if anomaly_type:
        return 1

    lowered_source = source.lower()
    if "anomal" in lowered_source:
        return 1
    return 0


def load_system_training_corpus(
    extra_logs: Optional[List[dict[str, Any]]] = None,
    max_normal: Optional[int] = DEFAULT_MAX_NORMAL_SAMPLES,
    max_anomaly: Optional[int] = DEFAULT_MAX_ANOMALY_SAMPLES,
) -> TrainingCorpus:
    """Load labeled normal/anomaly logs from repo datasets plus optional user logs."""
    normal_logs: list[dict[str, Any]] = []
    anomaly_logs: list[dict[str, Any]] = []
    normal_records: list[LabeledLogRecord] = []
    anomaly_records: list[LabeledLogRecord] = []

    def append_log(log: dict[str, Any], source: str) -> None:
        label = infer_log_label(log, source)
        text = log_to_training_text(log)
        record = LabeledLogRecord(
            log_id=str(log.get("log_id") or f"{source}:{len(labeled_records)}"),
            label=label,
            source=source,
            anomaly_type=(str(log.get("anomaly_type")) if log.get("anomaly_type") else None),
            payload=log,
            text=text,
        )
        if label == 1:
            anomaly_logs.append(log)
            anomaly_records.append(record)
        else:
            normal_logs.append(log)
            normal_records.append(record)

    for path in SYSTEM_DATASET_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                append_log(json.loads(line), str(path.relative_to(DATA_ROOT)))

    for idx, log in enumerate(extra_logs or []):
        append_log(log, f"user_input:{idx}")

    normal_logs = _downsample(normal_logs, max_normal)
    anomaly_logs = _downsample(anomaly_logs, max_anomaly)
    normal_records = _downsample(normal_records, len(normal_logs))
    anomaly_records = _downsample(anomaly_records, len(anomaly_logs))
    labeled_records = normal_records + anomaly_records

    return TrainingCorpus(
        normal_logs=normal_logs,
        anomaly_logs=anomaly_logs,
        labeled_records=labeled_records,
    )
