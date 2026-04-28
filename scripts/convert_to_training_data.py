#!/usr/bin/env python3
"""Build unified training datasets from ContextWatch JSONL sources.

Outputs:
  - contextwatch/data/training/training_all.jsonl
  - contextwatch/data/training/training_train.jsonl
  - contextwatch/data/training/training_val.jsonl
  - contextwatch/data/training/training_test.jsonl
  - contextwatch/data/training/training_stats.json
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "contextwatch" / "data"
OUT_DIR = DATA_ROOT / "training"

SOURCE_FILES = [
    DATA_ROOT / "golden_dataset" / "golden_normal.jsonl",
    DATA_ROOT / "golden_dataset" / "golden_anomalies.jsonl",
    DATA_ROOT / "synthetic" / "mcp" / "mcp_normal_logs.jsonl",
    DATA_ROOT / "synthetic" / "mcp" / "mcp_anomaly_logs.jsonl",
    DATA_ROOT / "synthetic" / "a2a" / "a2a_normal_logs.jsonl",
    DATA_ROOT / "synthetic" / "a2a" / "a2a_anomaly_logs.jsonl",
    DATA_ROOT / "synthetic" / "mixed" / "mcp_a2a_mixed_logs.jsonl",
]


def infer_label(record: dict[str, Any], source_path: Path) -> int:
    if isinstance(record.get("is_anomaly"), bool):
        return 1 if record["is_anomaly"] else 0
    gt = record.get("ground_truth_label")
    if gt is not None:
        try:
            return 1 if int(gt) == 1 else 0
        except (ValueError, TypeError):
            pass
    if record.get("anomaly_type"):
        return 1
    return 1 if "anomal" in source_path.name.lower() else 0


def normalize_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    label = infer_label(record, source_path)
    out = dict(record)
    out["ground_truth_label"] = label
    out["is_anomaly"] = bool(label)
    out["source_dataset"] = str(source_path.relative_to(DATA_ROOT))
    out["split"] = ""
    return out


def stratified_split(records: list[dict[str, Any]], seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    normal = [r for r in records if int(r.get("ground_truth_label", 0)) == 0]
    anomaly = [r for r in records if int(r.get("ground_truth_label", 0)) == 1]
    rng.shuffle(normal)
    rng.shuffle(anomaly)

    def split_class(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        return train, val, test

    n_train, n_val, n_test = split_class(normal)
    a_train, a_val, a_test = split_class(anomaly)

    train = n_train + a_train
    val = n_val + a_val
    test = n_test + a_test
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"
    return train, val, test


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for src in SOURCE_FILES:
        if not src.exists():
            continue
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                norm = normalize_record(raw, src)
                log_id = str(norm.get("log_id") or "")
                dedupe_key = f"{norm.get('source_dataset')}::{log_id}" if log_id else json.dumps(norm, sort_keys=True)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(norm)
                source_counts[str(src.relative_to(DATA_ROOT))] += 1

    train, val, test = stratified_split(rows)
    all_rows = train + val + test

    out_all = OUT_DIR / "training_all.jsonl"
    out_train = OUT_DIR / "training_train.jsonl"
    out_val = OUT_DIR / "training_val.jsonl"
    out_test = OUT_DIR / "training_test.jsonl"
    out_stats = OUT_DIR / "training_stats.json"

    write_jsonl(out_all, all_rows)
    write_jsonl(out_train, train)
    write_jsonl(out_val, val)
    write_jsonl(out_test, test)

    label_counts = Counter(int(r.get("ground_truth_label", 0)) for r in all_rows)
    stats = {
        "total": len(all_rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "normal": int(label_counts.get(0, 0)),
        "anomaly": int(label_counts.get(1, 0)),
        "source_counts": dict(source_counts),
        "files": {
            "all": str(out_all.relative_to(ROOT)),
            "train": str(out_train.relative_to(ROOT)),
            "val": str(out_val.relative_to(ROOT)),
            "test": str(out_test.relative_to(ROOT)),
        },
    }

    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Built training corpus")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
