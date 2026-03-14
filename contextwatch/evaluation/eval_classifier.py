from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from core.classifier import AnomalyClassifier
from core.schema import AnomalyResult
from core.normalizer import LogNormalizer


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_classifier(
    data_dir: Path = Path("data/golden_dataset"),
    max_rows: int | None = None,
    min_accuracy: float = 0.75,
) -> dict:
    rows = load(data_dir / "golden_anomalies.jsonl")
    if max_rows is not None:
        rows = rows[:max_rows]
    normalizer = LogNormalizer()
    classifier = AnomalyClassifier(get_settings())

    y_true = [x["anomaly_type"] for x in rows]
    y_pred = []
    for raw in rows:
        normalized = normalizer.normalize(raw)
        anomaly = AnomalyResult(
            log_id=normalized.log_id,
            anomaly_score=0.9,
            is_anomaly=True,
            anomaly_type=None,
            confidence=0.9,
        )
        pred = classifier.classify(normalized, anomaly, [])
        y_pred.append(pred.anomaly_type)

    acc = accuracy_score(y_true, y_pred)
    print(f"accuracy={acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    assert acc >= min_accuracy, f"Classifier accuracy below target: {acc:.4f}"
    return {"accuracy": acc}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate anomaly classifier")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for anomaly rows")
    parser.add_argument("--min-accuracy", type=float, default=0.75, help="Minimum acceptable accuracy")
    args = parser.parse_args()
    evaluate_classifier(max_rows=args.max_rows, min_accuracy=args.min_accuracy)
