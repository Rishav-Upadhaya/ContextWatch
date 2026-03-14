from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from core.detector import AnomalyDetector
from core.embedder import LogEmbedder
from core.normalizer import LogNormalizer


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_best_threshold(y_true: list[int], y_score: list[float], start: float = 0.20, end: float = 0.60, step: float = 0.01) -> tuple[float, float]:
    best_tau = start
    best_f1 = -1.0
    for tau in np.arange(start, end + 1e-9, step):
        y_pred = [1 if s > tau else 0 for s in y_score]
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(round(tau, 2))
    return best_tau, best_f1


def evaluate_detector(data_dir: Path = Path("data/golden_dataset"), max_normal: int | None = None, max_anomaly: int | None = None) -> dict:
    settings = get_settings()
    original_collection = settings.CHROMA_COLLECTION_NORMAL
    temp_collection = f"{original_collection}_eval_{uuid4().hex[:8]}"
    settings.CHROMA_COLLECTION_NORMAL = temp_collection
    normalizer = LogNormalizer()
    embedder = LogEmbedder(settings)
    detector = AnomalyDetector(embedder, settings)

    normal = load(data_dir / "golden_normal.jsonl")
    anomalies = load(data_dir / "golden_anomalies.jsonl")
    if max_normal is not None:
        normal = normal[:max_normal]
    if max_anomaly is not None:
        anomalies = anomalies[:max_anomaly]

    try:
        normal_norm = [normalizer.normalize(x) for x in normal]
        embedder.embed_batch(normal_norm)

        rows = normal + anomalies
        y_true = [0] * len(normal) + [1] * len(anomalies)
        normalized_rows = [normalizer.normalize(x) for x in rows]
        y_score = [detector.detect(x).anomaly_score for x in normalized_rows]

        best_tau, _ = find_best_threshold(y_true, y_score, start=0.20, end=0.60, step=0.01)

        y_pred = [1 if s > best_tau else 0 for s in y_score]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_true, y_pred)

        protocol_scores: dict[str, dict[str, float]] = {}
        for protocol in ("MCP", "A2A"):
            indices = [i for i, n in enumerate(normalized_rows) if n.protocol == protocol]
            if not indices:
                continue
            proto_true = [y_true[i] for i in indices]
            proto_score = [y_score[i] for i in indices]
            proto_tau, proto_f1 = find_best_threshold(proto_true, proto_score, start=0.20, end=0.60, step=0.01)
            protocol_scores[protocol] = {"threshold": proto_tau, "f1": float(proto_f1)}

        print(f"best_tau={best_tau} f1={f1:.4f} precision={precision:.4f} recall={recall:.4f} auc={roc_auc:.4f}")
        for protocol, stats in protocol_scores.items():
            print(f"best_tau_{protocol.lower()}={stats['threshold']} f1_{protocol.lower()}={stats['f1']:.4f}")
        print("confusion_matrix=", cm.tolist())
        if max_normal is None and max_anomaly is None:
            assert f1 >= 0.80, f"F1 below target: {f1:.4f}"
        return {
            "threshold": best_tau,
            "protocol_thresholds": protocol_scores,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": roc_auc,
        }
    finally:
        try:
            embedder.client.delete_collection(temp_collection)
        except Exception:
            pass
        settings.CHROMA_COLLECTION_NORMAL = original_collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate detector and calibrate thresholds")
    parser.add_argument("--max-normal", type=int, default=None, help="Optional cap for normal rows")
    parser.add_argument("--max-anomaly", type=int, default=None, help="Optional cap for anomaly rows")
    args = parser.parse_args()
    evaluate_detector(max_normal=args.max_normal, max_anomaly=args.max_anomaly)
