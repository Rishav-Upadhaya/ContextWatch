from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from core.classifier import AnomalyClassifier
from core.detector import AnomalyDetector
from core.embedder import LogEmbedder
from core.normalizer import LogNormalizer


def load_rows(path: Path, limit: int = 1000) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def benchmark_latency() -> dict:
    settings = get_settings()
    original_collection = settings.CHROMA_COLLECTION_NORMAL
    temp_collection = f"{original_collection}_latency_{uuid4().hex[:8]}"
    settings.CHROMA_COLLECTION_NORMAL = temp_collection
    normalizer = LogNormalizer()
    embedder = LogEmbedder(settings)
    detector = AnomalyDetector(embedder, settings)
    classifier = AnomalyClassifier(settings)

    try:
        normal_rows = load_rows(Path("data/golden_dataset/golden_normal.jsonl"), limit=1000)
        baseline = [normalizer.normalize(x) for x in normal_rows[:200]]
        embedder.embed_batch(baseline)

        latencies = []
        for row in normal_rows:
            start = time.perf_counter()
            normalized = normalizer.normalize(row)
            anomaly = detector.detect(normalized)
            _ = classifier.classify(normalized, anomaly, [])
            latencies.append((time.perf_counter() - start) * 1000)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        print(f"p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms")
        assert p95 < 500, f"p95 latency violation: {p95:.2f}"
        return {"p50": p50, "p95": p95, "p99": p99}
    finally:
        try:
            embedder.client.delete_collection(temp_collection)
        except Exception:
            pass
        settings.CHROMA_COLLECTION_NORMAL = original_collection


if __name__ == "__main__":
    benchmark_latency()
