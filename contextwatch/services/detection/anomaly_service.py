"""Anomaly detection service — application layer.

This service orchestrates the domain-level anomaly detection pipeline:
normalisation → embedding → anomaly scoring → threshold check → result.
All heavy algorithms live in ``domain/``; this file merely wires them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from contextwatch.services.detection.anomaly_detector import AnomalyDetector
from contextwatch.services.ai.classifier import KNearestNeighbors
from contextwatch.services.graph.knowledge_graph import KnowledgeGraph
from contextwatch.utils.vector_math import cosine_distance


@dataclass(frozen=True)
class DetectionResult:
    log_id: str
    is_anomaly: bool
    anomaly_score: float
    threshold_used: float
    anomaly_type: Optional[str] = None
    confidence: float = 0.0
    explanation: Optional[str] = None


class AnomalyService:
    """Application service for anomaly detection."""

    def __init__(
        self,
        baseline_embeddings: List[List[float]],
        threshold: float = 0.20,
        k_neighbours: int = 5,
    ):
        self._baseline = list(baseline_embeddings)  # shallow copy
        self._threshold = threshold
        self._detector = AnomalyDetector(window_size=100, threshold=3.0)
        self._knn = KNearestNeighbors(k=k_neighbours, weighted=True)
        self._trained = False
        self._graph = KnowledgeGraph(directed=True)
        self._recent_scores: List[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_classifier(
        self,
        features: List[List[float]],
        labels: List[str],
    ) -> None:
        """Retrain the k-NN classifier with labeled data."""
        if len(features) != len(labels):
            raise ValueError("Feature / label count mismatch")
        self._knn.fit(features, labels)
        self._trained = True

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------
    def detect(self, log_id: str, embedding: List[float],
               metadata: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """Run the full detection pipeline for a single log."""

        # 1. Nearest-neighbour distance to baseline
        min_distance = float("inf")
        for baseline_vec in self._baseline:
            d = cosine_distance(embedding, baseline_vec)
            if d < min_distance:
                min_distance = d

        is_anomaly = min_distance > self._threshold
        confidence = max(0.0, min(1.0, 1.0 - min_distance))

        # 2. Classification (only if anomaly detected)
        anomaly_type = None
        if is_anomaly and self._trained:
            predictions = self._knn.predict([embedding])
            anomaly_type = predictions[0] if predictions else "UNKNOWN"

        # 3. Update sliding window for adaptive detection
        self._detector.add_score(min_distance)
        self._recent_scores.append(min_distance)

        # 4. Update knowledge graph
        self._upsert_event(log_id, embedding, is_anomaly, anomaly_type, metadata)

        return DetectionResult(
            log_id=log_id,
            is_anomaly=is_anomaly,
            anomaly_score=min_distance,
            threshold_used=self._threshold,
            anomaly_type=anomaly_type,
            confidence=confidence,
            explanation=self._build_explanation(is_anomaly, anomaly_type, min_distance),
        )

    # ------------------------------------------------------------------
    # Graph helpers (infrastructure boundary)
    # ------------------------------------------------------------------
    def _upsert_event(
        self,
        log_id: str,
        embedding: List[float],
        is_anomaly: bool,
        anomaly_type: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        node = self._graph.add_node(log_id, "Event", {"embedding_dim": len(embedding)})
        if is_anomaly:
            anomaly_node = self._graph.add_node(f"anomaly_{log_id}", "Anomaly", {
                "log_id": log_id,
                "anomaly_type": anomaly_type,
            })
            self._graph.add_edge(log_id, f"anomaly_{log_id}", "HAS_ANOMALY")

    @staticmethod
    def _build_explanation(is_anomaly: bool, anomaly_type: Optional[str],
                          distance: float) -> Optional[str]:
        if not is_anomaly:
            return None
        parts = [f"Anomaly detected (distance={distance:.3f})."]
        if anomaly_type:
            parts.append(f"Classified as: {anomaly_type}.")
        return " ".join(parts)
