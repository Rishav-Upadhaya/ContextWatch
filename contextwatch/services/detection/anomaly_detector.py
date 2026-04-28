"""Anomaly detection implementation from first principles.

This module implements anomaly detection algorithms without external
dependencies, using only Python's math module and core language features.

Supports multiple detection methods:
1. Statistical Z-score (simple baseline)
2. LogBERT+VHM (advanced semantic anomaly detection)
3. Ensemble voting
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np


class DetectionMethod(Enum):
    """Available anomaly detection methods."""
    ZSCORE = "zscore"  # Simple statistical z-score
    LOGBERT_VHM = "logbert_vhm"  # Semantic + hypersphere
    ENSEMBLE = "ensemble"  # Weighted combination


@dataclass
class AnomalyScore:
    """Complete anomaly scoring result."""
    is_anomalous: bool
    score: float  # 0-1, higher = more anomalous
    confidence: float  # 0-1, model confidence
    method: str  # Which detector flagged it
    details: Dict = None  # Detector-specific details


class AnomalyDetector:
    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 3.0,
        method: DetectionMethod = DetectionMethod.ZSCORE,
        logbert_inference: Optional[object] = None,
    ):
        """Initialize detector with multiple method support.

        Parameters:
        -----------
        window_size: int
            Number of recent points to consider for statistics
        threshold: float
            Z-score threshold for anomaly detection (3.0 = 3 standard deviations)
        method: DetectionMethod
            Which detection method to use (ZSCORE, LOGBERT_VHM, or ENSEMBLE)
        logbert_inference: LogBERTInference
            Optional LogBERT inference engine for semantic detection
        """
        self.window_size = window_size
        self.threshold = threshold
        self.method = method
        self.logbert_inference = logbert_inference

        # Statistical tracking
        self.recent_scores: List[float] = []
        self.recent_embeddings: List[np.ndarray] = []

    def add_score(self, score: float) -> None:
        """Add a new anomaly score to the detection window.

        Parameters:
        -----------
        score: float
            Anomaly score to analyze
        """
        self.recent_scores.append(score)
        # Maintain fixed window size
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add embedding for LogBERT-based detection."""
        self.recent_embeddings.append(embedding)
        if len(self.recent_embeddings) > self.window_size:
            self.recent_embeddings.pop(0)

    def _detect_zscore(self) -> AnomalyScore:
        """Z-score based detection."""
        if len(self.recent_scores) < 3:
            return AnomalyScore(
                is_anomalous=False,
                score=0.0,
                confidence=0.0,
                method="zscore",
                details={"reason": "insufficient_data"},
            )

        # Calculate mean and standard deviation
        mean = sum(self.recent_scores) / len(self.recent_scores)
        variance = sum((x - mean) ** 2 for x in self.recent_scores) / len(self.recent_scores)
        std_dev = sqrt(variance) if variance > 0 else 0.0

        # Calculate z-score for latest score
        latest_score = self.recent_scores[-1]
        latest_z = (latest_score - mean) / std_dev if std_dev > 0 else 0.0

        is_anomalous = abs(latest_z) > self.threshold
        # Normalize z-score to 0-1 confidence
        confidence = min(1.0, abs(latest_z) / self.threshold) if self.threshold > 0 else 0.0

        return AnomalyScore(
            is_anomalous=is_anomalous,
            score=latest_score,
            confidence=confidence,
            method="zscore",
            details={
                "z_score": float(latest_z),
                "mean": float(mean),
                "std_dev": float(std_dev),
                "threshold": self.threshold,
            },
        )

    def _detect_logbert_vhm(self) -> AnomalyScore:
        """LogBERT+VHM semantic anomaly detection."""
        if self.logbert_inference is None:
            return AnomalyScore(
                is_anomalous=False,
                score=0.0,
                confidence=0.0,
                method="logbert_vhm",
                details={"reason": "logbert_not_initialized"},
            )

        if len(self.recent_embeddings) == 0:
            return AnomalyScore(
                is_anomalous=False,
                score=0.0,
                confidence=0.0,
                method="logbert_vhm",
                details={"reason": "no_embeddings"},
            )

        # Use latest embedding
        latest_embedding = self.recent_embeddings[-1]

        try:
            anomaly_score, confidence = self.logbert_inference.compute_anomaly_score(
                latest_embedding
            )
            # Use 0.5 as threshold for VHM (distance-based scoring)
            is_anomalous = anomaly_score > 0.5
            return AnomalyScore(
                is_anomalous=is_anomalous,
                score=anomaly_score,
                confidence=confidence,
                method="logbert_vhm",
                details={
                    "vhm_distance": anomaly_score,
                    "logbert_confidence": confidence,
                },
            )
        except Exception as e:
            return AnomalyScore(
                is_anomalous=False,
                score=0.0,
                confidence=0.0,
                method="logbert_vhm",
                details={"error": str(e)},
            )

    def _detect_ensemble(self) -> AnomalyScore:
        """Ensemble voting across multiple detectors."""
        results = []

        # Z-score detection
        zscore_result = self._detect_zscore()
        results.append((zscore_result, 0.5))  # 50% weight

        # LogBERT detection
        if self.logbert_inference is not None:
            logbert_result = self._detect_logbert_vhm()
            results.append((logbert_result, 0.5))  # 50% weight

        # Weighted voting
        total_weight = 0.0
        weighted_score = 0.0
        vote_anomalous = 0
        total_votes = 0

        for result, weight in results:
            total_weight += weight
            weighted_score += result.score * weight
            if result.is_anomalous:
                vote_anomalous += 1
            total_votes += 1

        # Normalize
        if total_weight > 0:
            avg_score = weighted_score / total_weight
        else:
            avg_score = 0.0

        # Consensus: anomalous if >50% of detectors agree
        is_anomalous = vote_anomalous >= (total_votes / 2.0)
        confidence = float(vote_anomalous) / float(total_votes)  if total_votes > 0 else 0.0

        return AnomalyScore(
            is_anomalous=is_anomalous,
            score=avg_score,
            confidence=confidence,
            method="ensemble",
            details={
                "votes_anomalous": vote_anomalous,
                "total_votes": total_votes,
                "weighted_score": avg_score,
            },
        )

    def detect(self) -> AnomalyScore:
        """Detect anomalies using configured method.

        Returns:
        --------
        AnomalyScore: Complete result with score, confidence, method
        """
        if self.method == DetectionMethod.ZSCORE:
            return self._detect_zscore()
        elif self.method == DetectionMethod.LOGBERT_VHM:
            return self._detect_logbert_vhm()
        elif self.method == DetectionMethod.ENSEMBLE:
            return self._detect_ensemble()
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    def get_statistics(self) -> Dict:
        """Return detector statistics for monitoring."""
        return {
            "method": self.method.value,
            "window_size": self.window_size,
            "threshold": self.threshold,
            "num_scores": len(self.recent_scores),
            "num_embeddings": len(self.recent_embeddings),
            "logbert_initialized": self.logbert_inference is not None,
            "recent_scores_mean": (
                sum(self.recent_scores) / len(self.recent_scores)
                if self.recent_scores
                else None
            ),
        }

