"""Production-grade multi-cluster Volume Hypersphere Minimization engine."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from contextwatch.services.detection.calibration import calibrate_threshold
from contextwatch.services.detection.clustering import heuristic_cluster_count, run_kmeans
from contextwatch.services.detection.distance import MetricName, cosine_distance, distance_to_center, euclidean_distance, pairwise_distance
from contextwatch.services.detection.normalization import EPSILON, as_2d_array, normalize

logger = logging.getLogger(__name__)


@dataclass
class ClusterProfile:
    """Per-cluster VHM state."""

    cluster_id: int
    center: list[float]
    radius: float
    volume: float
    size: int
    threshold: float

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return asdict(self)


@dataclass
class VHMScoreResult:
    """Rich anomaly result for a single embedding."""

    is_anomaly: bool
    score: float
    distance: float
    cluster_id: int
    threshold: float
    normalized_distance: float

    def to_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)


@dataclass
class DriftMetrics:
    """Drift statistics for recent embeddings."""

    detected: bool
    mean_shift: float
    std_shift: float
    kl_divergence: float
    recent_count: int
    refit_triggered: bool = False

    def to_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)


class VHMEngine:
    """Multi-cluster VHM with normalization, drift handling, and observability."""

    def __init__(
        self,
        dimensions: int = 64,
        metric: MetricName = "cosine",
        cluster_count: int | None = None,
        radius_quantile: float = 0.95,
        calibration_bins: int = 64,
        buffer_size: int = 10_000,
        refit_interval: int = 500,
        drift_mean_threshold: float = 0.15,
        drift_std_threshold: float = 0.15,
        drift_kl_threshold: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.dimensions = dimensions
        self.metric = metric
        self.cluster_count = cluster_count
        self.radius_quantile = float(np.clip(radius_quantile, 0.5, 0.999))
        self.calibration_bins = max(8, calibration_bins)
        self.buffer_size = max(1, buffer_size)
        self.refit_interval = max(1, refit_interval)
        self.drift_mean_threshold = drift_mean_threshold
        self.drift_std_threshold = drift_std_threshold
        self.drift_kl_threshold = drift_kl_threshold
        self.random_state = random_state

        self.center: list[float] = []
        self.radius: float = 0.0
        self.decision_radius: float = 0.0
        self.volume: float = 0.0
        self.is_fitted: bool = False
        self.decision_scale: float = 1.0
        self.clusters: list[dict[str, float | int | list[float]]] = []

        self._cluster_centers = np.empty((0, dimensions), dtype=np.float32)
        self._cluster_radii = np.empty(0, dtype=np.float32)
        self._cluster_thresholds = np.empty(0, dtype=np.float32)
        self._cluster_sizes = np.empty(0, dtype=np.int32)
        self._recent_buffer: deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self._recent_since_refit = 0
        self._training_score_mean = 0.0
        self._training_score_std = 0.0
        self._training_hist = np.full(20, 1.0 / 20.0, dtype=np.float32)
        self._last_drift = DriftMetrics(False, 0.0, 0.0, 0.0, 0, False)
        self._scored_count = 0
        self._anomaly_count = 0

    @staticmethod
    def _hypersphere_volume(radius: float, dimensions: int) -> float:
        if radius <= 0.0:
            return 0.0
        try:
            # Use log-space to prevent OverflowError in high dimensions
            # Formula: V = (pi^(d/2) / Gamma(d/2 + 1)) * r^d
            # log(V) = (d/2)*log(pi) - log(Gamma(d/2 + 1)) + d*log(r)
            log_v = (dimensions / 2.0) * math.log(math.pi) - math.lgamma(dimensions / 2.0 + 1.0) + dimensions * math.log(radius)
            return math.exp(log_v)
        except (OverflowError, ValueError):
            return float('inf')

    def _geometric_median(self, points: np.ndarray, iterations: int = 25) -> np.ndarray:
        """Approximate a robust cluster center with Weiszfeld's algorithm."""
        if len(points) == 1:
            return points[0]
        current = points.mean(axis=0)
        for _ in range(iterations):
            dists = np.linalg.norm(points - current, axis=1)
            dists = np.clip(dists, EPSILON, None)
            weights = 1.0 / dists
            weight_sum = float(weights.sum())
            if weight_sum <= EPSILON:
                break
            current = (points * weights[:, None]).sum(axis=0) / weight_sum
        return normalize(current) if self.metric == "cosine" else current.astype(np.float32, copy=False)

    def _validate_metric(self) -> None:
        if self.metric not in {"euclidean", "cosine"}:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _prepare_embeddings(self, embeddings: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        pts = as_2d_array(embeddings)
        if self.is_fitted and pts.shape[1] != self.dimensions:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimensions}, got {pts.shape[1]}")
        if self.dimensions and pts.shape[1] != self.dimensions:
            self.dimensions = pts.shape[1]
        elif not self.dimensions:
            self.dimensions = pts.shape[1]
        if np.any(~np.isfinite(pts)):
            raise ValueError("Embeddings contain non-finite values")
        return normalize(pts)

    def _refresh_public_state(self) -> None:
        if self._cluster_centers.size == 0:
            self.center = []
            self.radius = 0.0
            self.decision_radius = 0.0
            self.volume = 0.0
            self.clusters = []
            return

        weights = self._cluster_sizes.astype(np.float32)
        total = float(weights.sum()) if weights.size else 1.0
        aggregate_center = (self._cluster_centers * weights[:, None]).sum(axis=0) / max(total, EPSILON)
        aggregate_center = normalize(aggregate_center) if self.metric == "cosine" else aggregate_center
        self.center = aggregate_center.astype(np.float32, copy=False).tolist()
        self.radius = float(np.average(self._cluster_radii, weights=weights))
        self.decision_radius = float(np.average(self._cluster_thresholds, weights=weights))
        self.volume = float(sum(self._hypersphere_volume(float(radius), self.dimensions) for radius in self._cluster_radii))
        self.clusters = [
            ClusterProfile(
                cluster_id=int(cluster_id),
                center=self._cluster_centers[cluster_id].tolist(),
                radius=float(self._cluster_radii[cluster_id]),
                volume=self._hypersphere_volume(float(self._cluster_radii[cluster_id]), self.dimensions),
                size=int(self._cluster_sizes[cluster_id]),
                threshold=float(self._cluster_thresholds[cluster_id]),
            ).to_dict()
            for cluster_id in range(len(self._cluster_centers))
        ]

    def _score_matrix(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted or self._cluster_centers.size == 0:
            zeros = np.zeros(len(embeddings), dtype=np.float32)
            negative_ids = np.full(len(embeddings), -1, dtype=np.int32)
            return negative_ids, zeros, zeros, zeros

        distances = pairwise_distance(embeddings, self._cluster_centers, self.metric)
        cluster_ids = distances.argmin(axis=1).astype(np.int32)
        min_distances = distances[np.arange(len(embeddings)), cluster_ids]
        cluster_thresholds = self._cluster_thresholds[cluster_ids]
        cluster_radii = np.clip(self._cluster_radii[cluster_ids], EPSILON, None)
        normalized_distance = min_distances / cluster_radii
        return cluster_ids, min_distances.astype(np.float32), cluster_thresholds.astype(np.float32), normalized_distance.astype(np.float32)

    def _update_training_distribution(self, embeddings: np.ndarray) -> None:
        _, _, _, normalized = self._score_matrix(embeddings)
        self._training_score_mean = float(np.mean(normalized)) if normalized.size else 0.0
        self._training_score_std = float(np.std(normalized)) if normalized.size else 0.0
        hist, _ = np.histogram(normalized, bins=20, range=(0.0, max(2.0, float(np.max(normalized, initial=1.0)))), density=True)
        hist = hist.astype(np.float32) + EPSILON
        self._training_hist = hist / hist.sum()

    def _compute_drift(self) -> DriftMetrics:
        if not self.is_fitted or len(self._recent_buffer) < max(10, len(self._cluster_centers)):
            return DriftMetrics(False, 0.0, 0.0, 0.0, len(self._recent_buffer), False)

        recent = np.asarray(self._recent_buffer, dtype=np.float32)
        _, _, _, normalized = self._score_matrix(recent)
        recent_mean = float(np.mean(normalized)) if normalized.size else 0.0
        recent_std = float(np.std(normalized)) if normalized.size else 0.0
        hist, _ = np.histogram(normalized, bins=len(self._training_hist), range=(0.0, max(2.0, float(np.max(normalized, initial=1.0)))), density=True)
        hist = hist.astype(np.float32) + EPSILON
        hist = hist / hist.sum()
        kl_divergence = float(np.sum(hist * np.log(hist / self._training_hist)))
        mean_shift = abs(recent_mean - self._training_score_mean)
        std_shift = abs(recent_std - self._training_score_std)
        detected = (
            mean_shift > self.drift_mean_threshold
            or std_shift > self.drift_std_threshold
            or kl_divergence > self.drift_kl_threshold
        )
        return DriftMetrics(detected, mean_shift, std_shift, kl_divergence, len(self._recent_buffer), False)

    def fit(self, embeddings: list[list[float]] | np.ndarray) -> "VHMEngine":
        """Fit clustered hyperspheres on normalized embeddings."""
        self._validate_metric()
        points = self._prepare_embeddings(embeddings)
        cluster_count = heuristic_cluster_count(len(points), self.cluster_count)
        labels, _ = run_kmeans(points, cluster_count, metric=self.metric, random_state=self.random_state)

        centers: list[np.ndarray] = []
        radii: list[float] = []
        thresholds: list[float] = []
        sizes: list[int] = []

        for cluster_id in range(cluster_count):
            cluster_points = points[labels == cluster_id]
            if len(cluster_points) == 0:
                continue
            center = self._geometric_median(cluster_points)
            distances = distance_to_center(cluster_points, center, self.metric)
            radius = float(np.quantile(distances, self.radius_quantile)) + 1e-6
            threshold = radius * self.decision_scale
            centers.append(center.astype(np.float32, copy=False))
            radii.append(radius)
            thresholds.append(threshold)
            sizes.append(int(len(cluster_points)))
            logger.info(
                "VHM cluster fitted: cluster_id=%s size=%s radius=%.6f threshold=%.6f metric=%s",
                cluster_id,
                len(cluster_points),
                radius,
                threshold,
                self.metric,
            )

        if not centers:
            raise ValueError("Unable to fit VHM clusters from embeddings")

        self._cluster_centers = np.asarray(centers, dtype=np.float32)
        self._cluster_radii = np.asarray(radii, dtype=np.float32)
        self._cluster_thresholds = np.asarray(thresholds, dtype=np.float32)
        self._cluster_sizes = np.asarray(sizes, dtype=np.int32)
        self.is_fitted = True
        self._refresh_public_state()
        self._update_training_distribution(points)
        self._recent_buffer.clear()
        for point in points[-self.buffer_size:]:
            self._recent_buffer.append(point)
        self._recent_since_refit = 0
        self._last_drift = DriftMetrics(False, 0.0, 0.0, 0.0, len(self._recent_buffer), False)
        return self

    def calibrate(self, normal_embeddings: list[list[float]], anomaly_embeddings: list[list[float]]) -> dict[str, float | int | list[dict[str, float]]]:
        """Calibrate a global normalized threshold using sampled candidates."""
        # Fix: Handle numpy arrays properly - convert to lists if needed
        if isinstance(normal_embeddings, np.ndarray):
            normal_embeddings = normal_embeddings.tolist()
        if isinstance(anomaly_embeddings, np.ndarray):
            anomaly_embeddings = anomaly_embeddings.tolist()
            
        if not normal_embeddings or len(normal_embeddings) == 0:
            raise ValueError("Normal embeddings are required for calibration")

        if not anomaly_embeddings or len(anomaly_embeddings) == 0:
            raise ValueError("Anomaly embeddings are required for calibration")

        if not self.is_fitted:
            self.fit(normal_embeddings)

        normal_points = self._prepare_embeddings(normal_embeddings)
        _, _, _, normal_scores = self._score_matrix(normal_points)
        if not anomaly_embeddings:
            self.decision_scale = 1.0
            self._cluster_thresholds = self._cluster_radii.copy()
            self._refresh_public_state()
            return {
                "decision_radius": float(self.decision_radius),
                "decision_scale": float(self.decision_scale),
                "best_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "normal_samples": len(normal_embeddings),
                "anomaly_samples": 0,
                "precision_recall_curve": [],
            }

        anomaly_points = self._prepare_embeddings(anomaly_embeddings)
        _, _, _, anomaly_scores = self._score_matrix(anomaly_points)
        calibration = calibrate_threshold(
            normal_scores=normal_scores.flatten(),
            anomaly_scores=anomaly_scores.flatten(),
            bins=self.calibration_bins,
            fallback=self.decision_scale,
        )
        self.decision_scale = max(float(calibration["threshold"]), 1e-6)
        self._cluster_thresholds = np.clip(self._cluster_radii * self.decision_scale, 1e-6, None)
        self._refresh_public_state()
        logger.info(
            "VHM calibrated: decision_scale=%.6f decision_radius=%.6f f1=%.4f precision=%.4f recall=%.4f",
            self.decision_scale,
            self.decision_radius,
            calibration["best_f1"],
            calibration["precision"],
            calibration["recall"],
        )
        return {
            "decision_radius": float(self.decision_radius),
            "decision_scale": float(self.decision_scale),
            "best_f1": float(calibration["best_f1"]),
            "precision": float(calibration["precision"]),
            "recall": float(calibration["recall"]),
            "normal_samples": len(normal_embeddings),
            "anomaly_samples": len(anomaly_embeddings),
            "precision_recall_curve": calibration["precision_recall_curve"],
        }

    def score_details(self, embedding: Sequence[float]) -> dict[str, float | bool | int]:
        """Return rich anomaly details for one embedding."""
        if not self.is_fitted:
            return VHMScoreResult(False, 0.0, 0.0, -1, 0.0, 0.0).to_dict()

        point = normalize(np.asarray(embedding, dtype=np.float32))
        if point.ndim != 1 or point.size == 0:
            return VHMScoreResult(False, 0.0, 0.0, -1, 0.0, 0.0).to_dict()
        if point.shape[0] != self.dimensions:
            raise ValueError("score_details expects a single embedding vector")

        cluster_ids, distances, thresholds, normalized = self._score_matrix(point.reshape(1, -1))
        cluster_id = int(cluster_ids[0])
        distance = float(distances[0])
        threshold = float(thresholds[0])
        normalized_distance = float(normalized[0])
        score = max(0.0, distance / max(threshold, EPSILON) - 1.0)
        is_anomaly = distance > threshold

        self._scored_count += 1
        if is_anomaly:
            self._anomaly_count += 1
        if is_anomaly or self._scored_count % 100 == 0:
            logger.info(
                "VHM scoring metrics: scored=%s anomalies=%s anomaly_rate=%.4f cluster_id=%s distance=%.6f threshold=%.6f",
                self._scored_count,
                self._anomaly_count,
                self._anomaly_count / max(self._scored_count, 1),
                cluster_id,
                distance,
                threshold,
            )

        return VHMScoreResult(
            is_anomaly=is_anomaly,
            score=float(score),
            distance=distance,
            cluster_id=cluster_id,
            threshold=threshold,
            normalized_distance=normalized_distance,
        ).to_dict()

    def score(self, embedding: Sequence[float]) -> tuple[bool, float]:
        """Backward-compatible scoring interface."""
        result = self.score_details(embedding)
        return bool(result["is_anomaly"]), float(result["score"])

    def score_batch(self, embeddings: Iterable[Sequence[float]]) -> list[dict[str, float | bool | int]]:
        """Vectorized batch scoring."""
        items = list(embeddings)
        if not items:
            return []
        if not self.is_fitted:
            return [VHMScoreResult(False, 0.0, 0.0, -1, 0.0, 0.0).to_dict() for _ in items]

        points = self._prepare_embeddings(items)
        cluster_ids, distances, thresholds, normalized = self._score_matrix(points)
        is_anomaly = distances > thresholds
        scores = np.maximum(0.0, distances / np.clip(thresholds, EPSILON, None) - 1.0)

        batch_results: list[dict[str, float | bool | int]] = []
        for idx in range(len(points)):
            batch_results.append(
                VHMScoreResult(
                    is_anomaly=bool(is_anomaly[idx]),
                    score=float(scores[idx]),
                    distance=float(distances[idx]),
                    cluster_id=int(cluster_ids[idx]),
                    threshold=float(thresholds[idx]),
                    normalized_distance=float(normalized[idx]),
                ).to_dict()
            )

        self._scored_count += len(batch_results)
        self._anomaly_count += int(np.sum(is_anomaly))
        logger.info(
            "VHM batch scoring metrics: batch=%s anomalies=%s anomaly_rate=%.4f",
            len(batch_results),
            int(np.sum(is_anomaly)),
            self._anomaly_count / max(self._scored_count, 1),
        )
        return batch_results

    def contains(self, point: Sequence[float], margin: float = 1.0) -> tuple[bool, float]:
        """Check whether a point lies inside its assigned cluster sphere."""
        result = self.score_details(point)
        threshold = float(result["threshold"]) * max(margin, 0.0)
        return float(result["distance"]) <= threshold, float(result["distance"])

    def update_buffer(self, embeddings: Iterable[Sequence[float]], auto_refit: bool = True) -> dict[str, float | bool | int]:
        """Store recent normal embeddings and optionally refit on drift."""
        items = list(embeddings)
        if not items:
            return self._last_drift.to_dict()

        points = self._prepare_embeddings(items)
        for point in points:
            self._recent_buffer.append(point)
        self._recent_since_refit += len(points)

        drift = self._compute_drift()
        should_refit = auto_refit and len(self._recent_buffer) >= max(10, len(self._cluster_centers)) and (
            drift.detected or self._recent_since_refit >= self.refit_interval
        )
        if should_refit:
            logger.warning(
                "VHM refit triggered: recent_count=%s mean_shift=%.4f std_shift=%.4f kl=%.4f",
                drift.recent_count,
                drift.mean_shift,
                drift.std_shift,
                drift.kl_divergence,
            )
            self.fit(np.asarray(self._recent_buffer, dtype=np.float32))
            drift.refit_triggered = True

        self._last_drift = drift
        logger.info(
            "VHM drift metrics: detected=%s mean_shift=%.4f std_shift=%.4f kl=%.4f",
            drift.detected,
            drift.mean_shift,
            drift.std_shift,
            drift.kl_divergence,
        )
        return drift.to_dict()

    def get_metrics(self) -> dict[str, object]:
        """Expose runtime metrics for observability."""
        anomaly_rate = float(self._anomaly_count / self._scored_count) if self._scored_count else 0.0
        return {
            "metric": self.metric,
            "cluster_count": len(self._cluster_centers),
            "radius": float(self.radius),
            "decision_radius": float(self.decision_radius),
            "decision_scale": float(self.decision_scale),
            "cluster_radii": [float(radius) for radius in self._cluster_radii.tolist()],
            "cluster_thresholds": [float(threshold) for threshold in self._cluster_thresholds.tolist()],
            "anomaly_rate": anomaly_rate,
            "drift": self._last_drift.to_dict(),
            "buffer_size": len(self._recent_buffer),
        }


__all__ = [
    "ClusterProfile",
    "DriftMetrics",
    "VHMEngine",
    "VHMScoreResult",
    "cosine_distance",
    "euclidean_distance",
]
