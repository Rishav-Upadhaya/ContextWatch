"""Lightweight clustering utilities for multi-cluster VHM."""

from __future__ import annotations

import math

import numpy as np

from contextwatch.services.detection.distance import MetricName, pairwise_distance
from contextwatch.services.detection.normalization import normalize


def heuristic_cluster_count(sample_count: int, requested_k: int | None = None) -> int:
    """Choose a pragmatic cluster count for the current sample size."""
    if sample_count <= 0:
        return 1
    if requested_k is not None:
        return max(1, min(int(requested_k), sample_count))
    if sample_count < 8:
        return 1
    return max(1, min(sample_count, int(math.sqrt(sample_count / 2.0))))


def _init_centers(points: np.ndarray, k: int, metric: MetricName, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point initialization to reduce poor local minima."""
    first_idx = int(rng.integers(0, len(points)))
    centers = [points[first_idx]]
    while len(centers) < k:
        dist_matrix = pairwise_distance(points, np.asarray(centers, dtype=np.float32), metric)
        min_dists = dist_matrix.min(axis=1)
        next_idx = int(np.argmax(min_dists))
        centers.append(points[next_idx])
    centers_arr = np.asarray(centers, dtype=np.float32)
    if metric == "cosine":
        return normalize(centers_arr)
    return centers_arr


def run_kmeans(
    points: np.ndarray,
    k: int,
    metric: MetricName = "cosine",
    max_iter: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster points with a NumPy-only k-means implementation."""
    if len(points) == 0:
        raise ValueError("Cannot cluster an empty point set")

    k = max(1, min(int(k), len(points)))
    if k == 1:
        center = points.mean(axis=0, keepdims=True)
        if metric == "cosine":
            center = normalize(center)
        return np.zeros(len(points), dtype=np.int32), center.astype(np.float32, copy=False)

    rng = np.random.default_rng(random_state)
    centers = _init_centers(points, k, metric, rng)
    labels = np.zeros(len(points), dtype=np.int32)

    for _ in range(max_iter):
        dist_matrix = pairwise_distance(points, centers, metric)
        new_labels = dist_matrix.argmin(axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        updated_centers = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if not np.any(mask):
                farthest_idx = int(np.argmax(dist_matrix.min(axis=1)))
                updated_center = points[farthest_idx]
            else:
                updated_center = points[mask].mean(axis=0)
            updated_centers.append(updated_center)

        centers = np.asarray(updated_centers, dtype=np.float32)
        if metric == "cosine":
            centers = normalize(centers)

    return labels, centers.astype(np.float32, copy=False)
