"""Distance utilities used by the VHM engine."""

from __future__ import annotations

from typing import Literal

import numpy as np

MetricName = Literal["euclidean", "cosine"]
EPSILON = 1e-12


def cosine_distance(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute cosine distance safely."""
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= EPSILON or right_norm <= EPSILON:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    similarity = float(np.clip(similarity, -1.0, 1.0))
    return 1.0 - similarity


def euclidean_distance(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Compute Euclidean distance."""
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(left - right))


def pairwise_distance(points: np.ndarray, centers: np.ndarray, metric: MetricName) -> np.ndarray:
    """Return the [n_points, n_centers] distance matrix."""
    if points.ndim != 2 or centers.ndim != 2:
        raise ValueError("pairwise_distance expects 2D arrays")
    if points.shape[1] != centers.shape[1]:
        raise ValueError("Points and centers must have the same dimensionality")

    if metric == "euclidean":
        deltas = points[:, None, :] - centers[None, :, :]
        return np.linalg.norm(deltas, axis=2)

    if metric == "cosine":
        similarities = np.clip(points @ centers.T, -1.0, 1.0)
        return 1.0 - similarities

    raise ValueError(f"Unsupported metric: {metric}")


def distance_to_center(points: np.ndarray, center: np.ndarray, metric: MetricName) -> np.ndarray:
    """Return point-to-center distances for a batch."""
    return pairwise_distance(points, center.reshape(1, -1), metric)[:, 0]
