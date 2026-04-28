"""Normalization helpers for embedding-based anomaly detection."""

from __future__ import annotations

from typing import Iterable

import numpy as np


EPSILON = 1e-12


def normalize(x: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """L2-normalize a vector or matrix safely.

    Zero vectors are preserved as zeros to avoid propagating NaNs.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm <= eps:
            return np.zeros_like(arr)
        return arr / norm

    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms <= eps, 1.0, norms)
    normalized = arr / safe_norms
    normalized[norms[:, 0] <= eps] = 0.0
    return normalized.astype(np.float32, copy=False)


def as_2d_array(embeddings: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert embeddings into a non-empty 2D float32 matrix."""
    arr = np.asarray(list(embeddings), dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Embeddings are empty")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError("Embeddings must have at least one dimension")
    return arr
