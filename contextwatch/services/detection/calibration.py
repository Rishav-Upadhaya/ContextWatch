"""Threshold calibration helpers for the VHM engine."""

from __future__ import annotations

import math

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def sampled_thresholds(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    bins: int = 64,
    fallback: float = 1.0,
) -> np.ndarray:
    """Generate a bounded set of candidate thresholds for calibration."""
    combined = np.concatenate([normal_scores, anomaly_scores]).astype(np.float32, copy=False)
    if combined.size == 0:
        return np.asarray([fallback], dtype=np.float32)

    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if math.isclose(lower, upper):
        return np.asarray([lower, fallback], dtype=np.float32)

    quantiles = np.linspace(0.0, 1.0, max(3, bins), dtype=np.float32)
    candidates = np.quantile(combined, quantiles)
    candidates = np.unique(np.concatenate([candidates, np.asarray([fallback], dtype=np.float32)]))
    return candidates.astype(np.float32, copy=False)


def calibrate_threshold(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    bins: int = 64,
    fallback: float = 1.0,
) -> dict[str, float | list[dict[str, float]]]:
    """Optimize a normalized anomaly threshold with sampled candidates."""
    candidates = sampled_thresholds(normal_scores, anomaly_scores, bins=bins, fallback=fallback)
    best_threshold = float(fallback)
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    curve: list[dict[str, float]] = []

    for threshold in candidates:
        tp = float(np.sum(anomaly_scores > threshold))
        fp = float(np.sum(normal_scores > threshold))
        fn = float(np.sum(anomaly_scores <= threshold))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        curve.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        if f1 > best_f1 or (math.isclose(f1, best_f1) and float(threshold) < best_threshold):
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_precision = float(precision)
            best_recall = float(recall)

    return {
        "threshold": best_threshold,
        "best_f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "precision_recall_curve": curve,
    }
