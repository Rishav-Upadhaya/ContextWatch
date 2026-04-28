"""Backward-compatible entrypoint for the production VHM engine."""

from __future__ import annotations

from contextwatch.services.detection.distance import cosine_distance, euclidean_distance
from contextwatch.services.detection.normalization import normalize
from contextwatch.services.detection.vhm_core import DriftMetrics, VHMEngine, VHMScoreResult

__all__ = [
    "DriftMetrics",
    "VHMEngine",
    "VHMScoreResult",
    "cosine_distance",
    "euclidean_distance",
    "normalize",
]
