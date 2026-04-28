"""Vector math utilities implemented from first principles.

The implementations avoid any third‑party dependencies – only the stdlib
``math`` module is used.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable, Sequence

__all__ = [
    "dot_product",
    "l2_norm",
    "normalize",
    "cosine_similarity",
    "cosine_distance",
]


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product of two equal‑length numeric sequences."""
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length to compute dot product")
    return sum(x * y for x, y in zip(a, b))


def l2_norm(v: Sequence[float]) -> float:
    """Return the Euclidean L2 norm of *v*."""
    return sqrt(sum(x * x for x in v))


def normalize(v: Sequence[float]) -> list[float]:
    """Return *v* normalised to unit magnitude."""
    norm = l2_norm(v)
    if norm == 0.0:
        return [0.0] * len(v)
    return [x / norm for x in v]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the cosine similarity of two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same length for cosine similarity")
    return dot_product(a, b) / (l2_norm(a) * l2_norm(b)) if l2_norm(a) and l2_norm(b) else 0.0


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine distance between two vectors."""
    similarity = cosine_similarity(a, b)
    # Clamp to avoid floating‑point drift
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity
