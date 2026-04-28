#!/usr/bin/env python3
"""
Diagnostic script to analyze VHM embedding distribution and radius settings.

Generates embedding distribution analysis to understand why VHM radii are too small,
causing over-classification of logs as anomalies.

Output: contextwatch/data/diagnostics/radius_analysis.json
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextwatch.services.ai.training_data import load_system_training_corpus, log_to_training_text
from contextwatch.services.ai.logbert import LogBERTEncoder, LogBERTConfig
from contextwatch.utils.tokenizer import LogTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_embedding_distribution(
    corpus_size: int = 1000,
    normal_limit: int = 500,
    anomaly_limit: int = 500,
) -> dict[str, Any]:
    """Analyze embedding distribution for normal vs anomalous logs."""
    
    config = LogBERTConfig()
    tokenizer = LogTokenizer()
    encoder = LogBERTEncoder(config=config)
    
    logger.info("Loading training corpus...")
    corpus = load_system_training_corpus(
        max_normal=normal_limit,
        max_anomaly=anomaly_limit,
    )
    logger.info(
        f"Loaded {len(corpus.normal_logs)} normal, "
        f"{len(corpus.anomaly_logs)} anomaly logs"
    )
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    normal_embeddings: list[np.ndarray] = []
    anomaly_embeddings: list[np.ndarray] = []
    
    for log in corpus.normal_logs[:normal_limit]:
        text = log_to_training_text(log)
        token_ids = tokenizer.encode(text, max_length=config.max_seq_len)
        emb = np.array(encoder.forward(token_ids), dtype=np.float32)
        normal_embeddings.append(emb)
    
    for log in corpus.anomaly_logs[:anomaly_limit]:
        text = log_to_training_text(log)
        token_ids = tokenizer.encode(text, max_length=config.max_seq_len)
        emb = np.array(encoder.forward(token_ids), dtype=np.float32)
        anomaly_embeddings.append(emb)
    
    logger.info(f"Generated {len(normal_embeddings)} normal, {len(anomaly_embeddings)} anomaly embeddings")
    
    # Compute statistics
    normal_embs = np.array(normal_embeddings, dtype=np.float32)
    anomaly_embs = np.array(anomaly_embeddings, dtype=np.float32)
    
    # Intra-class distances
    logger.info("Computing intra-class distances...")
    normal_distances = compute_pairwise_distances(normal_embs)
    anomaly_distances = compute_pairwise_distances(anomaly_embs)
    
    # Inter-class distances
    inter_distances = compute_cross_distances(normal_embs, anomaly_embs)
    
    # Analysis
    analysis = {
        "embedding_count": {
            "normal": len(normal_embeddings),
            "anomaly": len(anomaly_embeddings),
            "total": len(normal_embeddings) + len(anomaly_embeddings),
        },
        "embedding_dimension": config.d_model,
        "intra_class_distances": {
            "normal": {
                "mean": float(np.mean(normal_distances)),
                "std": float(np.std(normal_distances)),
                "min": float(np.min(normal_distances)),
                "max": float(np.max(normal_distances)),
                "percentile_50": float(np.percentile(normal_distances, 50)),
                "percentile_75": float(np.percentile(normal_distances, 75)),
                "percentile_90": float(np.percentile(normal_distances, 90)),
                "percentile_95": float(np.percentile(normal_distances, 95)),
                "percentile_99": float(np.percentile(normal_distances, 99)),
                "count": len(normal_distances),
            },
            "anomaly": {
                "mean": float(np.mean(anomaly_distances)),
                "std": float(np.std(anomaly_distances)),
                "min": float(np.min(anomaly_distances)),
                "max": float(np.max(anomaly_distances)),
                "percentile_50": float(np.percentile(anomaly_distances, 50)),
                "percentile_75": float(np.percentile(anomaly_distances, 75)),
                "percentile_90": float(np.percentile(anomaly_distances, 90)),
                "percentile_95": float(np.percentile(anomaly_distances, 95)),
                "percentile_99": float(np.percentile(anomaly_distances, 99)),
                "count": len(anomaly_distances),
            },
        },
        "inter_class_distances": {
            "mean": float(np.mean(inter_distances)),
            "std": float(np.std(inter_distances)),
            "min": float(np.min(inter_distances)),
            "max": float(np.max(inter_distances)),
            "percentile_25": float(np.percentile(inter_distances, 25)),
            "percentile_50": float(np.percentile(inter_distances, 50)),
            "percentile_75": float(np.percentile(inter_distances, 75)),
            "percentile_90": float(np.percentile(inter_distances, 90)),
            "count": len(inter_distances),
        },
        "diagnosis": diagnose_overlap(
            normal_distances,
            anomaly_distances,
            inter_distances,
        ),
    }
    
    return analysis


def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute all pairwise cosine distances within a set of embeddings."""
    # Normalize
    embs_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    # Cosine distance matrix
    distances = 1.0 - np.dot(embs_norm, embs_norm.T)
    # Extract upper triangle (avoid self-distances)
    upper_triangle = np.triu_indices_from(distances, k=1)
    return distances[upper_triangle]


def compute_cross_distances(normal_embs: np.ndarray, anomaly_embs: np.ndarray) -> np.ndarray:
    """Compute all pairwise cosine distances between normal and anomaly embeddings."""
    # Normalize
    normal_norm = normal_embs / (np.linalg.norm(normal_embs, axis=1, keepdims=True) + 1e-8)
    anomaly_norm = anomaly_embs / (np.linalg.norm(anomaly_embs, axis=1, keepdims=True) + 1e-8)
    # Cosine distance matrix
    distances = 1.0 - np.dot(normal_norm, anomaly_norm.T)
    return distances.flatten()


def diagnose_overlap(
    normal_distances: np.ndarray,
    anomaly_distances: np.ndarray,
    inter_distances: np.ndarray,
) -> dict[str, Any]:
    """Diagnose why radius might be too small."""
    
    normal_95 = np.percentile(normal_distances, 95)
    anomaly_95 = np.percentile(anomaly_distances, 95)
    inter_mean = np.mean(inter_distances)
    inter_std = np.std(inter_distances)
    
    # Check overlap
    normal_max = np.max(normal_distances)
    anomaly_min = np.min(anomaly_distances)
    inter_overlap_ratio = len(inter_distances[inter_distances < normal_95]) / len(inter_distances)
    
    diagnosis: dict[str, Any] = {
        "vhm_radius_95th_percentile_normal": float(normal_95),
        "vhm_radius_95th_percentile_anomaly": float(anomaly_95),
        "inter_class_mean_distance": float(inter_mean),
        "inter_class_std_distance": float(inter_std),
        "normal_max_distance": float(normal_max),
        "anomaly_min_distance": float(anomaly_min),
        "inter_distance_within_normal_radius_ratio": float(inter_overlap_ratio),
        "potential_issues": [],
        "recommendations": [],
    }
    
    # Issue 1: Small radius absolute values
    if normal_95 < 0.1:
        diagnosis["potential_issues"].append(
            "SMALL_ABSOLUTE_RADIUS: VHM radius (95th percentile) < 0.1, "
            "indicating very tight clustering. This will cause over-classification."
        )
        diagnosis["recommendations"].append(
            "REC_1: Generate more diverse training data to increase intra-class spread."
        )
    
    # Issue 2: High overlap between classes
    if inter_overlap_ratio > 0.8:
        diagnosis["potential_issues"].append(
            f"HIGH_CLASS_OVERLAP: {inter_overlap_ratio*100:.1f}% of normal-vs-anomaly distances "
            "fall within normal class radius, indicating embeddings too similar."
        )
        diagnosis["recommendations"].append(
            "REC_2: Create more distinct error type variations in training data "
            "(tool hallucination, context poisoning, registry overflow, delegation failure)."
        )
    
    # Issue 3: Insufficient anomaly diversity
    if len(anomaly_distances) < 1000:
        diagnosis["potential_issues"].append(
            f"INSUFFICIENT_ANOMALY_SAMPLES: Only {len(anomaly_distances)} anomaly samples. "
            "Current training uses max 1000 anomalies, mostly binary classification."
        )
        diagnosis["recommendations"].append(
            "REC_3: Generate at least 50K anomaly samples balanced across 4 error types "
            "(TOOL_HALLUCINATION, CONTEXT_POISONING, REGISTRY_OVERFLOW, DELEGATION_CHAIN_FAILURE)."
        )
    
    # Issue 4: Close anomaly/normal clusters
    if inter_mean < normal_95 * 1.5:
        diagnosis["potential_issues"].append(
            f"CLOSE_CLASS_CENTERS: Inter-class distance mean ({inter_mean:.4f}) "
            f"is only {inter_mean/normal_95:.2f}x the normal radius threshold. "
            "Normal and anomaly clusters too close."
        )
        diagnosis["recommendations"].append(
            "REC_4: Either (a) increase radius_quantile from 0.95 to 0.75 or (b) generate "
            "more diverse/extreme anomalies."
        )
    
    # Issue 5: High variance in anomaly distances
    anomaly_cv = np.std(anomaly_distances) / (np.mean(anomaly_distances) + 1e-8)
    if anomaly_cv > 1.0:
        diagnosis["potential_issues"].append(
            f"HIGH_ANOMALY_VARIANCE: Coefficient of variation {anomaly_cv:.2f} indicates "
            "anomalies are too heterogeneous. Some may be too close to normal."
        )
        diagnosis["recommendations"].append(
            "REC_5: Label anomalies by type and cluster them separately during training."
        )
    
    return diagnosis


def save_analysis(analysis: dict[str, Any]) -> None:
    """Save analysis to JSON file."""
    output_dir = Path(__file__).resolve().parent.parent / "contextwatch" / "data" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "radius_analysis.json"
    
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis saved to {output_file}")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ContextWatch VHM Radius Analysis")
    logger.info("=" * 80)
    
    analysis = analyze_embedding_distribution(
        corpus_size=1000,
        normal_limit=500,
        anomaly_limit=500,
    )
    
    print("\n" + "=" * 80)
    print("EMBEDDING DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(json.dumps(analysis, indent=2))
    print("=" * 80)
    
    save_analysis(analysis)
    
    print("\nKey Findings:")
    for issue in analysis["diagnosis"]["potential_issues"]:
        print(f"  ❌ {issue}")
    print("\nRecommendations:")
    for rec in analysis["diagnosis"]["recommendations"]:
        print(f"  ✓ {rec}")
