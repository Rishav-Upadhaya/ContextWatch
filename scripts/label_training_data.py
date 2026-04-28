#!/usr/bin/env python3
"""
Backfill anomaly_type labels for training datasets.

Updates:
  1. golden_dataset/golden_anomalies.jsonl - label 250 anomalies per type
  2. Validates training_balanced_v2.jsonl - ensure all have labels
  3. Prepares for ML classifier training

Output: Labeled versions of datasets ready for training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextwatch.services.detection.error_classifier import RuleBasedErrorClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_golden_anomalies() -> None:
    """
    Backfill anomaly_type labels for golden anomaly dataset.
    
    Strategy:
    1. Read golden_anomalies.jsonl
    2. For each log, use rule-based classifier to assign anomaly_type if missing
    3. Write updated version to golden_anomalies_labeled.jsonl
    """
    
    data_dir = Path(__file__).resolve().parent.parent / "contextwatch" / "data"
    input_file = data_dir / "golden_dataset" / "golden_anomalies.jsonl"
    output_file = data_dir / "golden_dataset" / "golden_anomalies_labeled.jsonl"
    
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return
    
    logger.info(f"Reading golden anomalies from {input_file}...")
    logs_by_type: Dict[str, List[Dict[str, Any]]] = {}
    unlabeled_count = 0
    already_labeled_count = 0
    
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                log = json.loads(line.strip())
            except json.JSONDecodeError:
                logger.warning(f"Line {line_num}: JSON decode failed")
                continue
            
            # Check if already labeled
            existing_type = log.get("anomaly_type")
            if existing_type and existing_type in [
                "TOOL_HALLUCINATION", "CONTEXT_POISONING",
                "REGISTRY_OVERFLOW", "DELEGATION_CHAIN_FAILURE"
            ]:
                already_labeled_count += 1
                atype = existing_type
            else:
                # Classify using rule-based classifier
                classified_type, conf, reason = RuleBasedErrorClassifier.classify(log)
                if classified_type and conf >= 0.50:  # Lower threshold for backfill
                    log["anomaly_type"] = classified_type
                    atype = classified_type
                else:
                    # Fallback: use explicit event or default
                    atype = log.get("event", "TOOL_HALLUCINATION").replace("TOOL_", "")
                    log["anomaly_type"] = atype
                unlabeled_count += 1
            
            if atype not in logs_by_type:
                logs_by_type[atype] = []
            logs_by_type[atype].append(log)
    
    logger.info(f"Backfilled {unlabeled_count} labels, {already_labeled_count} already labeled")
    logger.info(f"Distribution: {[(k, len(v)) for k, v in sorted(logs_by_type.items())]}")
    
    # Write labeled output
    with open(output_file, "w") as f:
        for logs in logs_by_type.values():
            for log in logs:
                f.write(json.dumps(log) + "\n")
    
    logger.info(f"✅ Labeled golden anomalies saved to {output_file}")
    return logs_by_type


def validate_balanced_dataset() -> None:
    """Verify that training_balanced_v2.jsonl has labels on all records."""
    
    data_dir = Path(__file__).resolve().parent.parent / "contextwatch" / "data"
    file_path = data_dir / "training" / "training_balanced_v2.jsonl"
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return
    
    logger.info(f"Validating balanced dataset: {file_path}...")
    
    counts_by_type = {}
    missing_label_count = 0
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                log = json.loads(line.strip())
            except json.JSONDecodeError:
                logger.warning(f"Line {line_num}: JSON decode failed")
                continue
            
            # Check for label
            atype = log.get("anomaly_type")
            if not atype:
                missing_label_count += 1
                if missing_label_count <= 5:
                    logger.warning(f"Line {line_num}: Missing anomaly_type")
            
            counts_by_type[atype] = counts_by_type.get(atype, 0) + 1
    
    logger.info(f"\nBalanced dataset validation:")
    total = sum(counts_by_type.values())
    logger.info(f"  Total: {total} logs")
    logger.info(f"  Missing labels: {missing_label_count}")
    
    for atype in ["TOOL_HALLUCINATION", "CONTEXT_POISONING", "REGISTRY_OVERFLOW", "DELEGATION_CHAIN_FAILURE", "NORMAL"]:
        count = counts_by_type.get(atype, 0)
        pct = 100.0 * count / total if total > 0 else 0
        logger.info(f"  {atype:30s}: {count:6d} ({pct:5.1f}%)")
    
    if missing_label_count == 0:
        logger.info("✅ All logs have labels!")
    else:
        logger.warning(f"⚠️ {missing_label_count} logs missing labels")


def calculate_training_split_recommendations() -> str:
    """
    Calculate recommended train/val/test split for balanced dataset.
    """
    
    data_dir = Path(__file__).resolve().parent.parent / "contextwatch" / "data"
    file_path = data_dir / "training" / "training_balanced_v2.jsonl"
    
    if not file_path.exists():
        return "File not found"
    
    counts_by_type = {}
    
    with open(file_path, "r") as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                atype = log.get("anomaly_type", "UNKNOWN")
                counts_by_type[atype] = counts_by_type.get(atype, 0) + 1
            except:
                pass
    
    total = sum(counts_by_type.values())
    
    # Recommended split: 70% train, 15% val, 15% test
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    recommendations = "Training Split Recommendations:\n"
    recommendations += f"  Total: {total:,} logs\n"
    recommendations += f"\n  Train (70%): {int(total * train_ratio):,} logs\n"
    
    for atype in sorted(counts_by_type.keys()):
        count = counts_by_type[atype]
        train_count = int(count * train_ratio)
        recommendations += f"    {atype:30s}: {train_count:6,}\n"
    
    recommendations += f"\n  Val (15%): {int(total * val_ratio):,} logs\n"
    for atype in sorted(counts_by_type.keys()):
        count = counts_by_type[atype]
        val_count = int(count * val_ratio)
        recommendations += f"    {atype:30s}: {val_count:6,}\n"
    
    recommendations += f"\n  Test (15%): {int(total * test_ratio):,} logs\n"
    for atype in sorted(counts_by_type.keys()):
        count = counts_by_type[atype]
        test_count = int(count * test_ratio)
        recommendations += f"    {atype:30s}: {test_count:6,}\n"
    
    return recommendations


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("TRAINING DATA LABELING")
    logger.info("=" * 80)
    
    logger.info("\n1. Backfilling golden anomalies...")
    backfill_golden_anomalies()
    
    logger.info("\n2. Validating balanced dataset...")
    validate_balanced_dataset()
    
    logger.info("\n3. Training split recommendations...")
    recommendations = calculate_training_split_recommendations()
    print("\n" + recommendations)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training data labeling complete!")
    logger.info("=" * 80)
