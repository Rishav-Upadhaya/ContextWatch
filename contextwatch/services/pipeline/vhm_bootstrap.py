
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from contextwatch.services.detection.vhm_core import VHMEngine

logger = logging.getLogger(__name__)

def bootstrap_vhm(vhm: VHMEngine, logbert, normal_logs: List[Dict[str, Any]], anomaly_logs: List[Dict[str, Any]], tokenizer=None, min_samples: int = 20):
    """
    Safe VHM bootstrap with validation.
    Implements Level 2 and Level 3 of the calibration strategy.
    
    Args:
        vhm: VHMEngine instance to calibrate
        logbert: LogBERTInference instance for encoding
        normal_logs: List of normal log dictionaries  
        anomaly_logs: List of anomalous log dictionaries
        tokenizer: Optional LogTokenizer instance; if None, assumes logbert.tokenizer exists
        min_samples: Minimum number of normal samples required for calibration
    """
    logger.info(f"🚀 Starting VHM Bootstrap: Normal={len(normal_logs)}, Anomalies={len(anomaly_logs)}")
    
    if len(normal_logs) < min_samples:
        logger.warning(f"VHM needs at least {min_samples} normal logs to calibrate. Got {len(normal_logs)}. Using fallback scale.")
        vhm.decision_scale = 3.0
        vhm._cluster_thresholds = vhm._cluster_radii * 3.0
        vhm._refresh_public_state()
        return {"status": "fallback", "decision_scale": 3.0}

    try:
        def get_signal(log):
            return log["params"]["data"]["message"]

        def encode_fn(log):
            # Use provided tokenizer, or fall back to logbert's tokenizer
            tok = tokenizer if tokenizer is not None else logbert.tokenizer
            tokens = tok.tokenize(get_signal(log))
            return logbert.encode_sequence(tokens)

        logger.info(f"📊 Encoding {len(normal_logs)} normal logs...")
        normal_embs = [encode_fn(l) for l in normal_logs]
        logger.info(f"📊 Encoding {len(anomaly_logs)} anomaly logs...")
        anomaly_embs = [encode_fn(l) for l in anomaly_logs]
        
        normal_embs_np = np.array(normal_embs, dtype=np.float32)
        anomaly_embs_np = np.array(anomaly_embs, dtype=np.float32)
        
        logger.info(f"📐 Normal embeddings shape: {normal_embs_np.shape}")
        logger.info(f"📐 Anomaly embeddings shape: {anomaly_embs_np.shape}")

        logger.info("🔧 Running vhm.fit()...")
        vhm.fit(normal_embs_np)
        logger.info(f"✓ fit() complete. Radius before calibration: {vhm.radius:.6f}")
        
        # Fixed: Pass NumPy arrays directly to calibrate instead of converting to lists
        logger.info("🎯 Running vhm.calibrate()...")
        result = vhm.calibrate(normal_embs_np, anomaly_embs_np)
        
        logger.info(f"📊 Calibration result: F1={result.get('best_f1', 0):.4f}, "
                   f"Scale={result.get('decision_scale', 1.0):.4f}, "
                   f"Radius={result.get('decision_radius', 0):.6f}")
        
        if result["best_f1"] < 0.5:
            logger.warning(f"VHM calibration F1={result['best_f1']:.4f} is low. Falling back to scale 3.0")
            vhm.decision_scale = 3.0
            vhm._cluster_thresholds = vhm._cluster_radii * 3.0
            vhm._refresh_public_state()
            return {"status": "fallback", "best_f1": float(result["best_f1"])}
            
        logger.info(f"✓ VHM Calibrated Successfully: F1={result['best_f1']:.4f}, Scale={result['decision_scale']:.4f}")
        return result

    except Exception as e:
        logger.error(f"VHM Bootstrap failed: {e}", exc_info=True)
        vhm.decision_scale = 3.0
        vhm._refresh_public_state()
        return {"status": "error", "error": str(e)}
