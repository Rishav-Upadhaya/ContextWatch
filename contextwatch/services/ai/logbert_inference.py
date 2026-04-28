"""LogBERT Inference: Load trained weights and encode log sequences.

Provides embedding generation and anomaly scoring via VHM distance.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

import numpy as np

from contextwatch.utils.weight_manager import ModelConfig, WeightManager
from contextwatch.services.detection.vhm_engine import VHMEngine, euclidean_distance

logger = logging.getLogger(__name__)


class LogBERTInference:
    """Load trained LogBERT weights and encode sequences to embeddings.
    
    Pipeline for anomaly detection:
    1. Load pretrained weights (or train if missing)
    2. Encode log sequence → d_model dimensional embedding
    3. Compute VHM distance to normal hypersphere
    4. Return anomaly score [0, 1] based on distance
    """

    def __init__(self, weight_manager: WeightManager):
        self.weight_manager = weight_manager
        self.config = weight_manager.load_config()

        # Load weights
        self.encoder = weight_manager.load_encoder_weights()
        self.attention = weight_manager.load_attention_weights()
        self.mlkp_head = weight_manager.load_mlkp_head()

        # Load VHM parameters
        vhm_params = weight_manager.load_vhm_params()
        if vhm_params:
            self.vhm_center, self.vhm_radius, self.vhm_volume = vhm_params
        else:
            self.vhm_center = None
            self.vhm_radius = 1.0
            self.vhm_volume = 1.0

        self.is_ready = (
            self.encoder is not None
            and self.attention is not None
            and self.mlkp_head is not None
            and self.vhm_center is not None
        )

    def encode_sequence(self, token_sequence: List[int]) -> np.ndarray:
        """Encode a token sequence to embedding vector.
        
        Args:
            token_sequence: List of token IDs
            
        Returns:
            Embedding vector [d_model]
        """
        if not self.is_ready:
            raise RuntimeError("Model not ready; load weights first")

        # Truncate or pad to max length
        seq = token_sequence[:self.config.seq_len_max]
        seq = seq + [0] * (self.config.seq_len_max - len(seq))

        # Normalize sequence
        seq_array = np.array(seq, dtype=np.float32)
        seq_array = seq_array / (self.config.vocab_size + 1.0)

        # Pass through encoder (simplified)
        # In full implementation: apply each transformer layer with attention
        x = seq_array
        for layer in range(self.config.n_layers):
            # Simplified: matrix projection
            if self.encoder is not None and layer < self.encoder.shape[0]:
                projection = self.encoder[layer]
                # Pad or slice to match dimensions
                if len(x) >= projection.shape[0]:
                    x = x[:projection.shape[0]]
                else:
                    x = np.pad(x, (0, projection.shape[0] - len(x)), 'constant')
                # Matrix multiply approximation
                x = x @ projection[:, :min(len(x), projection.shape[1])]

        # Extract embedding (final state or average pooling)
        if len(x) >= self.config.d_model:
            embedding = x[:self.config.d_model]
        else:
            embedding = np.pad(x, (0, self.config.d_model - len(x)), 'constant')

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def encode_batch(self, sequences: List[List[int]]) -> np.ndarray:
        """Encode multiple sequences.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            Embeddings [batch_size, d_model]
        """
        embeddings = []
        for seq in sequences:
            emb = self.encode_sequence(seq)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)

    def compute_anomaly_score(self, embedding: np.ndarray) -> Tuple[float, float]:
        """Compute anomaly score using VHM distance.
        
        Args:
            embedding: Single embedding vector
            
        Returns:
            (anomaly_score, confidence) where:
            - anomaly_score: 0-1, higher = more anomalous
            - confidence: 0-1, model confidence in the score
        """
        if self.vhm_center is None:
            raise RuntimeError("VHM center not loaded")

        # Compute distance to center
        distance = euclidean_distance(embedding, self.vhm_center)

        # Normalize distance to score [0, 1]
        # Distance <= radius → normal, distance >> radius → anomalous
        margin = 2.0
        normalized_dist = distance / (self.vhm_radius * margin)

        # Sigmoid-like: asymptotic to 0-1 range
        anomaly_score = min(1.0, max(0.0, normalized_dist))

        # Confidence: higher when we're far from boundary
        boundary_margin = abs(distance - self.vhm_radius)
        if boundary_margin > self.vhm_radius * 0.5:
            confidence = 0.95
        elif boundary_margin > self.vhm_radius * 0.1:
            confidence = 0.85
        else:
            confidence = 0.70  # Low confidence when near decision boundary

        return float(anomaly_score), float(confidence)

    def score_batch(
        self, sequences: List[List[int]]
    ) -> List[Tuple[float, float]]:
        """Score a batch of sequences.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            List of (anomaly_score, confidence) tuples
        """
        embeddings = self.encode_batch(sequences)
        scores = []
        for emb in embeddings:
            score, conf = self.compute_anomaly_score(emb)
            scores.append((score, conf))
        return scores

    def get_statistics(self) -> dict:
        """Return model statistics for debugging."""
        return {
            "model_ready": self.is_ready,
            "config": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "vocab_size": self.config.vocab_size,
            },
            "vhm": {
                "center_norm": float(np.linalg.norm(self.vhm_center)) if self.vhm_center is not None else None,
                "radius": self.vhm_radius,
                "volume": self.vhm_volume,
            },
        }


def load_or_train_model(
    version: str = "v1",
    force_retrain: bool = False,
    training_sequences: Optional[List[List[int]]] = None,
) -> LogBERTInference:
    """Load model or train if missing.
    
    Args:
        version: Model version (v1, v2, etc.)
        force_retrain: Force retraining even if weights exist
        training_sequences: Sequences to train on (required if training)
        
    Returns:
        LogBERTInference instance ready for inference
    """
    weight_manager = WeightManager(version)

    if not force_retrain and weight_manager.has_weights():
        logger.info(f"✓ Loading pretrained LogBERT {version}")
        return LogBERTInference(weight_manager)

    # Train new model
    if training_sequences is None or len(training_sequences) == 0:
        raise ValueError(
            "Training sequences required when weights don't exist. "
            "Provide training_sequences parameter."
        )

    logger.info(f"Training LogBERT {version}...")
    config = ModelConfig()
    trainer = LogBERTTrainer(
        config=config,
        learning_rate=0.001,
        vhm_weight=0.5,
    )

    metrics = trainer.train(
        sequences=training_sequences,
        epochs=10,
        batch_size=32,
    )

    trainer.save_weights(weight_manager)
    weight_manager.save_training_metadata(
        num_epochs=10,
        final_loss_mlkp=metrics[-1].loss_mlkp,
        final_loss_vhm=metrics[-1].loss_vhm,
        dataset_size=len(training_sequences),
        dataset_name="synthetic_training",
    )

    logger.info(f"✓ Model training complete. Weights saved.")
    return LogBERTInference(weight_manager)


# Import trainer only if needed
try:
    from contextwatch.services.ai.logbert_trainer import LogBERTTrainer
except ImportError as e:
    logger.warning("LogBERTTrainer module not available. Fine-tuning features will be disabled.", exc_info=True)
    LogBERTTrainer = None  # Will fail gracefully if not available
