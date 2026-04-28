"""
ML-based 4-way error type classifier using logistic regression on log embeddings.

Used as fallback when rule-based classifier has low confidence (<70%).

Supports training and inference on LogBERT embeddings (64-dim vectors).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ErrorTypeClassifierML:
    """
    Logistic regression classifier for 4-way error type classification.
    
    Maps embeddings to: TOOL_HALLUCINATION, CONTEXT_POISONING, 
                       REGISTRY_OVERFLOW, DELEGATION_CHAIN_FAILURE
    """
    
    ERROR_TYPES = [
        "TOOL_HALLUCINATION",
        "CONTEXT_POISONING",
        "REGISTRY_OVERFLOW",
        "DELEGATION_CHAIN_FAILURE",
    ]
    
    def __init__(self, embedding_dim: int = 64, learning_rate: float = 0.01, epochs: int = 100):
        """Initialize classifier weights."""
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_trained = False
        
        # Weights: [num_classes, embedding_dim + 1] (include bias)
        self.weights = np.random.randn(len(self.ERROR_TYPES), embedding_dim + 1) * 0.01
        self.weights[:, -1] = 0  # Initialize bias to 0
    
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / (e.sum(axis=1, keepdims=True) + 1e-9)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for embeddings.
        
        Args:
            embeddings: [batch_size, embedding_dim]
        
        Returns:
            probabilities: [batch_size, num_classes] with softmax probabilities
        """
        # Add bias term
        X_with_bias = np.hstack([embeddings, np.ones((len(embeddings), 1))])
        logits = X_with_bias @ self.weights.T
        return self.softmax(logits)
    
    def predict(self, embeddings: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Predict error types for embeddings.
        
        Returns:
            (error_types, confidences)
        """
        probs = self.predict_proba(embeddings)
        class_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        error_types = [self.ERROR_TYPES[idx] for idx in class_indices]
        return error_types, confidences
    
    def train(
        self,
        embeddings: List[np.ndarray],
        labels: List[str],
        val_embeddings: Optional[List[np.ndarray]] = None,
        val_labels: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train classifier using batch gradient descent.
        
        Args:
            embeddings: List of embedding vectors [N, 64]
            labels: List of error type labels (must be in ERROR_TYPES)
            val_embeddings: Optional validation set
            val_labels: Optional validation labels
        
        Returns:
            history: Dict with train/val loss over epochs
        """
        X = np.array(embeddings, dtype=np.float32)
        
        # Convert labels to one-hot
        label_to_idx = {etype: i for i, etype in enumerate(self.ERROR_TYPES)}
        y_indices = np.array([label_to_idx[label] for label in labels], dtype=np.int32)
        y = np.eye(len(self.ERROR_TYPES))[y_indices]  # One-hot
        
        X_with_bias = np.hstack([X, np.ones((len(X), 1))])
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.epochs):
            # Forward pass
            logits = X_with_bias @ self.weights.T
            probs = self.softmax(logits)
            
            # Cross-entropy loss
            eps = 1e-9
            loss = -np.mean(np.sum(y * np.log(probs + eps), axis=1))
            history["train_loss"].append(float(loss))
            
            # Backward pass (gradient for cross-entropy + softmax)
            error = probs - y  # [N, 4]
            grad = X_with_bias.T @ error / len(X)  # [65, 4]
            
            # Update weights
            self.weights -= self.learning_rate * grad.T
            
            # Validation loss
            if val_embeddings is not None:
                val_X = np.array(val_embeddings, dtype=np.float32)
                val_X_with_bias = np.hstack([val_X, np.ones((len(val_X), 1))])
                val_logits = val_X_with_bias @ self.weights.T
                val_probs = self.softmax(val_logits)
                
                val_y_indices = np.array([label_to_idx[label] for label in val_labels])
                val_y = np.eye(len(self.ERROR_TYPES))[val_y_indices]
                
                val_loss = -np.mean(np.sum(val_y * np.log(val_probs + eps), axis=1))
                history["val_loss"].append(float(val_loss))
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs}: train_loss={loss:.4f}")
        
        self.is_trained = True
        logger.info(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")
        
        return history
    
    def save_weights(self, path: Path) -> None:
        """Save weights to numpy file."""
        np.save(path, self.weights)
        logger.info(f"Weights saved to {path}")
    
    def load_weights(self, path: Path) -> None:
        """Load weights from numpy file."""
        self.weights = np.load(path)
        self.is_trained = True
        logger.info(f"Weights loaded from {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "embedding_dim": self.embedding_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "is_trained": self.is_trained,
            "weights": self.weights.tolist(),
            "error_types": self.ERROR_TYPES,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorTypeClassifierML":
        """Deserialize from dictionary."""
        obj = cls(
            embedding_dim=data["embedding_dim"],
            learning_rate=data["learning_rate"],
            epochs=data["epochs"],
        )
        obj.weights = np.array(data["weights"], dtype=np.float32)
        obj.is_trained = data["is_trained"]
        return obj


class EnsembleErrorClassifier:
    """
    Ensemble combining rule-based and ML classifiers.
    
    Strategy:
    1. Try rule-based classifier
    2. If confidence >= 70%, use it
    3. Otherwise, use ML classifier as fallback
    """
    
    def __init__(self, ml_classifier: Optional[ErrorTypeClassifierML] = None):
        self.ml_classifier = ml_classifier or ErrorTypeClassifierML()
        self.rule_based_threshold = 0.70
    
    def classify(
        self,
        log: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float, str]:
        """
        Classify log using ensemble strategy.
        
        Args:
            log: Full log dictionary (for rule-based classifier)
            embedding: Optional embedding vector (for ML classifier fallback)
        
        Returns:
            (error_type, confidence, explanation)
        """
        from contextwatch.services.detection.error_classifier import RuleBasedErrorClassifier
        
        # Try rule-based first
        rule_type, rule_conf, rule_reason = RuleBasedErrorClassifier.classify(log)
        
        if rule_type and rule_conf >= self.rule_based_threshold:
            return rule_type, rule_conf, f"Rule-based: {rule_reason}"
        
        # Fallback to ML if embedding available
        if embedding is not None and self.ml_classifier.is_trained:
            embedding_arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
            types, confs = self.ml_classifier.predict(embedding_arr)
            ml_type = types[0]
            ml_conf = confs[0]
            
            if rule_type:
                # Both methods available - report both
                reason = f"ML fallback (rule={rule_type}@{rule_conf:.2f}, ml={ml_type}@{ml_conf:.2f})"
            else:
                reason = f"ML-only: {ml_type}@{ml_conf:.2f}"
            
            return ml_type, float(ml_conf), reason
        
        # Neither rule nor ML confident
        if rule_type:
            return rule_type, rule_conf, f"Rule (low conf): {rule_reason}"
        
        return None, 0.0, "No reliable classifier match"


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs
    print("ErrorTypeClassifierML initialized successfully")
    
    classifier = ErrorTypeClassifierML(embedding_dim=64, epochs=20)
    
    # Test with dummy embeddings
    dummy_embeddings = [np.random.randn(64) for _ in range(10)]
    dummy_labels = [
        "TOOL_HALLUCINATION", "CONTEXT_POISONING",
        "REGISTRY_OVERFLOW", "DELEGATION_CHAIN_FAILURE",
    ] * 2 + ["TOOL_HALLUCINATION"] * 2
    
    print("Training...")
    history = classifier.train(dummy_embeddings, dummy_labels, verbose=False)
    print(f"Final loss: {history['train_loss'][-1]:.4f}")
    
    print("Predicting...")
    types, confs = classifier.predict(np.array(dummy_embeddings[:3], dtype=np.float32))
    for t, c in zip(types, confs):
        print(f"  {t:30s} confidence={c:.3f}")
