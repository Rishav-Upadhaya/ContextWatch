"""contextwatch.services.ai.logbert
===================================
Pure NumPy LogBERT encoder for semantic log representation.

This module implements core transformer components from first principles:
sinusoidal positional encoding, multi-head attention, feed-forward layers,
GELU activation, and layer normalization.

Used by: contextwatch.services.pipeline.pipeline
Depends on: numpy
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class LogBERTConfig:
    vocab_size: int = 5000
    d_model: int = 64       # Small enough for fast pure-math inference
    n_heads: int = 4        # d_model must be divisible by n_heads
    d_k: int = 16           # d_model // n_heads
    d_v: int = 16           # d_model // n_heads
    d_ff: int = 256         # 4 * d_model
    max_seq_len: int = 64
    n_layers: int = 2


# ── Activation functions (from scratch) ──────────────────────────────────────

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit — same formula as in original BERT."""
    return x * 0.5 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer normalisation: zero-mean, unit-variance, then affine transform."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# ── Positional encoding (from scratch, sinusoidal) ───────────────────────────

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal PE from 'Attention is All You Need' (Vaswani et al., 2017)."""
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pos = np.arange(seq_len, dtype=np.float32).reshape(-1, 1)
    dims = np.arange(0, d_model, 2, dtype=np.float32)
    pe[:, 0::2] = np.sin(pos / (10000.0 ** (dims / d_model)))
    pe[:, 1::2] = np.cos(pos / (10000.0 ** (dims / d_model)))
    return pe


# ── Multi-head self-attention (from scratch) ──────────────────────────────────

def multi_head_attention(
    X: np.ndarray,      # [seq_len, d_model]
    W_Q: np.ndarray,    # [n_heads, d_model, d_k]
    W_K: np.ndarray,    # [n_heads, d_model, d_k]
    W_V: np.ndarray,    # [n_heads, d_model, d_v]
    W_O: np.ndarray,    # [n_heads * d_v, d_model]
    n_heads: int,
    d_k: int,
) -> np.ndarray:
    """
    Scaled dot-product multi-head self-attention.

    For each head h:
        Q_h = X @ W_Q[h]               # [seq, d_k]
        K_h = X @ W_K[h]               # [seq, d_k]
        V_h = X @ W_V[h]               # [seq, d_v]
        A_h = softmax(Q_h @ K_h.T / sqrt(d_k)) @ V_h   # [seq, d_v]

    concat(A_0, ..., A_{H-1}) @ W_O   # [seq, d_model]
    """
    scale = math.sqrt(d_k)
    head_outputs = []
    for h in range(n_heads):
        Q = X @ W_Q[h]                        # [seq, d_k]
        K = X @ W_K[h]                        # [seq, d_k]
        V = X @ W_V[h]                        # [seq, d_v]
        scores = (Q @ K.T) / scale            # [seq, seq]
        attn = softmax(scores)                # [seq, seq]
        head_outputs.append(attn @ V)         # [seq, d_v]

    concat = np.concatenate(head_outputs, axis=-1)   # [seq, n_heads*d_v]
    return concat @ W_O                              # [seq, d_model]


# ── Position-wise feed-forward network (from scratch) ────────────────────────

def feed_forward(
    X: np.ndarray,   # [seq_len, d_model]
    W1: np.ndarray,  # [d_model, d_ff]
    b1: np.ndarray,  # [d_ff]
    W2: np.ndarray,  # [d_ff, d_model]
    b2: np.ndarray,  # [d_model]
) -> np.ndarray:
    """Two-layer FFN with GELU: Linear(d_model→d_ff) → GELU → Linear(d_ff→d_model)."""
    hidden = gelu(X @ W1 + b1)   # [seq, d_ff]
    return hidden @ W2 + b2       # [seq, d_model]


# ── LogBERT Encoder ───────────────────────────────────────────────────────────

class LogBERTEncoder:
    """Encodes tokenized logs into dense semantic embeddings.

    Attributes:
        config: Static model dimensions and transformer depth parameters.
        weights: Trainable matrices/vectors for all encoder layers.
        _pe_cache: Precomputed sinusoidal positional encodings.
    """

    def __init__(self, config: LogBERTConfig, weights: Optional[Dict] = None):
        """Initializes the encoder and allocates model weights.

        Args:
            config: LogBERT architecture configuration.
            weights: Optional preloaded model parameters.
        """
        self.config = config
        self.weights = weights if weights is not None else self._init_weights()
        self._pe_cache = sinusoidal_positional_encoding(config.max_seq_len, config.d_model)

    # ── Weight initialisation ────────────────────────────────────────────────

    def _xavier(self, rng: np.random.RandomState, rows: int, cols: int) -> np.ndarray:
        """Xavier / Glorot uniform initialisation."""
        limit = math.sqrt(6.0 / (rows + cols))
        return rng.uniform(-limit, limit, (rows, cols)).astype(np.float32)

    def _init_weights(self) -> Dict:
        """Initialise all trainable parameters with Xavier uniform."""
        rng = np.random.RandomState(42)
        cfg = self.config
        w: Dict = {}

        # Token embedding table: vocab_size × d_model
        w["embed"] = self._xavier(rng, cfg.vocab_size, cfg.d_model)

        for l in range(cfg.n_layers):
            # Multi-head attention projections (stored per-head for clarity)
            w[f"W_Q_{l}"] = np.stack([self._xavier(rng, cfg.d_model, cfg.d_k) for _ in range(cfg.n_heads)])
            w[f"W_K_{l}"] = np.stack([self._xavier(rng, cfg.d_model, cfg.d_k) for _ in range(cfg.n_heads)])
            w[f"W_V_{l}"] = np.stack([self._xavier(rng, cfg.d_model, cfg.d_v) for _ in range(cfg.n_heads)])
            w[f"W_O_{l}"] = self._xavier(rng, cfg.n_heads * cfg.d_v, cfg.d_model)

            # LayerNorm 1 (after attention + residual)
            w[f"ln1_g_{l}"] = np.ones(cfg.d_model, dtype=np.float32)
            w[f"ln1_b_{l}"] = np.zeros(cfg.d_model, dtype=np.float32)

            # Feed-forward
            w[f"W1_{l}"] = self._xavier(rng, cfg.d_model, cfg.d_ff)
            w[f"b1_{l}"] = np.zeros(cfg.d_ff, dtype=np.float32)
            w[f"W2_{l}"] = self._xavier(rng, cfg.d_ff, cfg.d_model)
            w[f"b2_{l}"] = np.zeros(cfg.d_model, dtype=np.float32)

            # LayerNorm 2 (after FFN + residual)
            w[f"ln2_g_{l}"] = np.ones(cfg.d_model, dtype=np.float32)
            w[f"ln2_b_{l}"] = np.zeros(cfg.d_model, dtype=np.float32)

        return w

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, token_ids: List[int]) -> List[float]:
        """Encodes a token sequence and returns the CLS embedding.

        Args:
            token_ids: Integer token IDs produced by the tokenizer.

        Returns:
            List of floats with length equal to ``config.d_model``.

        Raises:
            ValueError: When downstream NumPy operations receive empty input.
        """
        cfg = self.config
        seq_len = min(len(token_ids), cfg.max_seq_len)
        ids = [max(0, min(tid, cfg.vocab_size - 1)) for tid in token_ids[:seq_len]]

        # 1. Embedding + positional encoding
        X = self.weights["embed"][ids] + self._pe_cache[:seq_len]  # [seq, d_model]

        # 2. Transformer encoder layers
        for l in range(cfg.n_layers):
            # Self-attention sublayer
            attn = multi_head_attention(
                X,
                W_Q=self.weights[f"W_Q_{l}"],
                W_K=self.weights[f"W_K_{l}"],
                W_V=self.weights[f"W_V_{l}"],
                W_O=self.weights[f"W_O_{l}"],
                n_heads=cfg.n_heads,
                d_k=cfg.d_k,
            )
            X = layer_norm(X + attn, self.weights[f"ln1_g_{l}"], self.weights[f"ln1_b_{l}"])

            # FFN sublayer
            ffn = feed_forward(
                X,
                W1=self.weights[f"W1_{l}"],
                b1=self.weights[f"b1_{l}"],
                W2=self.weights[f"W2_{l}"],
                b2=self.weights[f"b2_{l}"],
            )
            X = layer_norm(X + ffn, self.weights[f"ln2_g_{l}"], self.weights[f"ln2_b_{l}"])

        # 3. Return CLS token embedding (position 0)
        return X[0].tolist()


"""End of logbert module."""
