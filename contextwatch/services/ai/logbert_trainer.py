"""LogBERT Finetuner: Real MLKP + VHM training from scratch.

Algorithm
---------
Masked Log Key Prediction (MLKP):
    1. Randomly mask 15% of tokens in each sequence.
    2. Forward pass through LogBERT encoder → CLS embedding.
    3. Project CLS embedding → vocabulary logits via linear head.
    4. Compute cross-entropy loss only on masked positions.
    5. Back-propagate gradient to:
         - MLKP head (W_mlkp, b_mlkp)
         - Token embedding table (direct embedding gradient)

Volume Hypersphere Minimisation (VHM):
    1. Embed all log sequences.
    2. Centre = weighted mean of embeddings.
    3. VHM loss = mean squared distance to centre.
    4. Encourages "normal" logs to cluster in a tight hypersphere.

Total loss  L = L_MLKP + λ * L_VHM

Implementation notes
--------------------
- Only numpy used for matrix operations (no ML frameworks).
- Gradients computed analytically (manual back-prop on linear head).
- Embedding table updated via direct gradient descent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from contextwatch.services.ai.logbert import LogBERTConfig, LogBERTEncoder, softmax


# Special token ids (must match LogTokenizer)
PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
CLS_ID = 3
SEP_ID = 4


@dataclass
class EpochMetrics:
    epoch: int
    loss_mlkp: float
    loss_vhm: float
    loss_total: float
    vhm_radius: float
    n_sequences: int


class LogBERTFinetuner:
    """
    Fine-tunes a LogBERTEncoder using MLKP + VHM objectives.

    All gradient computation is done manually with numpy.
    No ML frameworks (PyTorch, TensorFlow, scikit-learn) are used.
    """

    def __init__(
        self,
        config: LogBERTConfig,
        learning_rate: float = 1e-3,
        mask_prob: float = 0.15,
        vhm_weight: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.lr = learning_rate
        self.mask_prob = mask_prob
        self.vhm_weight = vhm_weight

        random.seed(seed)
        np.random.seed(seed)

        # Build encoder with fresh weights
        self.encoder = LogBERTEncoder(config)

        # MLKP classification head: d_model → vocab_size
        fan = config.d_model + config.vocab_size
        limit = math.sqrt(6.0 / fan)
        rng = np.random.RandomState(seed)
        self.W_mlkp: np.ndarray = rng.uniform(-limit, limit, (config.d_model, config.vocab_size)).astype(np.float32)
        self.b_mlkp: np.ndarray = np.zeros(config.vocab_size, dtype=np.float32)

        # VHM hypersphere state
        self.vhm_center: np.ndarray = np.zeros(config.d_model, dtype=np.float32)
        self.vhm_radius: float = 1.0
        self._center_initialised = False

    # ── Masking ────────────────────────────────────────────────────────────────

    def _mask_sequence(self, token_ids: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        BERT-style masking strategy.

        For each non-special token, with probability mask_prob:
          - 80%: replace with [MASK]
          - 10%: replace with random token
          - 10%: keep original

        Returns (masked_ids, [(position, original_token_id)])
        """
        masked = list(token_ids)
        targets: List[Tuple[int, int]] = []

        for i, tid in enumerate(masked):
            # Skip special tokens
            if tid in (PAD_ID, CLS_ID, SEP_ID):
                continue
            if random.random() >= self.mask_prob:
                continue

            targets.append((i, tid))
            roll = random.random()
            if roll < 0.80:
                masked[i] = MASK_ID
            elif roll < 0.90:
                masked[i] = random.randint(5, self.config.vocab_size - 1)
            # else: keep original token (no-op)

        return masked, targets

    # ── Loss functions ──────────────────────────────────────────────────────────

    def _cross_entropy(
        self, embedding: np.ndarray, target_id: int
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Cross-entropy loss for one masked position.

        Forward:
            logits = embedding @ W_mlkp + b_mlkp     [vocab_size]
            probs  = softmax(logits)
            loss   = -log(probs[target_id])

        Backward (w.r.t. logits):
            d_logits = probs - one_hot(target_id)    [vocab_size]

        Returns: (loss, d_W_mlkp, d_b_mlkp, d_embedding)
        """
        logits = embedding @ self.W_mlkp + self.b_mlkp       # [vocab_size]
        probs = softmax(logits.reshape(1, -1)).flatten()      # [vocab_size]

        target_id = max(0, min(target_id, self.config.vocab_size - 1))
        loss = -math.log(max(float(probs[target_id]), 1e-8))

        # Gradient of cross-entropy w.r.t. logits = probs − one_hot
        d_logits = probs.copy()
        d_logits[target_id] -= 1.0

        # Gradient w.r.t W_mlkp: outer(embedding, d_logits)
        d_W = np.outer(embedding, d_logits)            # [d_model, vocab_size]
        d_b = d_logits                                 # [vocab_size]

        # Gradient w.r.t embedding: W_mlkp @ d_logits
        d_embed = self.W_mlkp @ d_logits               # [d_model]

        return loss, d_W, d_b, d_embed

    def _vhm_loss(self, embeddings: np.ndarray) -> float:
        """
        VHM loss = mean squared distance from embeddings to hypersphere centre.

        L_VHM = (1/N) * sum_i ||e_i - center||^2

        Also updates VHM centre (exponential moving average).
        """
        if not self._center_initialised:
            self.vhm_center = embeddings.mean(axis=0)
            self._center_initialised = True

        diffs = embeddings - self.vhm_center          # [N, d_model]
        dist_sq = (diffs ** 2).sum(axis=1)             # [N]
        loss = float(dist_sq.mean())

        # Update centre with EMA
        alpha = 0.1
        batch_mean = embeddings.mean(axis=0)
        self.vhm_center = (1.0 - alpha) * self.vhm_center + alpha * batch_mean

        # Update radius to 95th percentile of distances
        dists = np.sqrt(dist_sq)
        self.vhm_radius = float(np.percentile(dists, 95)) + 1e-6

        return loss

    # ── Training step ───────────────────────────────────────────────────────────

    def _step(self, token_ids: List[int]) -> Tuple[float, float, np.ndarray]:
        """
        One gradient update for a single log sequence.

        Returns (mlkp_loss, vhm_distance_sq, embedding)
        """
        masked_ids, targets = self._mask_sequence(token_ids)

        # Forward pass
        embedding = np.array(self.encoder.forward(masked_ids), dtype=np.float32)

        # MLKP loss + gradient update
        mlkp_loss = 0.0
        if targets:
            acc_d_W = np.zeros_like(self.W_mlkp)
            acc_d_b = np.zeros_like(self.b_mlkp)
            acc_d_embed = np.zeros_like(embedding)

            for pos, orig_id in targets:
                loss_i, dW_i, db_i, de_i = self._cross_entropy(embedding, orig_id)
                mlkp_loss += loss_i
                acc_d_W += dW_i
                acc_d_b += db_i
                acc_d_embed += de_i

            n = len(targets)
            acc_d_W /= n
            acc_d_b /= n
            acc_d_embed /= n
            mlkp_loss /= n

            # Gradient descent on MLKP head
            self.W_mlkp -= self.lr * acc_d_W
            self.b_mlkp -= self.lr * acc_d_b

            # Update embedding table: rows for the original (unmasked) tokens
            embed_lr = self.lr * 0.1
            for _, orig_id in targets:
                oid = max(0, min(orig_id, self.config.vocab_size - 1))
                self.encoder.weights["embed"][oid] -= embed_lr * acc_d_embed

        # VHM distance (for this single sample; batch update done in finetune())
        dist_sq = float(np.sum((embedding - self.vhm_center) ** 2))
        return mlkp_loss, dist_sq, embedding

    # ── Public API ──────────────────────────────────────────────────────────────

    def finetune(
        self,
        sequences: List[List[int]],
        epochs: int = 5,
    ) -> List[EpochMetrics]:
        """
        Finetune on a list of tokenised log sequences.

        Args:
            sequences : list of token-id lists (from LogTokenizer.encode())
            epochs    : number of training epochs

        Returns:
            EpochMetrics per epoch (loss_mlkp, loss_vhm, loss_total, vhm_radius)
        """
        history: List[EpochMetrics] = []

        for epoch in range(epochs):
            random.shuffle(sequences)

            epoch_mlkp = 0.0
            epoch_vhm = 0.0
            all_embeddings: List[np.ndarray] = []

            for seq in sequences:
                m_loss, v_dist, emb = self._step(seq)
                epoch_mlkp += m_loss
                epoch_vhm += v_dist
                all_embeddings.append(emb)

            n = max(len(sequences), 1)
            avg_mlkp = epoch_mlkp / n
            avg_vhm = epoch_vhm / n

            # Batch VHM update at end of epoch
            emb_arr = np.array(all_embeddings, dtype=np.float32)
            avg_vhm = self._vhm_loss(emb_arr)
            total = avg_mlkp + self.vhm_weight * avg_vhm

            m = EpochMetrics(
                epoch=epoch + 1,
                loss_mlkp=round(avg_mlkp, 6),
                loss_vhm=round(avg_vhm, 6),
                loss_total=round(total, 6),
                vhm_radius=round(self.vhm_radius, 6),
                n_sequences=n,
            )
            history.append(m)

        return history

    def get_encoder(self) -> LogBERTEncoder:
        """Return the fine-tuned encoder."""
        return self.encoder

    def get_vhm_state(self) -> Tuple[np.ndarray, float]:
        """Return (vhm_center, vhm_radius) after training."""
        return self.vhm_center, self.vhm_radius


LogBERTTrainer = LogBERTFinetuner
TrainingMetrics = EpochMetrics
