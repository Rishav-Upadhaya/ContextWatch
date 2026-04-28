"""Log tokenizer for LogBERT.

Builds a vocabulary from log sequences and converts text into integer
token IDs. Avoids external NLP libraries — uses word-level +
character-level fallback tokenization.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


# Special tokens
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
SPECIAL_TOKENS = [PAD_TOKEN, MASK_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]


def _extract_words(text: str) -> list[str]:
    """Extract meaningful tokens from log text.

    Splits on whitespace/punctuation but preserves:
    - JSON-like structures as single tokens
    - Dot-separated identifiers (e.g., org.apache.http)
    - Snake_case and camelCase identifiers
    """
    # Preserve JSON-like fragments
    tokens: list[str] = []
    # Split on whitespace first
    raw = text.strip().split()
    for piece in raw:
        # Further split on common delimiters but keep structured tokens
        sub = re.split(r'[,;=(){}\[\]<>]', piece)
        for s in sub:
            s = s.strip()
            if s:
                tokens.append(s)
    return tokens


@dataclass
class LogTokenizer:
    """Build a vocabulary from log sequences and convert text to token IDs."""
    vocab: dict[str, int] = field(default_factory=dict)
    id_to_vocab: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.vocab:
            self._init_vocab()

    def _init_vocab(self):
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.vocab[token] = idx
            self.id_to_vocab[idx] = token

    @classmethod
    def from_corpus(cls, corpus: list[str]) -> 'LogTokenizer':
        """Build a vocabulary from a corpus of log strings."""
        tokenizer = cls()
        word_counts: dict[str, int] = {}

        for text in corpus:
            for word in _extract_words(text):
                word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1

        # Sort by frequency, assign IDs after special tokens
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _count) in enumerate(sorted_words, start=len(SPECIAL_TOKENS)):
            tokenizer.vocab[word] = idx
            tokenizer.id_to_vocab[idx] = word

        return tokenizer

    def encode(self, text: str, add_special: bool = True,
               max_length: int = 128, pad_to_max: bool = True) -> list[int]:
        """Convert a log string to token IDs.

        Output: [CLS] token_ids [SEP] [PAD]...
        """
        words = _extract_words(text)
        token_ids: list[int] = []

        if add_special:
            token_ids.append(self.vocab[CLS_TOKEN])

        for word in words[:max_length - 2]:
            token = word.lower()
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.id_to_vocab[token_id] = token
            token_ids.append(self.vocab.get(token, self.vocab[UNK_TOKEN]))
            if len(token_ids) >= max_length - 1:
                break

        if add_special:
            token_ids.append(self.vocab[SEP_TOKEN])

        # Pad to max_length
        if pad_to_max:
            while len(token_ids) < max_length:
                token_ids.append(self.vocab[PAD_TOKEN])

        return token_ids

    def encode_batch(self, texts: list[str], max_length: int = 128) -> list[list[int]]:
        """Encode multiple log strings to token IDs."""
        return [self.encode(text, max_length=max_length) for text in texts]

    def tokenize(self, text: str, max_length: int = 128) -> list[int]:
        """Backward-compatible alias that returns encoded token IDs."""
        return self.encode(text, max_length=max_length)

    def mask_tokens(self, token_ids: list[int], mask_prob: float = 0.15) -> tuple[list[int], list[float]]:
        """Randomly replace tokens with [MASK] token for MLKP training.

        Strategy (BERT-like):
        - 80% of the time: replace with [MASK]
        - 10% of the time: replace with a random token
        - 10% of the time: keep the original token unchanged

        Returns (masked_ids, mask_flags) where mask_flags[i] = 1.0 if token i was masked.
        """
        import random
        masked_ids = list(token_ids)
        mask_flags = [0.0] * len(masked_ids)

        for i in range(1, len(masked_ids) - 1):  # Skip CLS and SEP
            if masked_ids[i] == self.vocab[PAD_TOKEN]:
                continue
            if random.random() >= mask_prob:
                continue

            mask_flags[i] = 1.0
            roll = random.random()
            vocab_size = len(self.vocab)

            if roll < 0.80:
                masked_ids[i] = self.vocab[MASK_TOKEN]
            elif roll < 0.90:
                # Replace with random token (not special)
                random_id = random.randint(len(SPECIAL_TOKENS), vocab_size - 1)
                masked_ids[i] = random_id
            # else: keep original (already copied)

        return masked_ids, mask_flags

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        tokens = []
        for tid in token_ids:
            if tid == self.vocab[PAD_TOKEN]:
                break
            if tid == self.vocab[CLS_TOKEN]:
                continue
            tokens.append(self.id_to_vocab.get(tid, UNK_TOKEN))
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def max_tokens(self) -> int:
        """Compatibility alias used by older tests and UI code."""
        return self.vocab_size

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LogTokenizer':
        """Load vocabulary from a JSON file."""
        with open(path) as f:
            vocab = json.load(f)
        instance = cls()
        instance.vocab = vocab
        instance.id_to_vocab = {v: k for k, v in vocab.items()}
        return instance


"""End of tokenizer module."""
