from __future__ import annotations

import json
import time
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import Settings
from core.schema import NormalizedLog


class LogEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NORMAL,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_batch(self, logs: list[NormalizedLog], batch_size: int | None = None) -> None:
        if not logs:
            return
        chunk_size = batch_size or self.settings.BATCH_SIZE
        for i in tqdm(range(0, len(logs), chunk_size), desc="Embedding logs"):
            chunk = logs[i : i + chunk_size]
            texts = [item.text_for_embedding for item in chunk]
            embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self.collection.add(
                ids=[item.log_id for item in chunk],
                embeddings=embeddings.tolist(),
                metadatas=[self._sanitize_metadata(item.metadata) for item in chunk],
                documents=texts,
            )

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                output[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                output[key] = value
            else:
                output[key] = json.dumps(value, default=str)
        return output

    def get_nearest_normal(self, log: NormalizedLog, n_results: int = 5) -> list[dict[str, Any]]:
        vector = self.model.encode([log.text_for_embedding], convert_to_numpy=True, normalize_embeddings=True)
        results = self.collection.query(
            query_embeddings=vector.tolist(),
            n_results=n_results,
            include=["distances", "metadatas"],
        )
        output: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for log_id, distance, metadata in zip(ids, distances, metadatas):
            output.append({"log_id": log_id, "distance": float(distance), "metadata": metadata})
        return output

    def compute_anomaly_score(self, log: NormalizedLog) -> float:
        nearest = self.get_nearest_normal(log, n_results=5)
        if not nearest:
            return 1.0
        score = min(item["distance"] for item in nearest)
        return float(max(0.0, min(2.0, score)))

    def benchmark_single_latency(self, log: NormalizedLog) -> float:
        start = time.perf_counter()
        _ = self.compute_anomaly_score(log)
        return (time.perf_counter() - start) * 1000


def cosine_similarity_from_embeddings(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    return float(np.dot(v1, v2))
