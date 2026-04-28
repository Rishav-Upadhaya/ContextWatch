"""Vector store adapters used by pipeline and MA-RCA.

Default behavior is an in-memory store to keep startup deterministic and avoid
hard dependency on optional PostgreSQL vector drivers in API boot paths.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StoredVector:
    embedding: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InMemoryVectorStore:
    """Simple in-memory vector/document store.

    Supports both embedding storage for VHM baselines and text-oriented
    similarity retrieval for MA-RCA hypotheses.
    """

    def __init__(self):
        self._items: List[StoredVector] = []

    def add(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        for emb, meta in zip(embeddings, metadatas):
            self._items.append(
                StoredVector(
                    embedding=[float(x) for x in emb],
                    metadata=dict(meta or {}),
                )
            )

    def items(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    def count(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def similarity_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return [(doc_id, score, metadata)] ranked by lightweight token overlap.

        This intentionally avoids heavyweight dependencies and works for boot-time
        retrieval use-cases where metadata commonly contains explanatory text.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for item in self._items:
            meta = item.metadata or {}
            haystack = " ".join(
                [
                    str(meta.get("text", "")),
                    str(meta.get("description", "")),
                    str(meta.get("event", "")),
                    str(meta.get("type", "")),
                    str(meta.get("log_id", "")),
                ]
            )
            score = self._jaccard(q_tokens, self._tokenize(haystack))
            if score <= 0:
                continue
            doc_id = str(meta.get("doc_id") or meta.get("log_id") or f"mem_{len(scored)}")
            scored.append((doc_id, float(score), meta))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(0, int(top_k))]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text = (text or "").lower().strip()
        if not text:
            return set()
        return {tok for tok in text.replace("\n", " ").split(" ") if tok}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)


class PGVectorStore:
    """Optional PostgreSQL-backed vector store.

    Uses psycopg (v3). This adapter is not required for default runtime and is
    only instantiated when VECTOR_STORE_BACKEND=postgres.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.getenv("DATABASE_URL", "")
        self.conn = None

    def _get_conn(self):
        if not self.dsn:
            raise RuntimeError("PGVectorStore requires DATABASE_URL/DSN")
        if self.conn is None or self.conn.closed:
            import psycopg

            self.conn = psycopg.connect(self.dsn)
        return self.conn

    def add(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        conn = self._get_conn()
        with conn.cursor() as cur:
            query = (
                "INSERT INTO contextwatch_normal_logs "
                "(log_id, protocol, normalized_text, embedding, trace_context) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (log_id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, "
                "trace_context = EXCLUDED.trace_context"
            )
            for emb, meta in zip(embeddings, metadatas):
                cur.execute(
                    query,
                    (
                        str(meta.get("log_id") or meta.get("doc_id") or "unknown"),
                        str(meta.get("protocol", "MCP")),
                        str(meta.get("text", "")),
                        json.dumps([float(x) for x in emb]),
                        json.dumps(meta),
                    ),
                )
            conn.commit()

    def items(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT embedding, trace_context FROM contextwatch_normal_logs")
            rows = cur.fetchall()
        out = []
        for emb, meta in rows:
            out.append({"embedding": emb if isinstance(emb, list) else [], "metadata": meta or {}})
        return out

    def count(self) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM contextwatch_normal_logs")
            return int(cur.fetchone()[0])

    def clear(self) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE contextwatch_normal_logs")
            conn.commit()

    def similarity_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        # Fallback behavior: use metadata text overlap over stored rows.
        # Keeps a stable API even without pgvector extension.
        mem = InMemoryVectorStore()
        for item in self.items():
            emb = item.get("embedding", [])
            meta = item.get("metadata", {})
            mem.add([emb if isinstance(emb, list) else []], [meta if isinstance(meta, dict) else {}])
        return mem.similarity_search(query=query, top_k=top_k)


def _make_default_store():
    backend = os.getenv("VECTOR_STORE_BACKEND", "memory").strip().lower()
    if backend == "postgres":
        return PGVectorStore()
    return InMemoryVectorStore()


VectorStore = _make_default_store().__class__
