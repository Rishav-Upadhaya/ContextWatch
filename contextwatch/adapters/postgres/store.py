"""Postgres DB repository / adapter class."""
from typing import Any, Optional
import json

try:
    import psycopg
except ImportError:  # pragma: no cover - optional runtime dependency
    psycopg = None

from .models import StoredNormalLog, StoredAnomaly
from .schema import (
    CREATE_NORMAL_LOGS_TABLE,
    CREATE_ANOMALIES_TABLE,
    CREATE_GRAPH_SNAPSHOT_TABLE
)
from .queries import (
    UPSERT_NORMAL_LOG,
    SELECT_EMBEDDINGS,
    COUNT_NORMAL_LOGS,
    SELECT_NORMAL_LOGS,
    UPSERT_ANOMALY,
    COUNT_ANOMALIES_BY_TYPE,
    SELECT_ANOMALIES_BY_TYPE,
    COUNT_ALL_ANOMALIES,
    SELECT_ALL_ANOMALIES,
    UPSERT_GRAPH_SNAPSHOT,
    SELECT_GRAPH_SNAPSHOT
)

class PostgresLogStore:
    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or ""
        self._disabled = not self.dsn or psycopg is None
        self._conn = None
        self._connect_if_needed()

    def _connect_if_needed(self) -> bool:
        if self._disabled:
            return False
        if self._conn is not None:
            return True
        
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        for attempt, wait in enumerate([1, 2, 4]):
            try:
                self._conn = psycopg.connect(self.dsn, connect_timeout=3)
                self._conn.autocommit = True
                self._ensure_schema()
                return True
            except psycopg.Error as e:
                logger.warning("Postgres connection attempt %d/3 failed: %s", attempt + 1, e, exc_info=True)
                if attempt < 2:
                    time.sleep(wait)
            except Exception as e:
                logger.warning("Postgres connection unexpected failure %d/3: %s", attempt + 1, e, exc_info=True)
                if attempt < 2:
                    time.sleep(wait)
        
        self._conn = None
        return False

    def _ensure_schema(self) -> None:
        if self._conn is None:
            return
        with self._conn.cursor() as cursor:
            cursor.execute(CREATE_NORMAL_LOGS_TABLE)
            cursor.execute(CREATE_ANOMALIES_TABLE)
            cursor.execute(CREATE_GRAPH_SNAPSHOT_TABLE)

    # ── Normal Logs ───────────────────────────────────────────────────────────

    def insert_normal_log(self, record: StoredNormalLog) -> None:
        if not self._connect_if_needed() or self._conn is None:
            return
        with self._conn.cursor() as cursor:
            cursor.execute(
                UPSERT_NORMAL_LOG,
                (
                    record.log_id,
                    record.protocol,
                    record.normalized_text,
                    json.dumps(record.embedding),
                    json.dumps(record.trace_context or {}),
                ),
            )

    def fetch_embeddings(self, limit: int = 1000) -> list[list[float]]:
        if not self._connect_if_needed() or self._conn is None:
            return []
        with self._conn.cursor() as cursor:
            cursor.execute(SELECT_EMBEDDINGS, (limit,))
            rows = cursor.fetchall()
            
        embeddings: list[list[float]] = []
        for (payload,) in rows:
            if isinstance(payload, list):
                embeddings.append([float(value) for value in payload])
            elif isinstance(payload, str):
                embeddings.append([float(value) for value in json.loads(payload)])
        return embeddings

    def fetch_normal_logs(self, page: int = 1, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
        if not self._connect_if_needed() or self._conn is None:
            return ([], 0)
        offset = (page - 1) * limit
        with self._conn.cursor() as cursor:
            cursor.execute(COUNT_NORMAL_LOGS)
            total = int(cursor.fetchone()[0])
            cursor.execute(SELECT_NORMAL_LOGS, (limit, offset))
            rows = cursor.fetchall()
            
        items = [
            {
                "log_id": row[0],
                "protocol": row[1],
                "normalized_text": row[2],
                "trace_context": row[3] or {},
                "created_at": row[4].isoformat() if row[4] else None,
            }
            for row in rows
        ]
        return (items, total)

    # ── Anomalies ─────────────────────────────────────────────────────────────

    def upsert_anomaly(self, anomaly: StoredAnomaly) -> None:
        if not self._connect_if_needed() or self._conn is None:
            return
        with self._conn.cursor() as cursor:
            cursor.execute(
                UPSERT_ANOMALY,
                (
                    anomaly.anomaly_id,
                    anomaly.log_id,
                    anomaly.anomaly_type,
                    float(anomaly.score),
                    float(anomaly.confidence),
                    anomaly.explanation,
                    json.dumps(anomaly.details or {}),
                ),
            )

    def fetch_anomalies(
        self,
        page: int = 1,
        limit: int = 100,
        anomaly_type: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self._connect_if_needed() or self._conn is None:
            return ([], 0)

        offset = (page - 1) * limit
        with self._conn.cursor() as cursor:
            if anomaly_type:
                cursor.execute(COUNT_ANOMALIES_BY_TYPE, (anomaly_type,))
                total = int(cursor.fetchone()[0])
                cursor.execute(SELECT_ANOMALIES_BY_TYPE, (anomaly_type, limit, offset))
            else:
                cursor.execute(COUNT_ALL_ANOMALIES)
                total = int(cursor.fetchone()[0])
                cursor.execute(SELECT_ALL_ANOMALIES, (limit, offset))
            rows = cursor.fetchall()

        items = [
            {
                "anomaly_id": row[0],
                "log_id": row[1],
                "anomaly_type": row[2],
                "score": float(row[3]),
                "confidence": float(row[4]),
                "explanation": row[5],
                "properties": row[6] or {},
                "created_at": row[7].isoformat() if row[7] else None,
            }
            for row in rows
        ]
        return (items, total)

    # ── Graph Snapshot ────────────────────────────────────────────────────────

    def upsert_graph_snapshot(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
        if not self._connect_if_needed() or self._conn is None:
            return
        with self._conn.cursor() as cursor:
            cursor.execute(UPSERT_GRAPH_SNAPSHOT, (json.dumps(nodes), json.dumps(edges)))

    def fetch_graph_snapshot(self) -> dict[str, Any]:
        if not self._connect_if_needed() or self._conn is None:
            return {"nodes": [], "edges": []}
        with self._conn.cursor() as cursor:
            cursor.execute(SELECT_GRAPH_SNAPSHOT)
            row = cursor.fetchone()
        if not row:
            return {"nodes": [], "edges": []}
        return {
            "nodes": row[0] or [],
            "edges": row[1] or [],
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
