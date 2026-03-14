from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from core.schema import AnomalyResult, ClassificationResult, NormalizedLog, RCAResult


@dataclass
class ProcessedLog:
    normalized: NormalizedLog
    anomaly: AnomalyResult
    classification: ClassificationResult
    explanation: str | None
    rca: RCAResult | None
    processed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class InMemoryStore:
    all_logs: dict[str, ProcessedLog] = field(default_factory=dict)
    anomalies: dict[str, ProcessedLog] = field(default_factory=dict)

    def upsert(self, item: ProcessedLog) -> None:
        self.all_logs[item.normalized.log_id] = item
        if item.anomaly.is_anomaly:
            self.anomalies[item.normalized.log_id] = item

    def anomaly_list(self) -> list[ProcessedLog]:
        return sorted(self.anomalies.values(), key=lambda x: x.normalized.timestamp, reverse=True)

    def latest_in_session(self, session_id: str, exclude_log_id: str | None = None) -> ProcessedLog | None:
        candidates = [
            item
            for item in self.all_logs.values()
            if item.normalized.session_id == session_id and item.normalized.log_id != exclude_log_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.normalized.timestamp)


class DurableStore(InMemoryStore):
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()
        self._load_into_memory()

    def _ensure_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_logs (
                    log_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    is_anomaly INTEGER NOT NULL,
                    anomaly_type TEXT,
                    anomaly_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    normalized_json TEXT NOT NULL,
                    anomaly_json TEXT NOT NULL,
                    classification_json TEXT NOT NULL,
                    explanation TEXT,
                    rca_json TEXT,
                    processed_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_session ON processed_logs(session_id, timestamp)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_anomaly ON processed_logs(is_anomaly, timestamp)"
            )

    def _row_to_processed(self, row: sqlite3.Row) -> ProcessedLog:
        normalized_payload = json.loads(row["normalized_json"])
        ts = normalized_payload.get("timestamp")
        if isinstance(ts, str):
            normalized_payload["timestamp"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        normalized = NormalizedLog.model_validate(normalized_payload)
        anomaly = AnomalyResult.model_validate(json.loads(row["anomaly_json"]))
        classification = ClassificationResult.model_validate(json.loads(row["classification_json"]))
        rca_json = row["rca_json"]
        rca = RCAResult.model_validate(json.loads(rca_json)) if rca_json else None
        processed = ProcessedLog(
            normalized=normalized,
            anomaly=anomaly,
            classification=classification,
            explanation=row["explanation"],
            rca=rca,
        )
        processed.processed_at = datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else processed.processed_at
        return processed

    def _load_into_memory(self) -> None:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM processed_logs ORDER BY timestamp ASC").fetchall()
        for row in rows:
            item = self._row_to_processed(row)
            self.all_logs[item.normalized.log_id] = item
            if item.anomaly.is_anomaly:
                self.anomalies[item.normalized.log_id] = item

    def upsert(self, item: ProcessedLog) -> None:
        super().upsert(item)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO processed_logs (
                    log_id, session_id, timestamp, protocol, agent_id,
                    is_anomaly, anomaly_type, anomaly_score, confidence,
                    normalized_json, anomaly_json, classification_json,
                    explanation, rca_json, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(log_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    timestamp=excluded.timestamp,
                    protocol=excluded.protocol,
                    agent_id=excluded.agent_id,
                    is_anomaly=excluded.is_anomaly,
                    anomaly_type=excluded.anomaly_type,
                    anomaly_score=excluded.anomaly_score,
                    confidence=excluded.confidence,
                    normalized_json=excluded.normalized_json,
                    anomaly_json=excluded.anomaly_json,
                    classification_json=excluded.classification_json,
                    explanation=excluded.explanation,
                    rca_json=excluded.rca_json,
                    processed_at=excluded.processed_at
                """,
                (
                    item.normalized.log_id,
                    item.normalized.session_id,
                    item.normalized.timestamp.isoformat(),
                    item.normalized.protocol,
                    item.normalized.agent_id,
                    1 if item.anomaly.is_anomaly else 0,
                    item.classification.anomaly_type,
                    float(item.anomaly.anomaly_score),
                    float(item.classification.confidence),
                    json.dumps(item.normalized.model_dump(mode="json")),
                    json.dumps(item.anomaly.model_dump(mode="json")),
                    json.dumps(item.classification.model_dump(mode="json")),
                    item.explanation,
                    json.dumps(item.rca.model_dump(mode="json")) if item.rca else None,
                    item.processed_at.isoformat(),
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
