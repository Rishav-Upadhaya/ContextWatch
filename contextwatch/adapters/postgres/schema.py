"""SQL schema (DDL) strings for ContextWatch postgres tables."""

CREATE_NORMAL_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS contextwatch_normal_logs (
    id BIGSERIAL PRIMARY KEY,
    log_id TEXT UNIQUE NOT NULL,
    protocol TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    embedding JSONB NOT NULL,
    trace_context JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

CREATE_ANOMALIES_TABLE = """
CREATE TABLE IF NOT EXISTS contextwatch_anomalies (
    id BIGSERIAL PRIMARY KEY,
    anomaly_id TEXT UNIQUE NOT NULL,
    log_id TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    explanation TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

CREATE_ANOMALY_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_anomalies_type ON contextwatch_anomalies(anomaly_type)",
    "CREATE INDEX IF NOT EXISTS idx_anomalies_created ON contextwatch_anomalies(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_anomalies_log_id ON contextwatch_anomalies(log_id)",
    "CREATE INDEX IF NOT EXISTS idx_anomalies_type_created ON contextwatch_anomalies(anomaly_type, created_at)",
]

CREATE_GRAPH_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS contextwatch_graph_snapshot (
    id SMALLINT PRIMARY KEY DEFAULT 1,
    nodes JSONB NOT NULL,
    edges JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT singleton_graph CHECK (id = 1)
)
"""
