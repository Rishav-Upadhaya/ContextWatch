"""SQL queries (DML) strings for ContextWatch postgres tables."""

UPSERT_NORMAL_LOG = """
INSERT INTO contextwatch_normal_logs (log_id, protocol, normalized_text, embedding, trace_context)
VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
ON CONFLICT (log_id) DO UPDATE SET
    protocol = EXCLUDED.protocol,
    normalized_text = EXCLUDED.normalized_text,
    embedding = EXCLUDED.embedding,
    trace_context = EXCLUDED.trace_context,
    created_at = NOW()
"""

SELECT_EMBEDDINGS = """
SELECT embedding
FROM contextwatch_normal_logs
ORDER BY created_at ASC
LIMIT %s
"""

COUNT_NORMAL_LOGS = "SELECT COUNT(*) FROM contextwatch_normal_logs"

SELECT_NORMAL_LOGS = """
SELECT log_id, protocol, normalized_text, trace_context, created_at
FROM contextwatch_normal_logs
ORDER BY created_at DESC
LIMIT %s OFFSET %s
"""

UPSERT_ANOMALY = """
INSERT INTO contextwatch_anomalies (
    anomaly_id, log_id, anomaly_type, score, confidence, explanation, details
)
VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT (anomaly_id) DO UPDATE SET
    log_id = EXCLUDED.log_id,
    anomaly_type = EXCLUDED.anomaly_type,
    score = EXCLUDED.score,
    confidence = EXCLUDED.confidence,
    explanation = EXCLUDED.explanation,
    details = EXCLUDED.details,
    created_at = NOW()
"""

COUNT_ANOMALIES_BY_TYPE = "SELECT COUNT(*) FROM contextwatch_anomalies WHERE anomaly_type = %s"

SELECT_ANOMALIES_BY_TYPE = """
SELECT anomaly_id, log_id, anomaly_type, score, confidence, explanation, details, created_at
FROM contextwatch_anomalies
WHERE anomaly_type = %s
ORDER BY created_at DESC
LIMIT %s OFFSET %s
"""

COUNT_ALL_ANOMALIES = "SELECT COUNT(*) FROM contextwatch_anomalies"

SELECT_ALL_ANOMALIES = """
SELECT anomaly_id, log_id, anomaly_type, score, confidence, explanation, details, created_at
FROM contextwatch_anomalies
ORDER BY created_at DESC
LIMIT %s OFFSET %s
"""

UPSERT_GRAPH_SNAPSHOT = """
INSERT INTO contextwatch_graph_snapshot (id, nodes, edges)
VALUES (1, %s::jsonb, %s::jsonb)
ON CONFLICT (id) DO UPDATE SET
    nodes = EXCLUDED.nodes,
    edges = EXCLUDED.edges,
    updated_at = NOW()
"""

SELECT_GRAPH_SNAPSHOT = "SELECT nodes, edges FROM contextwatch_graph_snapshot WHERE id = 1"
