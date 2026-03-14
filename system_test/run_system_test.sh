#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

MCP_FILE="$BASE_DIR/mcp_logs.jsonl"
A2A_FILE="$BASE_DIR/a2a_logs.jsonl"
MIXED_FILE="$BASE_DIR/mixed_logs.jsonl"

echo "[1/7] Health check"
curl -s "$API_URL/health" | jq '.data'

echo "[2/7] Ingest MCP-only logs"
curl -s -X POST "$API_URL/ingest/file" -F "file=@$MCP_FILE" | jq '.data'

echo "[3/7] Ingest A2A-only logs"
curl -s -X POST "$API_URL/ingest/file" -F "file=@$A2A_FILE" | jq '.data'

echo "[4/7] Ingest mixed MCP+A2A logs"
curl -s -X POST "$API_URL/ingest/file" -F "file=@$MIXED_FILE" | jq '.data'

echo "[5/7] Check stats"
curl -s "$API_URL/stats" | jq '.data'

echo "[6/7] Fetch anomalies (first 5)"
curl -s "$API_URL/anomalies" | jq '.data[:5]'

echo "[7/7] Pick first anomaly and run RCA"
LOG_ID=$(curl -s "$API_URL/anomalies" | jq -r '.data[0].log_id // empty')
if [[ -n "$LOG_ID" ]]; then
  echo "Using log_id=$LOG_ID"
  curl -s "$API_URL/anomalies/$LOG_ID" | jq '.data | {log_id: .log.log_id, anomaly_type: .classification.anomaly_type, confidence: .anomaly.confidence}'
  curl -s "$API_URL/graph/rca/$LOG_ID" | jq '.data | {root_cause_log_id, hop_count, causal_chain}'
else
  echo "No anomalies found yet."
fi

echo "System test completed."
