#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Running quick smoke test with mixed logs..."
curl -s "$API_URL/health" | jq '.data'
curl -s -X POST "$API_URL/ingest/file" -F "file=@$BASE_DIR/mixed_logs.jsonl" | jq '.data'
curl -s "$API_URL/stats" | jq '.data'
