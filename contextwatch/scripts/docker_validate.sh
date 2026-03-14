#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$ROOT_DIR/docker"
ENV_FILE="$ROOT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Missing env file at $ENV_FILE"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is not installed"
  exit 1
fi

FRESH=0
if [[ "${1:-}" == "--fresh" ]]; then
  FRESH=1
fi

cd "$DOCKER_DIR"

if [[ $FRESH -eq 1 ]]; then
  docker compose --env-file "$ENV_FILE" down -v
fi

docker compose --env-file "$ENV_FILE" up -d --build --force-recreate --remove-orphans

if ! docker compose --env-file "$ENV_FILE" ps --services --status running | grep -q '^neo4j$'; then
  echo "Neo4j not in running state; performing clean restart..."
  docker compose --env-file "$ENV_FILE" down -v
  docker compose --env-file "$ENV_FILE" up -d --build --force-recreate --remove-orphans
fi

echo "Waiting for API health..."
READY=0
for i in $(seq 1 120); do
  code="$(curl -s -o /tmp/contextwatch_health.json -w '%{http_code}' http://localhost:8000/health || true)"
  if [[ "$code" == "200" ]]; then
    READY=1
    break
  fi
  sleep 3
done

if [[ $READY -ne 1 ]]; then
  echo "API did not become healthy; retrying after clean restart..."
  docker compose --env-file "$ENV_FILE" down -v
  docker compose --env-file "$ENV_FILE" up -d --build --force-recreate --remove-orphans
  for i in $(seq 1 120); do
    code="$(curl -s -o /tmp/contextwatch_health.json -w '%{http_code}' http://localhost:8000/health || true)"
    if [[ "$code" == "200" ]]; then
      READY=1
      break
    fi
    sleep 3
  done
  if [[ $READY -ne 1 ]]; then
    echo "ERROR: API did not become healthy after retry"
    docker compose --env-file "$ENV_FILE" logs neo4j --tail 120 || true
    docker compose --env-file "$ENV_FILE" logs api --tail 120 || true
    exit 1
  fi
fi

API_KEY="$(grep '^API_KEY=' "$ENV_FILE" | cut -d= -f2-)"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: API_KEY is empty in $ENV_FILE"
  exit 1
fi

SESSION_ID="$(cat /proc/sys/kernel/random/uuid)"
LOG_ID_1="$(cat /proc/sys/kernel/random/uuid)"
LOG_ID_2="$(cat /proc/sys/kernel/random/uuid)"

echo "Running endpoint checks..."

health_code="$(curl -s -o /tmp/contextwatch_health.json -w '%{http_code}' http://localhost:8000/health)"
metrics_code="$(curl -s -o /tmp/contextwatch_metrics.json -w '%{http_code}' http://localhost:8000/metrics)"

no_key_code="$(curl -s -o /tmp/contextwatch_ingest_no_key.json -w '%{http_code}' \
  -X POST http://localhost:8000/ingest/log \
  -H 'Content-Type: application/json' \
  -d "{\"log_id\":\"$LOG_ID_1\",\"protocol\":\"MCP\",\"session\":{\"id\":\"mcp-session-${SESSION_ID:0:8}\",\"host\":\"Claude Desktop\",\"server\":\"figma-mcp-server v1.4.2\",\"connected_at\":\"2026-03-10T09:14:02.311Z\",\"transport\":\"stdio\"},\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\",\"logger\":\"figma-mcp-server\",\"data\":{\"timestamp\":\"2026-03-10T09:14:05.102Z\",\"event\":\"TOOL_CALL_RECEIVED\",\"message\":\"Tool invocation received: get_file\",\"meta\":{\"tool\":\"get_file\",\"arguments\":{\"file_key\":\"hU2yV3kPzXqN8mD1oL5rT9\"},\"request_id\":1,\"triggered_by\":\"user_prompt: 'Review dashboard'\"}}}}")"

with_key_code="$(curl -s -o /tmp/contextwatch_ingest_with_key.json -w '%{http_code}' \
  -X POST http://localhost:8000/ingest/log \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"log_id\":\"$LOG_ID_1\",\"protocol\":\"MCP\",\"session\":{\"id\":\"mcp-session-${SESSION_ID:0:8}\",\"host\":\"Claude Desktop\",\"server\":\"figma-mcp-server v1.4.2\",\"connected_at\":\"2026-03-10T09:14:02.311Z\",\"transport\":\"stdio\"},\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\",\"logger\":\"figma-mcp-server\",\"data\":{\"timestamp\":\"2026-03-10T09:14:05.901Z\",\"event\":\"TOOL_CALL_SUCCESS\",\"message\":\"Tool get_file completed.\",\"meta\":{\"tool\":\"get_file\",\"request_id\":1,\"response_content_type\":\"text\",\"payload_size_kb\":142.4,\"truncated\":false,\"triggered_by\":\"llm_reasoning: 'Return structured tree'\"}}}}")"

analyze_code="$(curl -s -o /tmp/contextwatch_analyze_with_key.json -w '%{http_code}' \
  -X POST http://localhost:8000/analyze \
  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"log_id\":\"$LOG_ID_2\",\"protocol\":\"MCP\",\"session\":{\"id\":\"mcp-session-${SESSION_ID:0:8}\",\"host\":\"Claude Desktop\",\"server\":\"figma-mcp-server v1.4.2\",\"connected_at\":\"2026-03-10T09:14:02.311Z\",\"transport\":\"stdio\"},\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"warning\",\"logger\":\"figma-mcp-server\",\"data\":{\"timestamp\":\"2026-03-10T09:14:09.103Z\",\"event\":\"FIGMA_API_RATE_LIMIT_APPROACHING\",\"message\":\"Figma API rate limit warning: 38 of 60 requests used.\",\"meta\":{\"tool\":\"get_node\",\"requests_used\":38,\"requests_limit\":60,\"window_resets_in_seconds\":34,\"recommendation\":\"Batch node requests\",\"triggered_by\":\"llm_reasoning: 'Inspect node for typography issues'\"}}}}")"

stats_code="$(curl -s -o /tmp/contextwatch_stats.json -w '%{http_code}' http://localhost:8000/stats)"
anomalies_code="$(curl -s -o /tmp/contextwatch_anomalies.json -w '%{http_code}' 'http://localhost:8000/anomalies?limit=5')"
session_graph_code="$(curl -s -o /tmp/contextwatch_session_graph.json -w '%{http_code}' "http://localhost:8000/graph/session/$SESSION_ID")"
trigger_graph_code="$(curl -s -o /tmp/contextwatch_trigger_graph.json -w '%{http_code}' "http://localhost:8000/graph/trigger/$LOG_ID_2")"
dashboard_code="$(curl -s -o /tmp/contextwatch_dashboard_headers.txt -w '%{http_code}' -I http://localhost:8501)"

check_code() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL: $name expected $expected got $actual"
    exit 1
  fi
}

check_code "health" "$health_code" "200"
check_code "metrics" "$metrics_code" "200"
check_code "ingest_no_key" "$no_key_code" "401"
check_code "ingest_with_key" "$with_key_code" "200"
check_code "analyze_with_key" "$analyze_code" "200"
check_code "stats" "$stats_code" "200"
check_code "anomalies" "$anomalies_code" "200"
check_code "session_graph" "$session_graph_code" "200"
check_code "trigger_graph" "$trigger_graph_code" "200"
check_code "dashboard_head" "$dashboard_code" "200"

echo "PASS: Docker stack validation succeeded"
echo "Session used: $SESSION_ID"
echo "Analyze log used: $LOG_ID_2"
