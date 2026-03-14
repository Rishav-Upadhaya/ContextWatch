# System Test Resources (MCP + A2A)

This folder contains ready-to-run resources for end-to-end system validation.

## Files

- `mcp_logs.jsonl` → MCP-only unlabeled logs (real-world style)
- `a2a_logs.jsonl` → A2A-only unlabeled logs (real-world style)
- `mixed_logs.jsonl` → Combined MCP + A2A logs
- `quick_smoke.sh` → Fast smoke test (health + mixed ingest + stats)
- `run_system_test.sh` → Full test flow across endpoints

## Prerequisites

- API running at `http://localhost:8000`
- `jq` installed for pretty JSON output
- Docker stack up (if using containers):

```bash
cd contextwatch/docker
docker compose up -d --build api dashboard neo4j
```

## Test Plan A: MCP-only validation

```bash
curl -s -X POST http://localhost:8000/ingest/file \
  -F "file=@system_test/mcp_logs.jsonl" | jq '.data'

curl -s http://localhost:8000/stats | jq '.data'
```

Expected:
- `inserted` equals number of MCP lines
- `errors` equals `0`
- `total_logs` increases

## Test Plan B: A2A-only validation

```bash
curl -s -X POST http://localhost:8000/ingest/file \
  -F "file=@system_test/a2a_logs.jsonl" | jq '.data'

curl -s http://localhost:8000/stats | jq '.data'
```

Expected:
- `inserted` equals number of A2A lines
- `errors` equals `0`

## Test Plan C: Mixed MCP + A2A validation

```bash
curl -s -X POST http://localhost:8000/ingest/file \
  -F "file=@system_test/mixed_logs.jsonl" | jq '.data'

curl -s http://localhost:8000/anomalies | jq '.data | length'
```

Expected:
- file ingests with `errors=0`
- anomaly list endpoint responds normally

## Full automated run

```bash
chmod +x system_test/*.sh
./system_test/run_system_test.sh
```

This script checks:
1. `/health`
2. MCP ingest
3. A2A ingest
4. Mixed ingest
5. `/stats`
6. `/anomalies`
7. `/anomalies/{log_id}` + `/graph/rca/{log_id}`

## Quick smoke run

```bash
chmod +x system_test/*.sh
./system_test/quick_smoke.sh
```

## Optional: Dashboard check

Open `http://localhost:8501` and validate:
- Overview shows updated counts
- Cognitive Trace can select anomalies
- Knowledge Graph page works with session IDs from logs
- RCA page resolves a known anomaly log_id

## Notes

- These logs are intentionally **unlabeled** (`is_anomaly` and `anomaly_type` omitted) to mimic real-world ingestion.
- The system handles anomaly detection and classification internally.
- If you see `errors > 0` for these files, your running API is likely an older image/schema. Rebuild API and retry:

```bash
cd contextwatch/docker
docker compose up -d --build api
```

## MCP Benchmark Validation (Answer-Key Locked)

### Benchmark files
- `system_test/mcp_benchmark_logs.jsonl`
- `system_test/mcp_benchmark_answer_key.jsonl`

### How to run (engine-level exact match)

```bash
cd contextwatch
PYTHONPATH=. ../.venv/bin/pytest -q tests/test_mcp_benchmark_answer_key.py
```

### Export compact analytics (JSON + CSV)

```bash
cd contextwatch
PYTHONPATH=. ../.venv/bin/python scripts/export_mcp_benchmark_reports.py
```

Generated files:
- `contextwatch/reports/mcp_benchmark_summary.json`
- `contextwatch/reports/mcp_benchmark_session_matrix.csv`
- `contextwatch/reports/mcp_benchmark_log_comparison.csv`

### Latest result (2026-03-10)
- Total logs: `617`
- Expected anomalies: `34`
- Predicted anomalies: `34`
- True positives: `34`
- False positives: `0`
- False negatives: `0`
- Score: `34/34` (**Perfect**)

### Session matrix validation

| Session | Expected | Predicted | Status |
|---|---:|---:|---|
| S01 | 0 | 0 | ✅ |
| S02 | 1 | 1 | ✅ |
| S03 | 2 | 2 | ✅ |
| S04 | 2 | 2 | ✅ |
| S05 | 3 | 3 | ✅ |
| S06 | 1 | 1 | ✅ |
| S07 | 1 | 1 | ✅ |
| S08 | 2 | 2 | ✅ |
| S09 | 1 | 1 | ✅ |
| S10 | 2 | 2 | ✅ |
| S11 | 1 | 1 | ✅ |
| S12 | 10 | 10 | ✅ |
| S13 | 1 | 1 | ✅ |
| S14 | 3 | 3 | ✅ |
| S15 | 0 | 0 | ✅ |
| S16 | 1 | 1 | ✅ |
| S17 | 1 | 1 | ✅ |
| S18 | 1 | 1 | ✅ |
| S19 | 1 | 1 | ✅ |
| S20 | 0 | 0 | ✅ |

### Edge cases explicitly covered
- **High warning threshold for rate limit**: `RATE_LIMIT_WARN` only escalates when near-breach (`used >= 95% of limit`).
- **Rate-limit lifecycle correlation**: warn → 429 → retry flow is preserved without over-flagging normal retries.
- **Semantic file_key abuse**: suspicious values like `system_root` are detected via argument-level semantic checks.
- **Runaway retry loop context rule (C-03)**: repeated call bursts plus recent `503` upstream errors trigger `runaway_retry_loop` early enough.
- **Orphaned retry logic (C-02)**: retries without prior concrete error remain flagged.
- **Latency boundary protection**: `799ms` remains normal when threshold is `800ms` (no off-by-one false positive).

### Naming conventions (clean + readable)
- Subtype naming uses snake_case and behavior-first labels (for example: `unknown_tool_call`, `invalid_tool_parameter`, `runaway_retry_loop`).
- Legacy taxonomy mapping remains stable for compatibility:
  - `TOOL_HALLUCINATION`
  - `CONTEXT_POISONING`
  - `REGISTRY_OVERFLOW`
  - `DELEGATION_CHAIN_FAILURE`
- Session summaries expose clean severity buckets:
  - `error`, `warning`, `info`, `debug`
  - plus `non_anomalous_warnings` and `non_anomalous_errors` for transparent suppression analytics.

### Hybrid rollout analytics (Phase-ready)
- Session summary now includes `policy_stats` (for example: `promoted`, `rule_only`, `ml_review_queue`).
- Finding rows can include:
  - `promotion_source` (`rule_only`, `rule_ml`, `ml_only`, `ml_review_queue`)
  - `policy_decision`
  - `ml_score`, `ml_label`
  - `context_rule_ids` (`C-02`, `C-03`, `C-04`)
- Report exports now include a `promotion_source_breakdown` section in `mcp_benchmark_summary.json` and extra hybrid columns in `mcp_benchmark_log_comparison.csv`.

### Hybrid environment flags
- `MCP_HYBRID_ENABLED` (default `false`)
- `MCP_HYBRID_PHASE` (`1` to `4`)
- `MCP_ML_KILLSWITCH`
- `MCP_ML_SHADOW_MODE`
- `MCP_ML_PROMOTION_THRESHOLD`
- `MCP_ML_MODEL_NAME`
