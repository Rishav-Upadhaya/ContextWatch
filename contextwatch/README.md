# ContextWatch

ContextWatch is an AI-powered anomaly detection and observability system for MCP and A2A logs.

## How the System Works (Core Fundamentals)

ContextWatch is designed for **AI-agent observability**, where the goal is not only to detect failures, but to explain **why** an agent decision path became risky. The system treats each log as a semantic event and processes it through a multi-stage pipeline.

### Core Design Principles

1. **Protocol-aware ingestion, unified analysis**
	- Input logs can be MCP or A2A and are normalized into one internal schema.
	- This keeps downstream algorithms protocol-agnostic while preserving protocol-specific metadata.

2. **Semantic detection over keyword matching**
	- Instead of rigid regex-only checks, the system embeds log meaning and compares it to normal behavior in vector space.

3. **Two-stage classification for reliability + flexibility**
	- Deterministic rule checks handle common, high-confidence cases.
	- LLM fallback handles ambiguous edge cases where semantic interpretation is needed.

4. **Graph-based RCA instead of flat event lists**
	- Events and anomalies are represented in Neo4j as relationships.
	- Root-cause traversal is performed with bounded graph depth.

5. **Explainability-first output**
	- API and dashboard expose anomaly score, class, confidence, causal chain, and optional LLM explanation.

## End-to-End Pipeline

The system executes this sequence for each ingested log:

1. **Ingest** (`/ingest/log` or `/ingest/file`)
2. **Normalize** raw MCP/A2A structure into `NormalizedLog`
3. **Embed** `text_for_embedding` using MiniLM
4. **Detect** anomaly by nearest-neighbor cosine distance to normal baseline
5. **Classify** anomaly type (rule-first, LLM fallback)
6. **Persist graph event** + anomaly nodes/edges in Neo4j
7. **Run RCA** (up to 3-hop causal lookup)
8. **Return enriched response** for API and dashboard

---

## Algorithms and Methods

### 1) Normalization Algorithm

**Goal:** Convert MCP and A2A logs into a unified representation.

- MCP input is validated and transformed into:
  - `log_id`, `session_id`, `protocol`, `agent_id`, `timestamp`
  - `text_for_embedding = reasoning_step | intent | tool_name | context_summary`
- A2A input is validated and transformed into:
  - `log_id`, `session_id`, `protocol`, `agent_id=source_agent`, `timestamp`
  - `text_for_embedding = message_content | task_intent | target_agent`

**Why this matters:** It creates a single semantic signal while retaining full raw metadata for RCA and dashboard views.

### 2) Embedding + Vector Indexing

**Model:** `all-MiniLM-L6-v2` (384 dimensions)

- Text is encoded into dense vectors with normalized embeddings.
- Vectors are stored in ChromaDB (cosine space, HNSW-backed collection).
- Metadata is sanitized and persisted for traceability.

**Complexity intuition:** Approximate nearest-neighbor retrieval with HNSW provides scalable similarity search without brute-force full scans.

### 3) Anomaly Detection Algorithm

**Method:** Distance-threshold nearest-normal detection

For incoming vector $x$ and normal reference vectors $N$:

$$
score(x) = \min_{n \in N} d_{cos}(x, n)
$$

- If `score > ANOMALY_THRESHOLD`, event is anomalous.
- Current calibrated threshold: `0.20` (from latest golden dataset sweep).
- Confidence is mapped via sigmoid around the threshold:

$$
confidence = \sigma\left((score - \tau) \cdot 10\right)
$$

where $\tau$ is the threshold.

### 4) Error Taxonomy Classification

**Taxonomy classes:**
- `TOOL_HALLUCINATION`
- `CONTEXT_POISONING`
- `REGISTRY_OVERFLOW`
- `DELEGATION_CHAIN_FAILURE`

**Two-stage strategy:**

1. **Rule-based classifier**
	- Unknown/invalid tools/params detection
	- Context overload and repetitive-context heuristics
	- Delegation depth/failure pattern checks
	- Intent-domain vs tool-domain mismatch checks

2. **LLM fallback classifier** (if needed)
	- Uses anomaly log + recent context
	- Returns class + confidence + rationale

### 5) Knowledge Graph + RCA

**Graph backend:** Neo4j

Stored entities include events, sessions, agents, tools, reasoning nodes, and anomaly nodes.

RCA query pattern uses bounded causal expansion:

$$
(Anomaly)-[:CAUSED\_BY*1..3]-(root)
$$

- Select shortest/closest causal chain.
- If no chain exists, anomaly is treated as root with hop count `0`.

### 6) LLM Explanation Layer

- Triggered only for sufficiently confident anomalies.
- Prompt includes:
  - anomaly type,
  - anomalous log payload,
  - up to 3 preceding context logs,
  - causal chain summary.
- Supports provider selection (Anthropic/OpenAI) with fallback text when no API key is set.

---

## Data Strategy: Synthetic vs Golden vs Real-World

- **Synthetic datasets**: high-volume generated data for pipeline stress and baseline formation.
- **Golden dataset**: curated labeled subset for threshold tuning and benchmark reporting.
- **Real-world logs**: can be unlabeled (no `is_anomaly` / `anomaly_type`); system infers these during detection/classification.

## Why This Architecture Works

- Handles both MCP and A2A without separate codepaths downstream.
- Gives fast inference (<500ms p95 target) through compact embeddings + ANN search.
- Balances deterministic controls (rules) with semantic generalization (LLM fallback).
- Produces operator-friendly outputs through graph RCA and dashboard traceability.

## Current Limitations (Important)

- Quality depends on the representativeness of the normal baseline vectors.
- Thresholds and class balance may need retuning per domain/team.
- RCA depth is bounded (3 hops) by design; long causal cascades may need iterative traversal.
- LLM explanations are best-effort and should be treated as assistive, not absolute truth.

## Architecture
- Synthetic log generation (`scripts/`)
- Core pipeline: normalize → embed (MiniLM) → detect → classify → RCA graph (`core/`)
- FastAPI service (`api/`)
- Streamlit dashboard (`dashboard/`)
- Evaluation suite (`evaluation/`)

## Setup
1. `cd contextwatch`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cp config/.env.example .env`

## Data Generation Order
1. `python scripts/generate_mcp_logs.py`
2. `python scripts/generate_a2a_logs.py`
3. `python scripts/generate_mixed_logs.py`
4. `python scripts/build_golden_dataset.py`
5. `python scripts/validate_logs.py data/synthetic/mcp/mcp_normal_logs.jsonl data/synthetic/mcp/mcp_anomaly_logs.jsonl data/synthetic/a2a/a2a_normal_logs.jsonl data/synthetic/a2a/a2a_anomaly_logs.jsonl data/synthetic/mixed/mcp_a2a_mixed_logs.jsonl --samples 10`

## Run API + Dashboard
- API: `uvicorn api.main:app --reload --port 8000`
- Dashboard: `streamlit run dashboard/app.py --server.port 8501`

## MCP Input Format (Real-World)

For MCP anomaly checks, send one `notifications/message` event per request to `/ingest/log`.

Example payload:

```json
{
	"log_id": "11111111-1111-4111-8111-111111111111",
	"protocol": "MCP",
	"session": {
		"id": "mcp-session-f9a2c841",
		"host": "Claude Desktop",
		"server": "figma-mcp-server v1.4.2",
		"connected_at": "2026-03-10T09:14:02.311Z",
		"transport": "stdio"
	},
	"jsonrpc": "2.0",
	"method": "notifications/message",
	"params": {
		"level": "info",
		"logger": "figma-mcp-server",
		"data": {
			"timestamp": "2026-03-10T09:14:05.102Z",
			"event": "TOOL_CALL_RECEIVED",
			"message": "Tool invocation received: get_file",
			"meta": {
				"tool": "get_file",
				"arguments": {"file_key": "hU2yV3kPzXqN8mD1oL5rT9"},
				"request_id": 1,
				"triggered_by": "user_prompt: 'Can you review my dashboard design and suggest improvements?'"
			}
		}
	}
}
```

Notes:
- `is_anomaly` and `anomaly_type` are optional for real unlabeled streams.
- Envelope shape with `session` + `logs` array is also supported; the latest log event is normalized.

## Docker
- `cd docker`
- `docker compose --env-file ../.env down -v`
- `docker compose --env-file ../.env up -d --build`
- `bash ../scripts/docker_validate.sh` (or `bash ../scripts/docker_validate.sh --fresh`)

## Evaluation
- `python evaluation/eval_detector.py`
- `python evaluation/eval_classifier.py`
- `python evaluation/eval_rca.py`
- `python evaluation/eval_latency.py`

Calibrated anomaly threshold from latest upgraded-data sweep: `ANOMALY_THRESHOLD=0.20`.

Optional protocol-specific override (recommended for mixed real-world streams):
- `ANOMALY_THRESHOLD_MCP=0.20` (latest calibrated)
- `ANOMALY_THRESHOLD_A2A=0.20` (latest calibrated)

When these are set, detector uses protocol-aware thresholding; otherwise it falls back to `ANOMALY_THRESHOLD`.

## Benchmark Table
| Metric | Target | Result |
|---|---:|---:|
| Detection F1 | >= 0.80 | 0.9455 |
| Classifier Accuracy | >= 0.75 | 0.8810 |
| RCA Precision@3 | >= 0.75 | 1.0000 |
| Latency p95 | < 500ms | 221.09ms |
