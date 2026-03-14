# ContextWatch

ContextWatch is an observability and anomaly-detection system for AI-agent interaction logs, with first-class support for MCP and A2A protocols.

## Topic

AI-agent observability, anomaly detection, and root-cause analysis for protocol-driven agent systems.

## Description

ContextWatch ingests MCP/A2A logs, normalizes them into a unified schema, embeds semantic content, detects anomalies against a normal-behavior baseline, classifies anomaly type, and provides graph-based root-cause analysis (RCA). It includes a FastAPI backend, Streamlit dashboard, and evaluation pipelines.

## Problem Statement

Modern agent systems generate complex interaction traces. Traditional log monitoring approaches (keyword-only, static rules) are weak at:

- Identifying semantic failures in tool usage and delegation behavior.
- Reducing false positives across heterogeneous log formats.
- Explaining why an anomaly happened and what preceded it.
- Supporting both online monitoring and offline quality benchmarking.

## Solution

ContextWatch addresses this with a hybrid pipeline:

1. Normalize MCP/A2A events into a common `NormalizedLog` format.
2. Build semantic embeddings (`all-MiniLM-L6-v2`, 384-dim).
3. Compute nearest-normal cosine distance using ChromaDB baseline vectors.
4. Detect anomalies via thresholding.
5. Classify anomaly taxonomy with rule-first logic and optional LLM fallback.
6. Persist causal context for graph traversal and RCA.
7. Expose analysis through API endpoints and dashboard pages.

## Architecture

### End-to-End Flow

1. Ingestion (`api/routes/`)
2. Normalization (`core/normalizer.py`)
3. Embedding (`core/embedder.py`)
4. Detection (`core/detector.py`)
5. Classification (`core/classifier.py`)
6. Knowledge Graph + RCA (`core/knowledge_graph.py`)
7. Explanation (`core/llm_explainer.py`)
8. Visualization (`dashboard/`)

### Core Components

- `contextwatch/api/`: FastAPI endpoints, middleware, security, storage access.
- `contextwatch/core/`: Detection engine, classifiers, schema, graph, and signal logic.
- `contextwatch/dashboard/`: Streamlit app for operations and RCA exploration.
- `contextwatch/config/`: Runtime settings and logging configuration.
- `contextwatch/scripts/`: Data generation and validation utilities.
- `contextwatch/evaluation/`: Offline metric and performance evaluation scripts.
- `contextwatch/tests/`: Unit/regression/API test coverage.

## Folder Structure

The structure below highlights source and documentation folders (excluding local/ignored artifacts such as virtual environments, caches, and generated vector DB directories):

```text
ContextWatch/
├── README.md
├── contextwatch/
│   ├── README.md
│   ├── requirements.txt
│   ├── api/
│   ├── config/
│   ├── core/
│   ├── dashboard/
│   ├── data/
│   ├── docker/
│   ├── evaluation/
│   ├── reports/
│   ├── scripts/
│   └── tests/
├── data/
└── system_test/
```

## Tech Stack

- Python, FastAPI, Streamlit
- SentenceTransformers (`all-MiniLM-L6-v2`)
- ChromaDB (vector similarity search)
- Neo4j (knowledge graph + RCA)
- Pytest (testing)
- Docker / Docker Compose (deployment)

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`
- (Optional) Docker + Docker Compose
- (Optional) Neo4j instance for graph-backed RCA workflows

### Local Setup

```bash
cd contextwatch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Services

```bash
# API
uvicorn api.main:app --reload --port 8000

# Dashboard (new terminal)
streamlit run dashboard/app.py --server.port 8501
```

### Run Tests

```bash
cd contextwatch
pytest -q
```

## Evaluation

```bash
cd contextwatch
python evaluation/eval_detector.py
python evaluation/eval_classifier.py
python evaluation/eval_rca.py
python evaluation/eval_latency.py
```

## Notes

- The embedding-based detector is a key signal source for anomaly scoring.
- Real-world unlabeled logs are supported; labels are optional and can be inferred.
- For deployment guidance, see `PRODUCTION.md`.