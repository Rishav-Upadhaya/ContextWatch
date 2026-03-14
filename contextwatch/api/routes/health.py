from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    neo4j_ok = hasattr(request.app.state, "kg") and request.app.state.kg is not None
    chroma_ok = hasattr(request.app.state, "embedder") and request.app.state.embedder is not None
    return {
        "status": "success",
        "data": {"status": "ok", "neo4j": neo4j_ok, "chroma": chroma_ok},
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.get("/metrics")
def metrics(request: Request):
    metrics_state = getattr(request.app.state, "metrics", {})
    requests_total = int(metrics_state.get("requests_total", 0))
    requests_failed = int(metrics_state.get("requests_failed", 0))
    latency_sum_ms = float(metrics_state.get("latency_sum_ms", 0.0))
    latency_count = int(metrics_state.get("latency_count", 0))
    avg_latency_ms = (latency_sum_ms / latency_count) if latency_count else 0.0

    return {
        "status": "success",
        "data": {
            "requests_total": requests_total,
            "requests_failed": requests_failed,
            "error_rate": (requests_failed / requests_total) if requests_total else 0.0,
            "avg_latency_ms": round(avg_latency_ms, 2),
            "started_at": metrics_state.get("started_at"),
        },
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }
