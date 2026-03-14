from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/session/{session_id}")
def session_graph(session_id: str, request: Request):
    density = request.app.state.kg.session_anomaly_density(session_id)
    logs = [
        item.normalized.model_dump(mode="json")
        for item in request.app.state.store.all_logs.values()
        if item.normalized.session_id == session_id
    ]
    return {
        "status": "success",
        "data": {"session_id": session_id, "logs": logs, "density": density},
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.get("/rca/{log_id}")
def rca(log_id: str, request: Request):
    result = request.app.state.kg.trace_rca(log_id)
    return {
        "status": "success",
        "data": result.model_dump(mode="json"),
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.get("/trigger/{log_id}")
def trigger_graph(log_id: str, request: Request):
    data = request.app.state.kg.trigger_graph(log_id)
    return {
        "status": "success",
        "data": data,
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }
