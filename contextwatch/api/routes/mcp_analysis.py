from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from api.security import require_api_key
from core.mcp_signal_engine import MCPSignalEngine

router = APIRouter(prefix="/analyze", tags=["analyze", "mcp"])


@router.post("/mcp/session")
def analyze_mcp_session(payload: dict | list = Body(...), request: Request = None, _auth: None = Depends(require_api_key)):
    if request is None:
        raise HTTPException(status_code=500, detail="Request context unavailable")
    if isinstance(payload, list):
        raw_logs = payload
    elif isinstance(payload, dict) and isinstance(payload.get("logs"), list):
        raw_logs = payload.get("logs") or []
    else:
        raise HTTPException(status_code=422, detail="Payload must be a JSON array of MCP logs or object with logs[].")

    normalizer = request.app.state.normalizer
    engine = MCPSignalEngine(
        settings=request.app.state.settings,
        ml_assist=getattr(request.app.state, "mcp_ml_assist", None),
    )

    normalized_logs = []
    errors = []

    for idx, raw in enumerate(raw_logs, 1):
        try:
            normalized = normalizer.normalize(raw)
            if normalized.protocol != "MCP":
                errors.append({"index": idx, "error": "Only MCP logs are supported by this endpoint"})
                continue
            normalized_logs.append(normalized)
        except Exception as exc:
            errors.append({"index": idx, "error": str(exc)[:280]})

    result = engine.analyze(normalized_logs)

    return {
        "status": "success",
        "data": result.model_dump(mode="json"),
        "meta": {
            "processing_time_ms": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        },
    }
