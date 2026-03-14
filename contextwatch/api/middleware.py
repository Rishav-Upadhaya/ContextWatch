from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse


async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics = getattr(request.app.state, "metrics", None)
        if isinstance(metrics, dict):
            metrics["requests_total"] = metrics.get("requests_total", 0) + 1
            metrics["requests_failed"] = metrics.get("requests_failed", 0) + 1
            metrics["latency_sum_ms"] = metrics.get("latency_sum_ms", 0.0) + elapsed_ms
            metrics["latency_count"] = metrics.get("latency_count", 0) + 1
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    metrics = getattr(request.app.state, "metrics", None)
    if isinstance(metrics, dict):
        metrics["requests_total"] = metrics.get("requests_total", 0) + 1
        if status_code >= 400:
            metrics["requests_failed"] = metrics.get("requests_failed", 0) + 1
        metrics["latency_sum_ms"] = metrics.get("latency_sum_ms", 0.0) + elapsed_ms
        metrics["latency_count"] = metrics.get("latency_count", 0) + 1

    response.headers["X-Processing-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


def error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "data": {"message": message},
            "meta": {
                "processing_time_ms": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        },
    )
