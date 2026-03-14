from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware import timing_middleware
from api.routes.health import router as health_router


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.state.metrics = {
        "requests_total": 0,
        "requests_failed": 0,
        "latency_sum_ms": 0.0,
        "latency_count": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
    }

    @app.get("/ok")
    def ok():
        return {"status": "ok"}

    app.middleware("http")(timing_middleware)
    app.include_router(health_router)
    return app


def test_metrics_endpoint_reports_aggregates():
    app = _build_test_app()
    with TestClient(app) as client:
        client.get("/ok")
        client.get("/missing")
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["requests_total"] == 2
    assert body["data"]["requests_failed"] == 1
    assert body["data"]["error_rate"] == 0.5
    assert body["data"]["avg_latency_ms"] >= 0.0
    assert body["data"]["started_at"] == "2026-01-01T00:00:00+00:00"


def test_metrics_endpoint_works_without_metrics_state():
    app = FastAPI()
    app.include_router(health_router)
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["requests_total"] == 0
    assert data["requests_failed"] == 0
    assert data["error_rate"] == 0.0
    assert data["avg_latency_ms"] == 0.0
    assert data["started_at"] is None