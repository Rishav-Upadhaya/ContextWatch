"""FastAPI interface — ContextWatch HTTP API.

Endpoints:
  GET  /health
  POST /ingest/log         – single log through full pipeline
  POST /ingest/batch       – batch of logs
  POST /finetune           – finetune LogBERT on provided logs (MLKP + VHM)
  GET  /anomalies          – paginated anomaly list from PostgreSQL
  GET  /normals            – paginated normal-log list from PostgreSQL
  GET  /rca/{anomaly_id}   – root cause analysis for an anomaly
  GET  /graph/snapshot     – current knowledge graph
  GET  /model/info         – LogBERT model info
"""

from __future__ import annotations

import uuid
import time
import asyncio
from functools import partial
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import logging

logger = logging.getLogger(__name__)

from contextwatch.services.detection.anomaly_service import AnomalyService
from contextwatch.services.pipeline.pipeline import build_default_pipeline

from contextwatch.models.schemas import (
    LogIngestRequest,
    LogIngestResponse,
    BatchLogIngestRequest,
    BatchIngestResponse,
    AnomalyQueryResponse,
    NormalLogQueryResponse,
    RCAResponse,
    FinetuneRequest,
    FinetuneResponse,
    ModelInfoResponse
)

# ── Application wiring ────────────────────────────────────────────────────────

_cache: dict[str, tuple[Any, float]] = {}
TTL = 30  # seconds

def get_cached(key: str) -> Any | None:
    if key in _cache:
        value, ts = _cache[key]
        if time.time() - ts < TTL:
            return value
    return None

def set_cached(key: str, value: Any) -> None:
    _cache[key] = (value, time.time())

def create_app() -> FastAPI:
    pipeline = build_default_pipeline()
    anomaly_service = AnomalyService(baseline_embeddings=[])

    app = FastAPI(
        title="ContextWatch API",
        version="3.0.0",
        description=(
            "Anomaly detection for MCP/A2A logs: "
            "batch ingestion → normalise → LogBERT embed → "
            "vector distance → VHM anomaly check → RCA → PostgreSQL"
        ),
    )

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        cached = get_cached("health")
        if cached: return cached
        
        from datetime import datetime, timezone
        res = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vhm_fitted": pipeline.vhm.is_fitted,
            "vhm_radius": float(pipeline.vhm.radius),
        }
        set_cached("health", res)
        return res

    # ── Single log ingestion ──────────────────────────────────────────────────
    @app.post("/ingest/log", response_model=LogIngestResponse)
    async def ingest_log(request: LogIngestRequest, background_tasks: BackgroundTasks):
        """
        Full pipeline for one log:
          normalize → tokenize → LogBERT embed → cosine/VHM distance
          → anomaly check → [RCA if anomaly] → store in PostgreSQL (background)
        """
        try:
            raw_log = request.dict(exclude_none=True)
            raw_log["log_id"] = raw_log.get("log_id") or str(uuid.uuid4())
            
            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(
                None,
                partial(
                    pipeline.process,
                    raw_log,
                    protocol=request.protocol,
                    enqueue_task=background_tasks.add_task
                )
            )
            
            return LogIngestResponse(
                success=True,
                log_id=outcome.log_id,
                is_anomaly=outcome.is_anomaly,
                anomaly_score=outcome.anomaly_score,
                anomaly_type=outcome.anomaly_type,
                confidence=outcome.confidence,
                cosine_distance=outcome.cosine_distance,
                explanation=outcome.explanation,
                token_efficiency=outcome.token_efficiency,
                trace_context=outcome.trace_context,
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            logger.error(f"Error ingesting log: {exc}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(f"Unexpected error ingesting log: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    # ── Batch log ingestion ───────────────────────────────────────────────────
    @app.post("/ingest/batch", response_model=BatchIngestResponse)
    async def ingest_batch(request: BatchLogIngestRequest, background_tasks: BackgroundTasks):
        try:
            raw_logs = [log.dict(exclude_none=True) for log in request.logs]
            loop = asyncio.get_running_loop()
            batch = await loop.run_in_executor(
                None,
                partial(
                    pipeline.process_batch,
                    raw_logs,
                    include_rca=request.include_rca,
                    enqueue_task=background_tasks.add_task
                )
            )
            return BatchIngestResponse(
                batch_size=batch.batch_size,
                anomaly_count=batch.anomaly_count,
                normal_count=batch.normal_count,
                results=[
                    LogIngestResponse(
                        success=True,
                        log_id=r.log_id,
                        is_anomaly=r.is_anomaly,
                        anomaly_score=r.anomaly_score,
                        anomaly_type=r.anomaly_type,
                        confidence=r.confidence,
                        cosine_distance=r.cosine_distance,
                        explanation=r.explanation,
                        token_efficiency=r.token_efficiency,
                        trace_context=r.trace_context,
                    )
                    for r in batch.results
                ],
                graph=pipeline.kg.serialize(),
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            logger.error(f"Error ingesting batch: {exc}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(f"Unexpected error ingesting batch: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    # ── Finetune LogBERT ──────────────────────────────────────────────────────
    @app.post("/finetune", response_model=FinetuneResponse)
    async def finetune(request: FinetuneRequest):
        try:
            if not request.logs and not request.use_system_data:
                raise ValueError("No logs provided for finetuning")

            loop = asyncio.get_running_loop()
            metrics = await loop.run_in_executor(
                None,
                partial(
                    pipeline.finetune,
                    raw_logs=request.logs,
                    epochs=request.epochs,
                    learning_rate=request.learning_rate,
                    use_system_data=request.use_system_data,
                )
            )
            _cache.pop("model_info", None)

            return FinetuneResponse(
                success=True,
                epochs_completed=len(metrics),
                n_sequences=len(request.logs) if not request.use_system_data else int(pipeline.calibration_metrics.get("training_normals", 0)) + int(pipeline.calibration_metrics.get("training_anomalies", 0)),
                metrics=metrics,
                vhm_radius_after=float(pipeline.vhm.radius),
                decision_radius_after=float(pipeline.vhm.decision_radius),
                calibration=pipeline.calibration_metrics,
            )
        except (ValueError, RuntimeError) as exc:
            logger.error(f"Error during finetuning: {exc}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(f"Unexpected error during finetuning: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    # ── Query anomalies ───────────────────────────────────────────────────────
    @app.get("/anomalies", response_model=AnomalyQueryResponse)
    async def list_anomalies(
        page: int = Query(1, ge=1),
        limit: int = Query(20, ge=1, le=5000),
        anomaly_type: Optional[str] = Query(None),
    ):
        # Database fetches are IO-bound but fast, still run in executor for pure async
        loop = asyncio.get_running_loop()
        anomalies, total = await loop.run_in_executor(
            None,
            partial(pipeline.normal_store.fetch_anomalies, page=page, limit=limit, anomaly_type=anomaly_type)
        )
        if total == 0:
            nodes = pipeline.kg.query_by_type("Anomaly")
            in_mem = []
            for node in nodes:
                ntype = node.properties.get("anomaly_type") or "UNKNOWN"
                if anomaly_type and ntype != anomaly_type:
                    continue
                in_mem.append({
                    "anomaly_id": node.id,
                    "log_id": node.properties.get("log_id"),
                    "anomaly_type": ntype,
                    "score": node.properties.get("score"),
                    "confidence": node.properties.get("confidence"),
                })
            start = (page - 1) * limit
            anomalies = in_mem[start:start + limit]
            total = len(in_mem)

        return {"anomalies": anomalies, "total": total, "page": page, "limit": limit}

    # ── Query normal logs ─────────────────────────────────────────────────────
    @app.get("/normals", response_model=NormalLogQueryResponse)
    async def list_normal_logs(
        page: int = Query(1, ge=1),
        limit: int = Query(20, ge=1, le=5000),
    ):
        loop = asyncio.get_running_loop()
        logs, total = await loop.run_in_executor(
            None,
            partial(pipeline.normal_store.fetch_normal_logs, page=page, limit=limit)
        )
        return {"logs": logs, "total": total, "page": page, "limit": limit}

    # ── RCA ───────────────────────────────────────────────────────────────────
    @app.get("/rca/{anomaly_id}", response_model=RCAResponse)
    async def get_rca(anomaly_id: str):
        if pipeline.kg.get_node(anomaly_id) is None:
            raise HTTPException(status_code=404, detail="Anomaly not found in knowledge graph")
        
        loop = asyncio.get_running_loop()
        trace = await loop.run_in_executor(
            None,
            partial(pipeline.marcra.analyze, anomaly_node_id=anomaly_id, context_logs=[])
        )
        chain = pipeline.kg.find_related_nodes(anomaly_id, max_hops=3)
        evidence = [f"Node {n.id} (type={n.type})" for n in chain]
        return RCAResponse(
            anomaly_id=anomaly_id,
            causal_chain=[{"id": n.id, "type": n.type, "properties": n.properties} for n in chain],
            evidence=evidence,
            confidence=trace.confidence if trace else 0.0,
        )

    # ── Graph snapshot ────────────────────────────────────────────────────────
    @app.get("/graph/snapshot")
    async def get_graph_snapshot():
        loop = asyncio.get_running_loop()
        snap = await loop.run_in_executor(None, pipeline.normal_store.fetch_graph_snapshot)
        if snap.get("nodes") or snap.get("edges"):
            return snap
        return pipeline.kg.serialize()

    # ── Model info ────────────────────────────────────────────────────────────
    @app.get("/model/info", response_model=ModelInfoResponse)
    async def get_model_info():
        cached = get_cached("model_info")
        if cached: return cached

        cfg = pipeline.config
        res = ModelInfoResponse(
            model_type="LogBERT (Transformer, from scratch, numpy only)",
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            max_seq_len=cfg.max_seq_len,
            vocab_size=cfg.vocab_size,
            vhm_fitted=pipeline.vhm.is_fitted,
            vhm_radius=float(pipeline.vhm.radius),
            decision_radius=float(pipeline.vhm.decision_radius),
            calibration=pipeline.calibration_metrics or None,
        )
        set_cached("model_info", res)
        return res

    # ── Error Explanation / Agentic Reasoning ─────────────────────────────────
    @app.post("/reasoning/explain_error")
    async def explain_error(request):
        """
        Agentic error explanation using Multi-node LangGraph workflow.
        
        Workflow:
          1. Context Gathering: Fetch error details + similar past errors
          2. Signal Extraction: Extract domain signals from error patterns
          3. Reasoning: Claude-powered RCA analysis
          4. Remediation: Generate actionable fix suggestions
          
        Returns structured explanation with root cause, remediation steps, and confidence.
        """
        from contextwatch.models.schemas import ErrorExplanationRequest, ErrorExplanationResponse
        from contextwatch.services.reasoning.error_explanation_agent import ErrorExplanationAgent
        
        try:
            # Parse request (use dict() if Pydantic model coming in)
            if hasattr(request, "dict"):
                req_dict = request.dict()
            else:
                req_dict = request
                
            anomaly_id = req_dict.get("anomaly_id", "unknown")
            error_type = req_dict.get("error_type", "UNKNOWN")
            raw_log = req_dict.get("raw_log", {})
            
            # Initialize reasoning agent
            agent = ErrorExplanationAgent()
            
            # Run reasoning workflow
            explanation = agent.explain_error(
                anomaly_id=anomaly_id,
                error_type=error_type,
                raw_log=raw_log,
            )
            
            return ErrorExplanationResponse(**explanation)
            
        except Exception as e:
            logger.error(f"Error explanation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error explanation workflow failed: {str(e)}"
            )

    return app

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
