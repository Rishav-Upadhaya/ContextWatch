import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.security import require_api_key
from api.store import ProcessedLog
from config.settings import get_settings

logger = logging.getLogger("contextwatch.ingest")
limiter = Limiter(key_func=get_remote_address)
settings = get_settings()
LOG_RATE_LIMIT = f"{settings.RATE_LIMIT_PER_MINUTE}/minute"
FILE_RATE_LIMIT = f"{settings.RATE_LIMIT_FILE_PER_MINUTE}/minute"

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _run_pipeline(request: Request, raw: dict, context_logs: list = None) -> ProcessedLog:
    normalizer = request.app.state.normalizer
    detector = request.app.state.detector
    classifier = request.app.state.classifier
    kg = request.app.state.kg
    explainer = request.app.state.explainer

    normalized = normalizer.normalize(raw)
    previous = request.app.state.store.latest_in_session(normalized.session_id, exclude_log_id=normalized.log_id)
    anomaly = detector.detect(normalized)
    classification = classifier.classify(normalized, anomaly, context_logs or [])
    anomaly.anomaly_type = classification.anomaly_type
    kg.upsert_event(normalized.model_dump(mode="json"), anomaly)
    if previous is not None:
        lag_ms = int(max((normalized.timestamp - previous.normalized.timestamp).total_seconds() * 1000, 0))
        kg.create_temporal_link(normalized.log_id, previous.normalized.log_id, lag_ms)

        if normalized.protocol == "A2A":
            try:
                src = str(normalized.metadata.get("source_agent", "")).strip()
                tgt = str(normalized.metadata.get("target_agent", "")).strip()
                if src and tgt and src != tgt:
                    kg.create_delegation_link(previous.normalized.log_id, normalized.log_id)
            except Exception:
                pass

        if anomaly.is_anomaly:
            kg.create_causal_link(normalized.log_id, previous.normalized.log_id, relation="SEQUENTIAL_CAUSAL_HINT")

    rca = kg.trace_rca(normalized.log_id) if anomaly.is_anomaly else None
    explanation = (
        explainer.explain_anomaly(normalized, classification, context_logs or [], rca, anomaly.anomaly_score, anomaly.confidence)
        if anomaly.is_anomaly and rca is not None
        else None
    )
    return ProcessedLog(
        normalized=normalized,
        anomaly=anomaly,
        classification=classification,
        explanation=explanation,
        rca=rca,
    )


@router.post("/log")
@limiter.limit(LOG_RATE_LIMIT)
def ingest_log(raw: dict, request: Request, _auth: None = Depends(require_api_key)):
    start = time.perf_counter()
    logger.info(f"Ingesting single log: {raw.get('log_id', 'unknown')}")
    try:
        processed = _run_pipeline(request, raw)
    except Exception as exc:
        logger.error(f"Failed to process log: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request.app.state.store.upsert(processed)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"Log processed in {elapsed_ms:.2f}ms, anomaly={processed.anomaly.is_anomaly}")
    return {
        "status": "success",
        "data": {
            "log_id": processed.normalized.log_id,
            "is_anomaly": processed.anomaly.is_anomaly,
            "anomaly_type": processed.classification.anomaly_type,
            "anomaly_score": processed.anomaly.anomaly_score,
            "confidence": processed.classification.confidence,
            "rule_lane_triggered": processed.anomaly.rule_lane_triggered,
            "embedding_lane_score": processed.anomaly.embedding_lane_score,
            "embedding_lane_threshold": processed.anomaly.embedding_lane_threshold,
            "arbitration_mode": processed.anomaly.arbitration_mode,
        },
        "meta": {"processing_time_ms": round(elapsed_ms, 2), "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.post("/file")
@limiter.limit(FILE_RATE_LIMIT)
async def ingest_file(request: Request, file: UploadFile = File(...), _auth: None = Depends(require_api_key)):
    start = time.perf_counter()
    logger.info(f"Ingesting file: {file.filename}")
    content = (await file.read()).decode("utf-8")
    lines = [line for line in content.splitlines() if line.strip()]
    logger.info(f"Processing {len(lines)} logs from file")
    inserted = 0
    errors = 0
    for line in lines:
        try:
            raw = json.loads(line)
            processed = _run_pipeline(request, raw)
            request.app.state.store.upsert(processed)
            inserted += 1
        except Exception as e:
            logger.debug(f"Error processing line: {e}")
            errors += 1
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"File processed: {inserted} inserted, {errors} errors in {elapsed_ms:.2f}ms")
    return {
        "status": "success",
        "data": {"inserted": inserted, "errors": errors, "total": len(lines)},
        "meta": {"processing_time_ms": round(elapsed_ms, 2), "timestamp": datetime.now(timezone.utc).isoformat()},
    }
