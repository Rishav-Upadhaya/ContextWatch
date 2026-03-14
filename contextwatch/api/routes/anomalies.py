from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from core.intent_outcome import compute_intent_outcome_gap

router = APIRouter(tags=["anomalies"])


@router.get("/anomalies")
def list_anomalies(
    request: Request,
    anomaly_type: str | None = Query(default=None),
    agent: str | None = Query(default=None),
):
    items = request.app.state.store.anomaly_list()
    if anomaly_type:
        items = [x for x in items if x.classification.anomaly_type == anomaly_type]
    if agent:
        items = [x for x in items if x.normalized.agent_id == agent]
    data = [
        {
            "log_id": x.normalized.log_id,
            "timestamp": x.normalized.timestamp.isoformat(),
            "agent_id": x.normalized.agent_id,
            "protocol": x.normalized.protocol,
            "anomaly_type": x.classification.anomaly_type,
            "anomaly_score": x.anomaly.anomaly_score,
            "confidence": x.classification.confidence,
            "rule_lane_triggered": x.anomaly.rule_lane_triggered,
            "embedding_lane_score": x.anomaly.embedding_lane_score,
            "embedding_lane_threshold": x.anomaly.embedding_lane_threshold,
            "arbitration_mode": x.anomaly.arbitration_mode,
        }
        for x in items
    ]
    return {
        "status": "success",
        "data": data,
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.get("/anomalies/{log_id}")
def get_anomaly_detail(log_id: str, request: Request):
    item = request.app.state.store.anomalies.get(log_id)
    if not item:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    return {
        "status": "success",
        "data": {
            "log": item.normalized.model_dump(mode="json"),
            "anomaly": item.anomaly.model_dump(mode="json"),
            "classification": item.classification.model_dump(mode="json"),
            "explanation": item.explanation,
            "rca": item.rca.model_dump(mode="json") if item.rca else None,
        },
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.get("/stats")
def stats(request: Request):
    total = len(request.app.state.store.all_logs)
    anomalies = len(request.app.state.store.anomalies)
    dist = Counter(x.classification.anomaly_type for x in request.app.state.store.anomalies.values())
    return {
        "status": "success",
        "data": {
            "total_logs": total,
            "anomaly_count": anomalies,
            "anomaly_rate": (anomalies / total) if total else 0.0,
            "type_distribution": dict(dist),
        },
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.get("/stats/overview")
def stats_overview(request: Request):
    store = request.app.state.store
    all_items = list(store.all_logs.values())
    anomaly_items = list(store.anomalies.values())

    total_logs = len(all_items)
    total_anomalies = len(anomaly_items)

    latest_data = []
    for item in sorted(all_items, key=lambda x: x.normalized.timestamp, reverse=True)[:10]:
        latest_data.append(
            {
                "log_id": item.normalized.log_id,
                "timestamp": item.normalized.timestamp.isoformat(),
                "session_id": item.normalized.session_id,
                "protocol": item.normalized.protocol,
                "agent_id": item.normalized.agent_id,
                "is_anomaly": item.anomaly.is_anomaly,
                "anomaly_type": item.classification.anomaly_type if item.anomaly.is_anomaly else None,
                "confidence": item.classification.confidence if item.anomaly.is_anomaly else None,
                "rule_lane_triggered": item.anomaly.rule_lane_triggered,
                "embedding_lane_score": item.anomaly.embedding_lane_score,
                "embedding_lane_threshold": item.anomaly.embedding_lane_threshold,
                "arbitration_mode": item.anomaly.arbitration_mode,
            }
        )

    anomaly_counts = Counter(item.classification.anomaly_type for item in anomaly_items)
    recurring_anomalies = [
        {"anomaly_type": anomaly_type, "count": count}
        for anomaly_type, count in anomaly_counts.most_common()
        if count >= 2
    ]

    return {
        "status": "success",
        "data": {
            "total_logs": total_logs,
            "total_anomalies": total_anomalies,
            "latest_data": latest_data,
            "recurring_anomalies": recurring_anomalies,
        },
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


@router.get("/stats/cognitive")
def cognitive_stats(request: Request):
    rows = list(request.app.state.store.all_logs.values())
    if not rows:
        data = {
            "total_logs": 0,
            "avg_intent_outcome_gap": 0.0,
            "avg_thought_action_coherence": 0.0,
            "high_gap_count": 0,
        }
        return {
            "status": "success",
            "data": data,
            "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
        }

    scored = [compute_intent_outcome_gap(item.normalized.metadata) for item in rows]
    total = len(scored)
    avg_gap = sum(x.gap_score for x in scored) / total
    avg_coherence = sum(x.coherence_score for x in scored) / total
    high_gap_count = sum(1 for x in scored if x.gap_score >= 0.70)

    return {
        "status": "success",
        "data": {
            "total_logs": total,
            "avg_intent_outcome_gap": round(avg_gap, 4),
            "avg_thought_action_coherence": round(avg_coherence, 4),
            "high_gap_count": high_gap_count,
        },
        "meta": {"processing_time_ms": 0, "timestamp": datetime.now(timezone.utc).isoformat()},
    }
