"""Pydantic schemas for ContextWatch API."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# ── Pydantic request / response models ───────────────────────────────────────

class LogIngestRequest(BaseModel):
    """Raw log payload (MCP or A2A)."""
    log_id: Optional[str] = None
    protocol: str = "MCP"
    session: Optional[Dict[str, Any]] = None
    jsonrpc: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

    class Config:
        extra = "allow"


class LogIngestResponse(BaseModel):
    success: bool
    log_id: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: Optional[str] = None
    confidence: float
    cosine_distance: Optional[float] = None
    explanation: str
    token_efficiency: Optional[Dict[str, int]] = None
    trace_context: Optional[Dict[str, str]] = None


class BatchLogIngestRequest(BaseModel):
    logs: List[LogIngestRequest]
    include_rca: bool = False


class BatchIngestResponse(BaseModel):
    batch_size: int
    anomaly_count: int
    normal_count: int
    results: List[LogIngestResponse]
    graph: Dict[str, Any]


class AnomalyQueryResponse(BaseModel):
    anomalies: List[Dict[str, Any]]
    total: int
    page: int
    limit: int


class NormalLogQueryResponse(BaseModel):
    logs: List[Dict[str, Any]]
    total: int
    page: int
    limit: int


class RCAResponse(BaseModel):
    anomaly_id: str
    causal_chain: List[Dict[str, Any]]
    evidence: List[str]
    confidence: float


class FinetuneRequest(BaseModel):
    """Logs to use for finetuning LogBERT."""
    logs: List[Dict[str, Any]]
    epochs: int = 5
    learning_rate: float = 1e-3
    use_system_data: bool = True


class FinetuneResponse(BaseModel):
    success: bool
    epochs_completed: int
    n_sequences: int
    metrics: List[Dict[str, Any]]
    vhm_radius_after: float
    decision_radius_after: float
    calibration: Optional[Dict[str, Any]] = None


class ModelInfoResponse(BaseModel):
    model_type: str
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    vocab_size: int
    vhm_fitted: bool
    vhm_radius: float
    decision_radius: float
    calibration: Optional[Dict[str, Any]] = None


# ── Error Explanation / Reasoning ────────────────────────────────────────────

class ErrorExplanationRequest(BaseModel):
    """Request for agentic error explanation."""
    anomaly_id: str
    error_type: str  # TOOL_HALLUCINATION, CONTEXT_POISONING, etc.
    raw_log: Optional[Dict[str, Any]] = None


class ErrorExplanationResponse(BaseModel):
    """Agentic error explanation with RCA and remediation."""
    anomaly_id: str
    error_type: str
    root_cause_analysis: str
    remediation_steps: List[str]
    prevention_measures: List[str]
    confidence_level: str  # low | medium | high
    signal_confidence: float
    execution_trace: List[str]
    created_at: str
