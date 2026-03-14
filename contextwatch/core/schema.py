from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AnomalyType = Literal[
    "TOOL_HALLUCINATION",
    "CONTEXT_POISONING",
    "REGISTRY_OVERFLOW",
    "DELEGATION_CHAIN_FAILURE",
]


class MCPLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class Session(BaseModel):
        model_config = ConfigDict(extra="allow")

        id: str
        host: str = ""
        server: str = ""
        connected_at: datetime | None = None
        transport: str = ""

    class ParamsData(BaseModel):
        model_config = ConfigDict(extra="allow")

        timestamp: datetime
        event: str
        message: str
        meta: dict[str, Any] = Field(default_factory=dict)

    class Params(BaseModel):
        model_config = ConfigDict(extra="allow")

        level: Literal["debug", "info", "warning", "error"]
        logger: str
        data: ParamsData

    log_id: UUID = Field(default_factory=uuid4)
    protocol: Literal["MCP"] = "MCP"
    session: Session
    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["notifications/message"] = "notifications/message"
    params: Params
    is_anomaly: bool | None = None
    anomaly_type: AnomalyType | None = None

    @field_validator("protocol", mode="before")
    @classmethod
    def default_protocol(cls, value: Any) -> str:
        return "MCP" if value is None else value

    @field_validator("session")
    @classmethod
    def session_id_non_empty(cls, value: Session) -> Session:
        if not value.id.strip():
            raise ValueError("session.id must not be empty")
        return value

    @field_validator("params")
    @classmethod
    def params_non_empty(cls, value: Params) -> Params:
        if not value.logger.strip():
            raise ValueError("params.logger must not be empty")
        if not value.data.event.strip():
            raise ValueError("params.data.event must not be empty")
        if not value.data.message.strip():
            raise ValueError("params.data.message must not be empty")
        return value

    @field_validator("anomaly_type")
    @classmethod
    def anomaly_type_requires_label(cls, value: AnomalyType | None, info) -> AnomalyType | None:
        is_anomaly = info.data.get("is_anomaly")
        if value is not None and is_anomaly is False:
            raise ValueError("anomaly_type must be null when is_anomaly is false")
        return value


class A2ALog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_id: UUID
    protocol: Literal["A2A"]
    timestamp: datetime
    session_id: UUID
    source_agent: str
    target_agent: str
    delegation_depth: int = Field(default=0, ge=0, le=5)
    delegation_chain: list[str] = Field(default_factory=list)
    message_type: Literal["task_delegation", "result_return", "clarification", "error_propagation"]
    message_content: str
    task_intent: str
    context_carried: dict[str, Any] = Field(default_factory=dict)
    response_status: Literal["success", "error", "timeout", "partial"]
    response_content: str = ""
    latency_ms: int = Field(ge=0)
    is_anomaly: bool | None = None
    anomaly_type: AnomalyType | None = None

    @field_validator("source_agent", "target_agent", "message_content", "task_intent")
    @classmethod
    def a2a_non_empty_str(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_chain_len(self) -> "A2ALog":
        if not self.delegation_chain:
            self.delegation_chain = [self.source_agent] if self.delegation_depth == 0 else [self.source_agent, self.target_agent]
        expected = self.delegation_depth + 1
        if len(self.delegation_chain) != expected:
            raise ValueError("delegation_chain length must equal delegation_depth + 1")
        return self


class NormalizedLog(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    log_id: str
    session_id: str
    timestamp: datetime
    protocol: Literal["MCP", "A2A"]
    agent_id: str
    text_for_embedding: str
    metadata: dict[str, Any]

    @field_validator("text_for_embedding")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text_for_embedding must not be empty")
        return value


class AnomalyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    log_id: str
    anomaly_score: float = Field(ge=0.0, le=2.0)
    is_anomaly: bool
    anomaly_type: AnomalyType | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rule_lane_triggered: bool = False
    embedding_lane_score: float | None = None
    embedding_lane_threshold: float | None = None
    arbitration_mode: str = "rule_first"


class RCAResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    root_cause_log_id: str
    causal_chain: list[str]
    hop_count: int = Field(ge=0)
    explanation: str


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    anomaly_type: AnomalyType | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["none", "rule", "llm"]
    reasoning: str | None = None


MCPConfidence = Literal["high", "medium", "low"]
MCPSignal = Literal["signal_1", "signal_2", "signal_3", "signal_4"]


class MCPFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    finding_id: str
    session_id: str
    log_id: str
    timestamp: datetime
    anomaly_type: str
    confidence: MCPConfidence
    signals_triggered: list[MCPSignal]
    event: str
    level: Literal["debug", "info", "warning", "error"]
    summary: str
    detail: str
    recommended_action: str
    correlated_log_ids: list[str] = Field(default_factory=list)
    context_rule_ids: list[str] = Field(default_factory=list)
    legacy_anomaly_type: AnomalyType | None = None
    subtype: str | None = None
    synthetic_test_data: bool = False
    ml_score: float | None = None
    ml_label: str | None = None
    promotion_source: str = "rule_only"
    policy_decision: str = "promoted"


class MCPSessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str
    total_logs_analysed: int = Field(ge=0)
    anomalies_found: int = Field(ge=0)
    anomaly_density_pct: float = Field(ge=0.0, le=100.0)
    by_type: dict[str, int] = Field(default_factory=dict)
    by_confidence: dict[str, int] = Field(default_factory=dict)
    by_level: dict[str, int] = Field(default_factory=dict)
    warning_logs: int = Field(default=0, ge=0)
    error_logs: int = Field(default=0, ge=0)
    non_anomalous_warnings: int = Field(default=0, ge=0)
    non_anomalous_errors: int = Field(default=0, ge=0)
    policy_stats: dict[str, int] = Field(default_factory=dict)
    highest_risk_finding_id: str | None = None
    assessment: str


class MCPFindingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    findings: list[MCPFinding]
    session_summary: MCPSessionSummary
