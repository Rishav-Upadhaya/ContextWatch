from __future__ import annotations

from core.schema import AnomalyType


REAL_EVENT_TO_SUBTYPE = {
    "UNKNOWN_TOOL": "unknown_tool_call",
    "SCHEMA_VALIDATION_ERROR": "invalid_tool_parameter",
    "FIGMA_API_ERROR": "upstream_api_failure",
    "FIGMA_API_AUTH_ERROR": "auth_failure",
    "FIGMA_API_SLOW": "latency_spike",
    "FIGMA_API_WARN": "resource_not_found",
    "TOOL_CALL_ERROR": "tool_execution_failure",
    "TOOL_CALL_RETRY": "retry_triggered",
}

SYNTHETIC_EVENTS = {
    "TOOL_HALLUCINATION",
    "CONTEXT_POISONING",
    "REGISTRY_OVERFLOW",
    "DELEGATION_CHAIN_FAILURE",
}

SUBTYPE_TO_LEGACY: dict[str, AnomalyType] = {
    "unknown_tool_call": "TOOL_HALLUCINATION",
    "invalid_tool_parameter": "TOOL_HALLUCINATION",
    "synthetic_test_data": "TOOL_HALLUCINATION",
    "prompt_injection_attempt": "CONTEXT_POISONING",
    "code_injection_attempt": "CONTEXT_POISONING",
    "sql_injection_attempt": "CONTEXT_POISONING",
    "input_anomaly": "CONTEXT_POISONING",
    "resource_not_found": "REGISTRY_OVERFLOW",
    "rate_limit_breach": "REGISTRY_OVERFLOW",
    "upstream_api_failure": "REGISTRY_OVERFLOW",
    "upstream_server_error": "REGISTRY_OVERFLOW",
    "auth_failure": "DELEGATION_CHAIN_FAILURE",
    "latency_spike": "REGISTRY_OVERFLOW",
    "retry_storm_indicator": "DELEGATION_CHAIN_FAILURE",
    "tool_execution_failure": "DELEGATION_CHAIN_FAILURE",
    "orphaned_retry_logic_error": "DELEGATION_CHAIN_FAILURE",
    "tight_retry_loop": "DELEGATION_CHAIN_FAILURE",
    "runaway_retry_loop": "DELEGATION_CHAIN_FAILURE",
    "inconsistent_state": "DELEGATION_CHAIN_FAILURE",
}


def map_subtype_to_legacy(subtype: str | None) -> AnomalyType | None:
    if not subtype:
        return None
    return SUBTYPE_TO_LEGACY.get(subtype)
