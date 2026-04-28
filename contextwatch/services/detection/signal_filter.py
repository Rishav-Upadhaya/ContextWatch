"""Signal-focused filter that strips protocol overhead.

Retains only 'decisions, outcomes, and exceptions' from MCP/A2A logs.
Removes 500-2000 tokens of metadata (schemas, envelopes, type definitions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Keys that carry signal vs keys that are protocol plumbing
SIGNAL_KEYS_MCP = frozenset({
    "method", "tool_name", "tool_call_id", "arguments", "result",
    "error", "error_code", "error_message", "status",
    "intent", "reasoning_step", "decision", "outcome",
    "exception", "triggered_by", "message", "data",
    "parameters", "response", "summary",
})

SIGNAL_KEYS_A2A = frozenset({
    "message_content", "task_intent", "target_agent", "source_agent",
    "result", "error", "status", "outcome", "decision",
    "exception", "response", "summary", "artifacts",
})

# Keys that are always protocol plumbing and should be stripped
PLUMBING_KEYS_MCP = frozenset({
    "jsonrpc", "id", "protocol_version", "transport",
    "connected_at", "host", "server", "session",
    "meta", "schema", "type", "additionalProperties",
    "required", "properties", "definitions", "$ref",
    "content_type", "encoding", "version",
})

PLUMBING_KEYS_A2A = frozenset({
    "protocol_version", "message_id", "correlation_id",
    "routing", "envelope", "headers", "transport",
    "schema", "type", "content_type", "encoding",
})


@dataclass
class FilteredSignal:
    """Cleaned signal from a log after protocol plumbing removal."""
    log_id: str
    protocol: str
    signal_text: str
    raw_tokens_preserved: int = 0
    overhead_tokens_removed: int = 0
    redacted_fields: list[str] = field(default_factory=list)


def estimate_token_count(text: str) -> int:
    """Rough token count: ~4 chars per token on average."""
    return max(1, len(text) // 4)


def _strip_plumbing(data: dict, protocol: str, depth: int = 0, max_depth: int = 4) -> dict:
    """Recursively strip protocol plumbing, keeping signal-bearing fields."""
    if depth > max_depth or not isinstance(data, dict):
        return data

    plumbing = PLUMBING_KEYS_MCP if protocol == "MCP" else PLUMBING_KEYS_A2A
    result: dict[str, Any] = {}

    for key, value in data.items():
        if key in plumbing:
            continue
        if isinstance(value, dict):
            # Inline dicts that are purely schema/definition data
            if _is_schema_blob(value):
                continue
            cleaned = _strip_plumbing(value, protocol, depth + 1, max_depth)
            if cleaned:
                result[key] = cleaned
        elif isinstance(value, list):
            cleaned_list = [
                _strip_plumbing(item, protocol, depth + 1, max_depth)
                if isinstance(item, dict) else item
                for item in value
            ]
            if cleaned_list:
                result[key] = cleaned_list
        elif value is not None and value != "":
            result[key] = value

    return result


def _is_schema_blob(data: dict) -> bool:
    """Detect if a dict is purely JSON Schema metadata (no actual data)."""
    schema_indicators = {"type", "properties", "required", "additionalProperties"}
    if not data:
        return False
    keys = set(data.keys())
    return len(keys & schema_indicators) >= 2


def _extract_signal_text(data: dict, indent: int = 0) -> str:
    """Flatten a cleaned dict into a single-line signal string."""
    parts: list[str] = []
    for key, value in data.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            parts.append(f"{prefix}{_extract_signal_text(value, indent + 1)}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    parts.append(f"{prefix}{key}[{i}]: {_extract_signal_text(item, indent + 1)}")
                else:
                    parts.append(f"{prefix}{key}[{i}]: {item}")
        else:
            parts.append(f"{prefix}{key}: {value}")
    return " | ".join(parts)


def filter_log_signal(log: dict, protocol: str) -> FilteredSignal:
    """Filter a raw MCP/A2A log, removing protocol plumbing.

    Returns a FilteredSignal containing:
    - Clean signal text preserving decisions/outcomes/exceptions
    - Estimated token counts before and after filtering
    """
    log_id = (
        log.get("log_id")
        or log.get("params", {}).get("log_id")
        or "unknown"
    )

    original_text = str(log)
    original_tokens = estimate_token_count(original_text)

    cleaned = _strip_plumbing(log, protocol, depth=0)
    signal_text = _extract_signal_text(cleaned)
    signal_tokens = estimate_token_count(signal_text)

    removed = max(0, original_tokens - signal_tokens)

    return FilteredSignal(
        log_id=log_id,
        protocol=protocol,
        signal_text=signal_text,
        raw_tokens_preserved=signal_tokens,
        overhead_tokens_removed=removed,
    )


def filter_signal_and_redact(log: dict, protocol: str) -> FilteredSignal:
    """Backward-compatible alias for older callers."""
    return filter_log_signal(log, protocol)


"""End of signal_filter module."""
