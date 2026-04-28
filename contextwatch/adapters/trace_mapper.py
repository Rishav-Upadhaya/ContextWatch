"""OpenTelemetry trace mapper.

Maps legacy identifiers (e.g., Unisys mix numbers, proprietary trace IDs)
to standardized OpenTelemetry trace parents to maintain trace continuity
across asynchronous A2A tasks and MCP sessions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TraceContext:
    """OpenTelemetry-compatible trace context."""
    trace_id: str  # 32-char hex (16 bytes)
    span_id: str   # 16-char hex (8 bytes)
    trace_flags: str = "01"  # sampled
    trace_state: str = ""

    def to_w3c_header(self) -> str:
        """Format as W3C Trace Context header: version-traceid-spanid-flags."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags}"


@dataclass
class MappedTrace:
    """Result of mapping legacy identifiers to OTEL trace context."""
    original_id: str
    original_system: str
    trace_context: TraceContext
    parent_span_id: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)


# Known legacy ID patterns
LEGACY_PATTERNS: dict[str, re.Pattern] = {
    "unisys_mix": re.compile(r"(?:MIX|mix)[-_]?(\d{6,10})", re.IGNORECASE),
    "proprietary_trace": re.compile(r"(?:trace|tracing)[-_]?id\s*[:=]\s*['\"]?([a-f0-9\-]{8,36})", re.IGNORECASE),
    "request_id": re.compile(r"(?:request|req)[-_]?id\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{8,36})", re.IGNORECASE),
    "correlation_id": re.compile(r"(?:correlation|corr)[-_]?id\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{8,36})", re.IGNORECASE),
    "session_id": re.compile(r"(?:session|sess)[-_]?id\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{8,36})", re.IGNORECASE),
}


def _legacy_id_to_trace_id(legacy_id: str) -> TraceContext:
    """Deterministically map a legacy identifier to an OTEL trace context.

    Uses SHA-256 of the original ID, taking first 16 bytes for trace_id
    and bytes 16-24 for span_id.
    """
    hash_bytes = hashlib.sha256(legacy_id.encode("utf-8")).hexdigest()
    trace_id = hash_bytes[:32]
    span_id = hash_bytes[32:48]
    return TraceContext(trace_id=trace_id, span_id=span_id)


def _extract_legacy_ids(log: dict) -> list[tuple[str, str]]:
    """Scan a log dict for legacy identifiers. Returns (system, id) pairs."""
    matches: list[tuple[str, str]] = []

    def scan_dict(d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                scan_dict(value)
            elif isinstance(value, str):
                # Check if key itself signals a legacy ID pattern
                for pattern_name, pattern in LEGACY_PATTERNS.items():
                    # Check value against pattern
                    val_match = pattern.search(value)
                    if val_match:
                        extracted = val_match.group(1)
                        matches.append((pattern_name, extracted))
                    # Also check if key name matches
                    if pattern_name.replace("_", "_") in key.lower() or \
                       any(ind in key.lower() for ind in ("mix_", "trace_id", "req_id")):
                        matches.append((key, value))

    scan_dict(log)
    return matches


def map_to_otel_trace(log: dict, protocol: str) -> Optional[MappedTrace]:
    """Map legacy identifiers in a log to OpenTelemetry trace context.

    Returns None if no legacy IDs were found.
    """
    legacy_ids = _extract_legacy_ids(log)
    if not legacy_ids:
        # Fall back: use log_id as trace anchor
        log_id = log.get("log_id", log.get("id", ""))
        if log_id:
            trace_ctx = _legacy_id_to_trace_id(str(log_id))
            return MappedTrace(
                original_id=str(log_id),
                original_system=protocol,
                trace_context=trace_ctx,
                attributes={"protocol": protocol},
            )
        return None

    # Use first found legacy ID as primary trace anchor
    system, legacy_id = legacy_ids[0]
    trace_ctx = _legacy_id_to_trace_id(legacy_id)

    attributes = {"protocol": protocol}
    parent_span_id = None

    # If multiple IDs found, chain them as parent spans
    for idx, (sys_name, lid) in enumerate(legacy_ids[1:], start=1):
        child_ctx = _legacy_id_to_trace_id(lid)
        if idx == 1:
            parent_span_id = child_ctx.span_id
        attributes[f"legacy_{sys_name}"] = lid

    return MappedTrace(
        original_id=legacy_id,
        original_system=system,
        trace_context=trace_ctx,
        parent_span_id=parent_span_id,
        attributes=attributes,
    )


"""End of trace_mapper module."""
