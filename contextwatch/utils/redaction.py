"""Secret redaction hooks.

Scans log payloads for API keys, PII, and sensitive tokens using
regex patterns and heuristic matching. All redaction happens BEFORE
the log reaches the domain detection layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Redaction patterns
SECRET_PATTERNS: dict[str, re.Pattern] = {
    # API Keys
    "aws_access_key": re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}", re.IGNORECASE),
    "aws_secret_key": re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*\S{20,}"),
    "stripe_key": re.compile(r"(?:sk|pk|rk)_live_[0-9a-zA-Z]{24,}", re.IGNORECASE),
    "stripe_test_key": re.compile(r"(?:sk|pk|rk)_test_[0-9a-zA-Z]{24,}", re.IGNORECASE),
    "openai_key": re.compile(r"sk-[a-zA-Z0-9]{20,}T3BlbkFJ[a-zA-Z0-9]{20,}", re.IGNORECASE),
    "anthropic_key": re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}", re.IGNORECASE),
    "generic_api_key": re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[a-zA-Z0-9]{16,}", re.IGNORECASE),
    "bearer_token": re.compile(r"(?:bearer|token)\s+['\"]?[a-zA-Z0-9\-._~+/]{20,}['\"]?", re.IGNORECASE),

    # PII - PII
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
    "ssn": re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
    "credit_card": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),

    # Generic secrets
    "private_key_header": re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
    "jwt_token": re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"),
}

REDACTED_PLACEHOLDER = "[REDACTED]"


@dataclass
class RedactionResult:
    data: Any
    redacted_fields: list[str] = field(default_factory=list)
    secrets_found_count: int = 0


def _redact_string(text: str) -> tuple[str, list[str]]:
    """Redact secrets from a single string. Returns (cleaned_text, fields_found)."""
    found_fields: list[str] = []
    for pattern_name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            found_fields.append(pattern_name)
            text = pattern.sub(REDACTED_PLACEHOLDER, text)
    return text, found_fields


def _redact_value(value: Any) -> tuple[Any, list[str]]:
    """Recursively redact secrets from any value."""
    found_fields: list[str] = []

    if isinstance(value, str):
        cleaned, fields = _redact_string(value)
        found_fields.extend(fields)
        return cleaned, found_fields

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            # Also check key name for secret indicators
            if any(indicator in k.lower() for indicator in
                   ("secret", "password", "token", "key", "api_key", "access_key",
                    "private_key", "credential", "auth")):
                result[k] = REDACTED_PLACEHOLDER
                found_fields.append(f"key_name:{k}")
            else:
                cleaned_v, fields = _redact_value(v)
                result[k] = cleaned_v
                found_fields.extend(fields)
        return result, found_fields

    if isinstance(value, list):
        result_list = []
        for item in value:
            cleaned_item, fields = _redact_value(item)
            result_list.append(cleaned_item)
            found_fields.extend(fields)
        return result_list, found_fields

    return value, found_fields


def redact_log(log: dict) -> RedactionResult:
    """Redact secrets and PII from a log payload.

    Must run before the log enters the domain layer.
    """
    cleaned_data, fields_found = _redact_value(log)
    return RedactionResult(
        data=cleaned_data,
        redacted_fields=list(set(fields_found)),  # deduplicate
        secrets_found_count=len(fields_found),
    )


"""End of redaction module."""
