from __future__ import annotations

from dataclasses import dataclass


INTENT_HINTS = {
    "communication": ["send", "notify", "message", "email", "slack", "ticket", "comment", "post"],
    "data": ["query", "database", "sql", "record", "fetch", "lookup"],
    "search": ["search", "find", "discover", "look up"],
    "file": ["file", "document", "pdf", "export", "asset"],
    "api": ["api", "endpoint", "token", "oauth"],
    "delegation": ["delegate", "subagent", "route"],
    "analytics": ["forecast", "trend", "dashboard", "anomaly", "kpi"],
    "nlp": ["summarize", "translate", "entity", "keyword"],
    "design": ["figma", "design", "component", "node", "ui", "ux", "accessibility"],
}

TOOL_DOMAIN = {
    "query_database": "data",
    "fetch_schema": "data",
    "run_sql": "data",
    "export_csv": "data",
    "import_csv": "data",
    "stream_table": "data",
    "get_row_count": "data",
    "upsert_records": "data",
    "web_search": "search",
    "semantic_search": "search",
    "knowledge_lookup": "search",
    "news_lookup": "search",
    "fact_check": "search",
    "crawl_site": "search",
    "search_index": "search",
    "send_email": "communication",
    "post_slack": "communication",
    "create_ticket": "communication",
    "schedule_meeting": "communication",
    "notify_webhook": "communication",
    "send_sms": "communication",
    "escalate_incident": "communication",
    "read_file": "file",
    "write_file": "file",
    "parse_pdf": "file",
    "extract_table": "file",
    "summarize_doc": "file",
    "convert_doc": "file",
    "archive_file": "file",
    "call_rest_api": "api",
    "authenticate_oauth": "api",
    "refresh_token": "api",
    "get_api_schema": "api",
    "post_json": "api",
    "retry_request": "api",
    "cache_response": "api",
    "delegate_to_agent": "delegation",
    "invoke_subagent": "delegation",
    "dispatch_task": "delegation",
    "route_workflow": "delegation",
    "set_reminder": "productivity",
    "create_todo": "productivity",
    "generate_report": "analytics",
    "build_dashboard": "analytics",
    "forecast_series": "analytics",
    "detect_outlier": "analytics",
    "map_entities": "nlp",
    "extract_keywords": "nlp",
    "translate_text": "nlp",
    "summarize_text": "nlp",
    "get_file": "design",
    "get_node": "design",
    "list_components": "design",
    "export_asset": "design",
    "post_comment": "design",
}


@dataclass
class IntentOutcomeScore:
    gap_score: float
    coherence_score: float
    intent_domain: str | None
    action_domain: str | None


def _infer_intent_domain(intent: str) -> str | None:
    lower = intent.lower()
    for domain, hints in INTENT_HINTS.items():
        if any(token in lower for token in hints):
            return domain
    return None


def compute_intent_outcome_gap(metadata: dict) -> IntentOutcomeScore:
    intent = str(metadata.get("intent") or metadata.get("task_intent") or metadata.get("event") or "")
    tool = str(metadata.get("tool_name") or "")
    response_status = str(metadata.get("response_status") or "").lower()

    intent_domain = _infer_intent_domain(intent)
    action_domain = TOOL_DOMAIN.get(tool)

    gap = 0.4
    if intent_domain is not None and action_domain is not None:
        gap = 0.1 if intent_domain == action_domain else 0.85
    elif tool:
        gap = 0.55

    if response_status in {"error", "timeout"}:
        gap = min(1.0, gap + 0.15)

    if bool(metadata.get("is_anomaly")):
        gap = min(1.0, gap + 0.1)

    return IntentOutcomeScore(
        gap_score=round(max(0.0, min(1.0, gap)), 4),
        coherence_score=round(max(0.0, min(1.0, 1.0 - gap)), 4),
        intent_domain=intent_domain,
        action_domain=action_domain,
    )
