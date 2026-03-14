from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

from config.settings import Settings
from core.schema import AnomalyResult, ClassificationResult, NormalizedLog

KNOWN_TOOLS = {
    "query_database", "fetch_schema", "run_sql", "export_csv", "import_csv", "stream_table", "get_row_count",
    "upsert_records", "web_search", "semantic_search", "knowledge_lookup", "news_lookup", "fact_check",
    "crawl_site", "search_index", "send_email", "post_slack", "create_ticket", "schedule_meeting",
    "notify_webhook", "send_sms", "escalate_incident", "read_file", "write_file", "parse_pdf", "extract_table",
    "summarize_doc", "convert_doc", "archive_file", "call_rest_api", "authenticate_oauth", "refresh_token",
    "get_api_schema", "post_json", "retry_request", "cache_response", "delegate_to_agent", "invoke_subagent",
    "dispatch_task", "route_workflow", "set_reminder", "create_todo", "generate_report", "build_dashboard",
    "forecast_series", "detect_outlier", "map_entities", "extract_keywords", "translate_text", "summarize_text",
    "get_file", "get_node", "list_components", "export_asset", "post_comment",
}

TOOL_DOMAIN = {
    "query_database": "data", "fetch_schema": "data", "run_sql": "data", "export_csv": "data",
    "import_csv": "data", "stream_table": "data", "get_row_count": "data", "upsert_records": "data",
    "web_search": "search", "semantic_search": "search", "knowledge_lookup": "search", "news_lookup": "search",
    "fact_check": "search", "crawl_site": "search", "search_index": "search", "send_email": "communication",
    "post_slack": "communication", "create_ticket": "communication", "schedule_meeting": "communication",
    "notify_webhook": "communication", "send_sms": "communication", "escalate_incident": "communication",
    "read_file": "file", "write_file": "file", "parse_pdf": "file", "extract_table": "file",
    "summarize_doc": "file", "convert_doc": "file", "archive_file": "file", "call_rest_api": "api",
    "authenticate_oauth": "api", "refresh_token": "api", "get_api_schema": "api", "post_json": "api",
    "retry_request": "api", "cache_response": "api", "delegate_to_agent": "delegation",
    "invoke_subagent": "delegation", "dispatch_task": "delegation", "route_workflow": "delegation",
    "set_reminder": "productivity", "create_todo": "productivity", "generate_report": "analytics",
    "build_dashboard": "analytics", "forecast_series": "analytics", "detect_outlier": "analytics",
    "map_entities": "nlp", "extract_keywords": "nlp", "translate_text": "nlp", "summarize_text": "nlp",
    "get_file": "design", "get_node": "design", "list_components": "design", "export_asset": "design", "post_comment": "design",
}

INTENT_DOMAIN_HINTS = {
    "communication": ["send", "notify", "message", "report", "email", "slack", "ticket"],
    "data": ["query", "database", "sql", "extract", "record", "invoice"],
    "search": ["search", "lookup", "find", "discover"],
    "file": ["file", "pdf", "document", "read", "write", "extract"],
    "api": ["api", "endpoint", "oauth", "token", "integration"],
    "delegation": ["delegate", "subagent", "route"],
    "analytics": ["forecast", "anomaly", "dashboard", "trend", "report"],
    "nlp": ["summarize", "translate", "keyword", "entity"],
    "design": ["design", "figma", "component", "node", "dashboard", "ui", "ux", "accessibility"],
}


class RuleBasedClassifier:
    def __init__(self) -> None:
        self.known_tools = KNOWN_TOOLS

    def _duplicate_ngram_ratio(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 8:
            return 0.0
        ngrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        uniq = len(set(ngrams))
        return 1.0 - (uniq / max(len(ngrams), 1))

    def _infer_intent_domain(self, intent: str) -> str | None:
        lower = intent.lower()
        for domain, hints in INTENT_DOMAIN_HINTS.items():
            if any(token in lower for token in hints):
                return domain
        return None

    def classify(self, log: NormalizedLog) -> str | None:
        md = log.metadata
        params = md.get("params", {}) if isinstance(md.get("params", {}), dict) else {}
        data = params.get("data", {}) if isinstance(params.get("data", {}), dict) else {}
        raw_meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}

        tool_name = str(md.get("tool_name") or raw_meta.get("tool") or "").strip()
        intent = str(md.get("intent", md.get("task_intent", "")))
        response_status = str(md.get("response_status", "")).lower()
        event = str(md.get("event") or data.get("event") or "").upper()
        level = str(md.get("level") or params.get("level") or "").lower()
        reasoning_step = str(md.get("reasoning_step") or data.get("message") or "")

        if log.protocol == "MCP":
            event_map = {
                "TOOL_HALLUCINATION": "TOOL_HALLUCINATION",
                "CONTEXT_POISONING": "CONTEXT_POISONING",
                "REGISTRY_OVERFLOW": "REGISTRY_OVERFLOW",
                "DELEGATION_CHAIN_FAILURE": "DELEGATION_CHAIN_FAILURE",
            }
            if event in event_map:
                return event_map[event]

            combined = " ".join([event.lower(), level, reasoning_step.lower(), json.dumps(raw_meta).lower()])
            if any(token in combined for token in ["hallucination", "non-existent tool", "invalid param", "not supported"]):
                return "TOOL_HALLUCINATION"
            if any(token in combined for token in ["context poisoning", "prompt injection", "buffer-stuffing", "token-stuffing", "instruction override"]):
                return "CONTEXT_POISONING"
            if any(token in combined for token in ["registry overflow", "table size limit", "bucket full", "memory pressure"]):
                return "REGISTRY_OVERFLOW"
            if any(token in combined for token in ["delegation chain failure", "chain broken", "credential hand-off", "depends on", "upstream"]):
                return "DELEGATION_CHAIN_FAILURE"

        if tool_name and tool_name not in self.known_tools:
            return "TOOL_HALLUCINATION"

        tool_parameters = md.get("tool_parameters", {}) if isinstance(md.get("tool_parameters", {}), dict) else {}
        if any(re.search(r"fake|hallucinated|undefined", str(k), re.IGNORECASE) for k in tool_parameters):
            return "TOOL_HALLUCINATION"

        context_tokens = int(md.get("context_window_tokens", 0) or 0)
        context_summary = str(md.get("context_summary", ""))
        response_payload = md.get("response_payload", {})
        context_carried = md.get("context_carried", {})
        context_carried_len = len(json.dumps(context_carried)) if isinstance(context_carried, dict) else 0
        requests_used = int(raw_meta.get("requests_used", 0) or 0)
        requests_limit = int(raw_meta.get("requests_limit", 0) or 0)
        if (
            context_tokens > 12_000
            or self._duplicate_ngram_ratio(context_summary) > 0.3
            or (response_status in {"partial", "success"} and isinstance(response_payload, dict) and len(response_payload) == 0)
            or context_carried_len > 5000
            or ("RATE_LIMIT" in event and requests_limit > 0 and requests_used >= int(0.6 * requests_limit))
            or bool(raw_meta.get("truncated"))
        ):
            return "CONTEXT_POISONING"

        delegation_depth = int(md.get("delegation_depth", 0) or 0)
        if (
            (response_status == "error" and any(token in tool_name.lower() for token in ["delegation", "subagent", "delegate"]))
            or (delegation_depth >= 3 and response_status == "error")
            or (response_status in {"error", "timeout"} and any(token in event for token in ["TIMEOUT", "FAIL", "ERROR"]))
            or (level == "error" and "TOOL_CALL" in event)
        ):
            return "DELEGATION_CHAIN_FAILURE"

        if tool_name in self.known_tools:
            tool_domain = TOOL_DOMAIN.get(tool_name)
            intent_domain = self._infer_intent_domain(intent)
            if tool_domain and intent_domain and tool_domain != intent_domain:
                return "REGISTRY_OVERFLOW"
        return None


class LLMClassifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = anthropic.Anthropic(api_key=settings.LLM_API_KEY) if settings.LLM_API_KEY else None

    def classify(self, log: NormalizedLog, context_logs: list[NormalizedLog]) -> ClassificationResult:
        if self.client is None:
            return ClassificationResult(
                anomaly_type=None,
                confidence=0.50,
                method="llm",
                reasoning="LLM API key unavailable; no deterministic fallback label matched.",
            )
        prompt = {
            "log": log.model_dump(mode="json"),
            "context_logs": [l.model_dump(mode="json") for l in context_logs[-3:]],
            "labels": [
                "TOOL_HALLUCINATION",
                "CONTEXT_POISONING",
                "REGISTRY_OVERFLOW",
                "DELEGATION_CHAIN_FAILURE",
            ],
        }
        message = self.client.messages.create(
            model=self.settings.LLM_MODEL,
            max_tokens=self.settings.LLM_MAX_TOKENS,
            system=(
                "You are an MCP/A2A log anomaly classifier. Classify into exactly one type and return JSON only "
                "with keys anomaly_type, confidence, reasoning."
            ),
            messages=[{"role": "user", "content": json.dumps(prompt)}],
        )
        text = message.content[0].text if message.content else "{}"
        payload = json.loads(text)
        return ClassificationResult(
            anomaly_type=payload.get("anomaly_type"),
            confidence=float(payload.get("confidence", 0.6)),
            method="llm",
            reasoning=payload.get("reasoning"),
        )


class AnomalyClassifier:
    def __init__(self, settings: Settings) -> None:
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMClassifier(settings)

    def classify(
        self,
        log: NormalizedLog,
        anomaly_result: AnomalyResult,
        context_logs: list[NormalizedLog],
    ) -> ClassificationResult:
        if not anomaly_result.is_anomaly:
            return ClassificationResult(anomaly_type=None, confidence=1.0, method="none", reasoning="Not anomalous")

        rule_result = self.rule_classifier.classify(log)
        if rule_result is not None:
            return ClassificationResult(anomaly_type=rule_result, confidence=0.85, method="rule", reasoning="Rule match")

        if log.protocol == "MCP":
            return ClassificationResult(
                anomaly_type=None,
                confidence=0.70,
                method="rule",
                reasoning="No explicit MCP anomaly signature matched",
            )

        return self.llm_classifier.classify(log, context_logs)
