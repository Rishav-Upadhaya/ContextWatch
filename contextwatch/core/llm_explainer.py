from __future__ import annotations

import json
import logging

import anthropic
import openai

from config.settings import Settings
from core.schema import ClassificationResult, NormalizedLog, RCAResult

logger = logging.getLogger("contextwatch.llm_explainer")


class LLMExplainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache: dict[str, str] = {}
        self.provider = settings.LLM_PROVIDER.lower()
        
        # Initialize appropriate client based on provider
        if self.provider == "anthropic" and settings.LLM_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.LLM_API_KEY)
            self.openai_client = None
            logger.info("Initialized Anthropic LLM client")
        elif self.provider == "openai" and settings.OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            self.anthropic_client = None
            logger.info("Initialized OpenAI LLM client")
        else:
            self.anthropic_client = None
            self.openai_client = None
            logger.warning("No LLM API key provided, using fallback explanations")

    def explain_anomaly(
        self,
        anomaly_log: NormalizedLog,
        classification: ClassificationResult,
        context_logs: list[NormalizedLog],
        rca_result: RCAResult,
        anomaly_score: float,
        confidence: float,
    ) -> str:
        if confidence <= 0.7:
            return "Confidence below explanation threshold; skipping LLM explanation."

        if anomaly_log.log_id in self.cache:
            return self.cache[anomaly_log.log_id]

        # Fallback if no client is available
        if self.anthropic_client is None and self.openai_client is None:
            fallback = (
                f"Detected {classification.anomaly_type} with anomaly score {anomaly_score:.3f}. "
                f"Likely root cause log is {rca_result.root_cause_log_id}."
            )
            self.cache[anomaly_log.log_id] = fallback
            return fallback

        payload = {
            "anomaly_type": classification.anomaly_type,
            "anomalous_log": anomaly_log.model_dump(mode="json"),
            "context_logs": [l.model_dump(mode="json") for l in context_logs[-3:]],
            "causal_chain": rca_result.causal_chain,
            "anomaly_score": anomaly_score,
        }
        
        system_prompt = (
            "You are an AI system observability expert analyzing MCP/A2A agent logs. "
            "Provide a concise technical explanation under 200 words: intent, observed behavior, "
            "why anomalous, probable root cause."
        )
        
        try:
            if self.anthropic_client:
                text = self._explain_with_anthropic(payload, system_prompt)
            else:
                text = self._explain_with_openai(payload, system_prompt)
            
            # Truncate to 200 words
            words = text.split()
            if len(words) > 200:
                text = " ".join(words[:200])
            
            self.cache[anomaly_log.log_id] = text
            logger.info(f"Generated explanation for {anomaly_log.log_id} using {self.provider}")
            return text
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            fallback = f"Detected {classification.anomaly_type} (error generating detailed explanation)"
            self.cache[anomaly_log.log_id] = fallback
            return fallback
    
    def _explain_with_anthropic(self, payload: dict, system_prompt: str) -> str:
        """Generate explanation using Anthropic Claude"""
        message = self.anthropic_client.messages.create(
            model=self.settings.LLM_MODEL,
            max_tokens=self.settings.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        return message.content[0].text.strip() if message.content else "No explanation generated."
    
    def _explain_with_openai(self, payload: dict, system_prompt: str) -> str:
        """Generate explanation using OpenAI GPT"""
        response = self.openai_client.chat.completions.create(
            model=self.settings.LLM_MODEL if self.settings.LLM_MODEL.startswith("gpt") else "gpt-4o-mini",
            max_tokens=self.settings.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)}
            ],
        )
        return response.choices[0].message.content.strip() if response.choices else "No explanation generated."
