"""
Rule-based error type classifier using domain signal extraction.

Uses heuristics on log fields to classify into:
  - TOOL_HALLUCINATION: Unknown tools, schema violations, irrelevant context
  - CONTEXT_POISONING: Malicious instructions in retrieval, output deviation
  - REGISTRY_OVERFLOW: Too many tools, tool confusion, high entropy
  - DELEGATION_CHAIN_FAILURE: Broken pipeline, missing fields, loops
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorSignalExtractor:
    """Extract domain signals from a log payload for error classification."""
    
    @staticmethod
    def extract_signals(log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured signals from log for classification."""
        signals = {
            "tool_signals": ErrorSignalExtractor._extract_tool_signals(log),
            "retrieval_signals": ErrorSignalExtractor._extract_retrieval_signals(log),
            "registry_signals": ErrorSignalExtractor._extract_registry_signals(log),
            "delegation_signals": ErrorSignalExtractor._extract_delegation_signals(log),
        }
        return signals
    
    @staticmethod
    def _extract_tool_signals(log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool-related signals."""
        signals = {
            "unknown_tool": False,
            "schema_violation": False,
            "irrelevant_context": False,
            "no_justification": False,
        }
        
        try:
            data = log.get("params", {}).get("data", {})
            pattern = data.get("pattern", {})
            
            # Check for unknown tool
            if "tool_call" in pattern:
                tool_call = pattern["tool_call"]
                tool_name = tool_call.get("name", "")
                lookup_status = pattern.get("tool_registry", {}).get("lookup_status", "")
                
                if lookup_status == "MISS" or "not found" in str(pattern.get("error", "")):
                    signals["unknown_tool"] = True
            
            # Check for schema violation
            if "schema_validation" in pattern:
                if pattern["schema_validation"].get("status") == "FAIL":
                    signals["schema_violation"] = True
            
            # Check for irrelevant context
            if "tool_call" in pattern and "user_query" in pattern:
                query = pattern.get("user_query", "").lower()
                tool = pattern.get("tool_call", {}).get("name", "").lower()
                reasoning = pattern.get("reasoning_trace", "").lower()
                
                # Heuristic: if tool is unrelated to query and reasoning is weak
                if reasoning == "no clear justification for tool selection" or "no clear" in reasoning:
                    signals["irrelevant_context"] = True
            
            # Check for no justification
            reasoning_obj = pattern.get("reasoning_trace", {})
            if isinstance(reasoning_obj, dict):
                if reasoning_obj.get("tool_justification") is None:
                    signals["no_justification"] = True
        
        except Exception as e:
            logger.debug(f"Error extracting tool signals: {e}")
        
        return signals
    
    @staticmethod
    def _extract_retrieval_signals(log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract retrieval/context-related signals."""
        signals = {
            "malicious_instruction": False,
            "output_deviation": False,
            "high_chunk_attribution": False,
        }
        
        try:
            data = log.get("params", {}).get("data", {})
            pattern = data.get("pattern", {})
            
            # Check for malicious instructions
            if "retrieval" in pattern:
                chunks = pattern["retrieval"].get("top_k_chunks", [])
                for chunk in chunks:
                    chunk_lower = str(chunk).lower()
                    if any(keyword in chunk_lower for keyword in [
                        "ignore", "override", "bypass", "secret", "leak", "malicious", "harmful"
                    ]):
                        signals["malicious_instruction"] = True
                        break
            
            # Check for output deviation
            if "actual_output" in pattern and "system_prompt" in pattern:
                system_prompt = pattern.get("system_prompt", "").lower()
                actual = pattern.get("actual_output", "").lower()
                
                if "safety" in system_prompt and "bypass" in actual:
                    signals["output_deviation"] = True
                elif pattern.get("alignment_score", 1.0) < 0.5:
                    signals["output_deviation"] = True
            
            # Check for safety flagging
            if "safety_filter" in pattern:
                if pattern["safety_filter"].get("flagged"):
                    signals["malicious_instruction"] = True
            
            # Check high attribution to single chunk
            if "output_tokens_from_chunk" in pattern:
                if pattern["output_tokens_from_chunk"] > 0.8:
                    signals["high_chunk_attribution"] = True
        
        except Exception as e:
            logger.debug(f"Error extracting retrieval signals: {e}")
        
        return signals
    
    @staticmethod
    def _extract_registry_signals(log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract registry/tool proliferation signals."""
        signals = {
            "large_registry": False,
            "tool_confusion": False,
            "high_entropy": False,
        }
        
        try:
            data = log.get("params", {}).get("data", {})
            pattern = data.get("pattern", {})
            
            # Check registry size
            registry_size = pattern.get("registry_size", 0)
            if registry_size > 50:
                signals["large_registry"] = True
            
            # Check tool selection history (oscillation)
            if "tool_selection_history" in pattern:
                history = pattern["tool_selection_history"]
                if len(history) >= 3:
                    # Check for repeated switching
                    same_count = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
                    if same_count >= 2:
                        signals["tool_confusion"] = True
            
            # Check logit entropy
            if "logit_entropy" in pattern:
                entropy = pattern["logit_entropy"]
                if entropy > 3.0:  # High entropy = uniform distribution = confusion
                    signals["high_entropy"] = True
            
            # Check selection confidence
            if "selection_confidence" in pattern:
                if pattern["selection_confidence"] < 0.3:
                    signals["high_entropy"] = True
            
            # Check tool similarity
            if "similar_tools" in pattern:
                similar_count = len(pattern.get("similar_tools", []))
                if similar_count >= 3:
                    signals["tool_confusion"] = True
        
        except Exception as e:
            logger.debug(f"Error extracting registry signals: {e}")
        
        return signals
    
    @staticmethod
    def _extract_delegation_signals(log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract delegation/pipeline signals."""
        signals = {
            "partial_output": False,
            "missing_fields": False,
            "silent_failure": False,
            "loop_detected": False,
        }
        
        try:
            data = log.get("params", {}).get("data", {})
            pattern = data.get("pattern", {})
            
            # Check for partial/invalid output
            if "step_1" in pattern:
                step1_output = pattern["step_1"].get("output", {})
                step1_status = pattern["step_1"].get("status", "")
                
                # Partial output (incomplete JSON, etc.)
                if isinstance(step1_output, str):
                    if not step1_output.endswith("}") and "{" in step1_output:
                        signals["partial_output"] = True
                
                if "parse" in str(pattern.get("error", "")).lower():
                    signals["partial_output"] = True
            
            # Check for missing fields
            if "step_2" in pattern:
                expected = pattern["step_2"].get("expected_input", {})
                if isinstance(expected, dict):
                    if any(v == "???" for v in expected.values()):
                        signals["missing_fields"] = True
                
                if "missing required field" in str(pattern.get("error", "")).lower():
                    signals["missing_fields"] = True
            
            # Check for silent failures
            if "step_1" in pattern and "step_2" in pattern:
                step1_out = pattern["step_1"].get("output")
                step1_err = pattern["step_1"].get("error")
                step2_status = pattern["step_2"].get("status", "")
                
                if step1_out is None and step1_err is None and "degraded" in str(step2_status).lower():
                    signals["silent_failure"] = True
            
            # Check for loops
            if "chain" in pattern:
                chain = pattern["chain"]
                agents_visited = [step.get("agent") for step in chain]
                if len(agents_visited) > 2:
                    # Detect cycle: A -> B -> A
                    if agents_visited[-1] == agents_visited[-3]:
                        signals["loop_detected"] = True
                
                if "infinite_loop" in str(pattern.get("status", "")).lower():
                    signals["loop_detected"] = True
        
        except Exception as e:
            logger.debug(f"Error extracting delegation signals: {e}")
        
        return signals


class RuleBasedErrorClassifier:
    """Classify logs into 4 error types using domain rules."""
    
    CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence to make a prediction
    
    @staticmethod
    def classify(log: Dict[str, Any]) -> Tuple[Optional[str], float, str]:
        """
        Classify log into error type.
        
        Returns: (error_type, confidence, reasoning)
        """
        signals = ErrorSignalExtractor.extract_signals(log)
        
        # Check for explicit anomaly type in log
        explicit_type = log.get("anomaly_type")
        if explicit_type and explicit_type in ["TOOL_HALLUCINATION", "CONTEXT_POISONING", "REGISTRY_OVERFLOW", "DELEGATION_CHAIN_FAILURE"]:
            return explicit_type, 0.98, f"Explicit type: {explicit_type}"
        
        # Score each error type
        scores = {
            "TOOL_HALLUCINATION": RuleBasedErrorClassifier._score_tool_hallucination(signals),
            "CONTEXT_POISONING": RuleBasedErrorClassifier._score_context_poisoning(signals),
            "REGISTRY_OVERFLOW": RuleBasedErrorClassifier._score_registry_overflow(signals),
            "DELEGATION_CHAIN_FAILURE": RuleBasedErrorClassifier._score_delegation_failure(signals),
        }
        
        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score < RuleBasedErrorClassifier.CONFIDENCE_THRESHOLD:
            return None, best_score, f"Low confidence: {best_type}={best_score:.2f}"
        
        return best_type, best_score, f"Best match: {best_type}={best_score:.2f}"
    
    @staticmethod
    def _score_tool_hallucination(signals: Dict[str, Any]) -> float:
        """Score likelihood of TOOL_HALLUCINATION."""
        tool_sig = signals.get("tool_signals", {})
        score = 0.0
        
        if tool_sig.get("unknown_tool"):
            score += 0.5
        if tool_sig.get("schema_violation"):
            score += 0.3
        if tool_sig.get("irrelevant_context"):
            score += 0.2
        if tool_sig.get("no_justification"):
            score += 0.15
        
        return min(score, 1.0)
    
    @staticmethod
    def _score_context_poisoning(signals: Dict[str, Any]) -> float:
        """Score likelihood of CONTEXT_POISONING."""
        retrieval_sig = signals.get("retrieval_signals", {})
        score = 0.0
        
        if retrieval_sig.get("malicious_instruction"):
            score += 0.6
        if retrieval_sig.get("output_deviation"):
            score += 0.3
        if retrieval_sig.get("high_chunk_attribution"):
            score += 0.15
        
        return min(score, 1.0)
    
    @staticmethod
    def _score_registry_overflow(signals: Dict[str, Any]) -> float:
        """Score likelihood of REGISTRY_OVERFLOW."""
        registry_sig = signals.get("registry_signals", {})
        score = 0.0
        
        if registry_sig.get("large_registry"):
            score += 0.4
        if registry_sig.get("tool_confusion"):
            score += 0.4
        if registry_sig.get("high_entropy"):
            score += 0.3
        
        return min(score, 1.0)
    
    @staticmethod
    def _score_delegation_failure(signals: Dict[str, Any]) -> float:
        """Score likelihood of DELEGATION_CHAIN_FAILURE."""
        delegation_sig = signals.get("delegation_signals", {})
        score = 0.0
        
        if delegation_sig.get("partial_output"):
            score += 0.4
        if delegation_sig.get("missing_fields"):
            score += 0.4
        if delegation_sig.get("silent_failure"):
            score += 0.3
        if delegation_sig.get("loop_detected"):
            score += 0.5
        
        return min(score, 1.0)


if __name__ == "__main__":
    # Example usage
    test_log_hallucination = {
        "log_id": "test-1",
        "params": {
            "data": {
                "pattern": {
                    "tool_call": {"name": "sendEmail_v2", "args": {}},
                    "tool_registry": {"lookup_status": "MISS"},
                    "error": "Tool sendEmail_v2 not found",
                }
            }
        }
    }
    
    error_type, conf, reason = RuleBasedErrorClassifier.classify(test_log_hallucination)
    print(f"Type: {error_type}, Confidence: {conf:.2f}, Reason: {reason}")
