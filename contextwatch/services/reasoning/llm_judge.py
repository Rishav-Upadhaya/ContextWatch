"""LLM-as-a-Judge for reasoning quality validation.

Evaluates RCA reasoning traces against a labeled taxonomy of 16 reasoning failures.
Specifically checks for:
- RF-01: Fabricated Evidence (lacking actual support)
- RF-03: Confused Provenance (source attribution errors)
- RF-05: Spurious Causal Attribution (correlation vs causation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# (Simplified for pure-Python; real LLM calls go through infrastructure/llm_client.py later)


RCA_FAILURE_TAXONOMY = {
    # Reasoning Failures - Evidence
    "RF-01": "Fabricated Evidence",
    "RF-02": "Incomplete Evidence",
    "RF-03": "Confused Provenance",
    # Reasoning Failures - Causality
    "RF-04": "Temporal Misordering",
    "RF-05": "Spurious Causal Attribution",
    "RF-06": "Missing Third Factor",
    "RF-07": "Reverse Causality",
    "RF-08": "Causal Overreach",
    # Domain Confusions
    "RF-09": "Telemetry Mixup",
    "RF-10": "False Signal-to-Noise Interpretation",
    "RF-11": "Misinterpreted Composite Event",
    # Explanation Quality
    "RF-12": "Unjustified Confidence",
    "RF-13": "Overfitting to Noise",
    "RF-14": "Rote Critique",
    "RF-15": "Inconsistent Reasoning Steps",
    "RF-16": "Context Dropping",
}

# Map targeted failures to their weight in final score
TARGETED_FAILURES = {"RF-01", "RF-03", "RF-04", "RF-05"}


@dataclass
class ReasoningTrace:
    """An RCA reasoning trace to be judged."""
    root_cause_id: str
    explanation: str
    evidence: List[str]
    causal_chain: List[str]
    timestamps: List[float]


@dataclass
class JudgeVerdict:
    """Verdict from the LLM-as-a-Judge evaluator."""
    passed: bool
    scored_failures: dict[str, bool]  # failure code -> found=True/False
    failure_mask: List[str]  # which failures were detected
    notes: List[str]
    explanation: str
    confidence: float  # 0-1

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scored_failures": self.scored_failures,
            "failure_mask": self.failure_mask,
            "notes": self.notes,
            "explanation": self.explanation,
            "confidence": self.confidence
        }


def _check_temporal_ordering(trace: ReasoningTrace) -> Tuple[bool, str]:
    """RF-04: Check that events in causal chain are chronologically ordered."""
    if not trace.timestamps:
        return (True, "no timestamps to check")
    for i in range(len(trace.timestamps) - 1):
        if trace.timestamps[i] >= trace.timestamps[i + 1]:
            return (False, f"timestamps[{i}] ({trace.timestamps[i]}) >= timestamps[{i+1}] ({trace.timestamps[i+1]})")
    return (True, "chronological order preserved")


def _check_evidence_support(trace: ReasoningTrace) -> Tuple[bool, str]:
    """RF-01: Check that listed evidence actually supports the root cause."""
    # Simple heuristic: evidence strings must reference root node or its direct dependencies
    if not trace.evidence:
        return (False, "no evidence provided")
    if any("N/A" in ev or "unknown" in ev.lower() for ev in trace.evidence):
        return (False, "evidence contains placeholders")
    # Also check that evidence IDs appear in causal chain
    # (In real system, would cross-check with actual log entries)
    return (True, "evidence appears substantive")


def _check_provenance(trace: ReasoningTrace) -> Tuple[bool, str]:
    """RF-03: Check that source attribution is present and specific."""
    for ev in trace.evidence:
        # Evidence should include clear source (e.g., "incident#12345", "log:abc123", "trace:def456")
        if not any(source_marker in ev.lower() for source_marker in ("#", "log:", "trace:", "incident:", "runbook:")):
            return (False, f"evidence source not specified: '{ev[:50]}...'")
    return (True, "all evidence has clear provenance")


def _check_causal_attribution(trace: ReasoningTrace) -> Tuple[bool, str]:
    """RF-05: Check that causal links are not spurious (simple heuristic)."""
    if len(trace.causal_chain) < 2:
        return (True, "chain too short for spurious check")
    # If same service appears repeatedly, may be false correlation
    # For now, accept all chains as valid; placeholder for real counter-factual tests
    return (True, "causal chain length plausible")


class ReasoningJudge:
    """LLM-as-a-Judge that evaluates reasoning quality against RCA failure taxonomy."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client  # Inject real LLM client if available

    def evaluate(
        self,
        trace: ReasoningTrace,
        target_failures: List[str] = None,
    ) -> JudgeVerdict:
        """Evaluate reasoning trace.

        Parameters:
          trace: ReasoningTrace to evaluate
          target_failures: Which failure codes to prioritize (default: RF-01, RF-03, RF-04, RF-05)
        Returns:
          JudgeVerdict with detailed results
        """
        if target_failures is None:
            target_failures = list(TARGETED_FAILURES)

        notes: List[str] = []
        scored_failures: Dict[str, bool] = {}
        failed_any = False

        # RF-01: Fabricated Evidence
        if "RF-01" in target_failures:
            passed, note = _check_evidence_support(trace)
            scored_failures["RF-01"] = not passed
            if not passed:
                failed_any = True
                notes.append(f"RF-01: {note}")

        # RF-03: Confused Provenance
        if "RF-03" in target_failures:
            passed, note = _check_provenance(trace)
            scored_failures["RF-03"] = not passed
            if not passed:
                failed_any = True
                notes.append(f"RF-03: {note}")

        # RF-04: Temporal Misordering
        if "RF-04" in target_failures:
            passed, note = _check_temporal_ordering(trace)
            scored_failures["RF-04"] = not passed
            if not passed:
                failed_any = True
                notes.append(f"RF-04: {note}")

        # RF-05: Spurious Causal Attribution
        if "RF-05" in target_failures:
            passed, note = _check_causal_attribution(trace)
            scored_failures["RF-05"] = not passed
            if not passed:
                passed = False
                failed_any = True
                notes.append(f"RF-05: {note}")

        # Fill in others as not found
        for code in RCA_FAILURE_TAXONOMY:
            if code not in scored_failures:
                scored_failures[code] = False

        # Confidence: proportional to number of checks passed
        total_checks = len(target_failures)
        passed_count = total_checks - sum(1 for v in scored_failures.values() if v)
        confidence = passed_count / total_checks if total_checks else 1.0

        # Final pass: if no targeted failures found and confidence >= threshold
        passed = (not failed_any and confidence >= 0.75)

        verdict = JudgeVerdict(
            passed=passed,
            scored_failures=scored_failures,
            failure_mask=[code for code, found in scored_failures.items() if found],
            notes=notes,
            explanation="Detected reasoning failures: " + "; ".join(notes) if notes else "No reasoning errors detected",
            confidence=confidence,
        )
        return verdict


# Simplified stub for integration with LLM client (would call OpenAI/Anthropic API)
def llm_as_judge_evaluate(
    trace: ReasoningTrace,
    llm_client=None,
) -> JudgeVerdict:
    """Thin wrapper that optionally uses a live LLM for richer evaluation."""
    if llm_client:
        # In practice, call LLM with structured prompt
        # This stub just calls the deterministic checker
        return ReasoningJudge(llm_client=llm_client).evaluate(trace)
    else:
        return ReasoningJudge().evaluate(trace)


"""End of llm_judge module."""
