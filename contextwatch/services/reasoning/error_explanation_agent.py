"""LangGraph-based agentic error explanation system.

Multi-step workflow for explaining detected anomalies:
1. Context Gathering: Fetch error details + similar historic errors
2. Signal Extraction: Extract domain signals from error patterns
3. Reasoning: Call Claude for structured RCA analysis
4. Remediation: Generate actionable fix suggestions

Example usage:
    agent = ErrorExplanationAgent()
    explanation = agent.explain_error(anomaly_id="log_123", error_type="TOOL_HALLUCINATION")
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from langgraph.graph import StateGraph
    from langgraph.graph import START, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

import anthropic

from contextwatch.services.detection.error_classifier import (
    RuleBasedErrorClassifier,
    ErrorSignalExtractor,
)


# ============================================================================
# 1. STATE DEFINITION
# ============================================================================


@dataclass
class ErrorExplanationState:
    """Shared state across reasoning nodes."""

    # Input/Output
    anomaly_id: str = ""
    error_type: str = ""  # TOOL_HALLUCINATION, CONTEXT_POISONING, etc.
    raw_log: Optional[Dict[str, Any]] = None

    # Node 1: Context gathering
    context_summary: str = ""
    similar_errors: List[Dict[str, Any]] = field(default_factory=list)

    # Node 2: Signal extraction
    extracted_signals: Dict[str, Any] = field(default_factory=dict)
    signal_confidence: float = 0.0

    # Node 3: Reasoning
    root_cause_analysis: str = ""
    confidence_level: str = "medium"  # low | medium | high

    # Node 4: Remediation
    remediation_steps: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    execution_trace: List[str] = field(default_factory=list)


# ============================================================================
# 2. ERROR EXPLANATION AGENT
# ============================================================================


class ErrorExplanationAgent:
    """Multi-node LangGraph agent for explaining detected errors."""

    # Error type → explanation template mapping
    ERROR_TYPE_PROMPTS = {
        "TOOL_HALLUCINATION": """The system attempted to call a tool that either:
1. Does not exist in the registry
2. Has an incompatible schema
3. Is being called with invalid parameters

Common causes: Poor indexing of tool availability, incorrect tool disambiguation, 
schema version mismatch, or incomplete dependency information.""",

        "CONTEXT_POISONING": """The system received malicious or misleading information in its context, 
leading to incorrect decisions. This could manifest as:
1. Injected instructions in retrieval results
2. Conflicting information sources
3. Tampered metadata or system prompts

Common causes: Inadequate input sanitization, untrusted retrieval sources, 
cache poisoning, or insufficient validation of external data.""",

        "REGISTRY_OVERFLOW": """The system's tool registry became too large or complex, 
causing decision-making degradation:
1. Too many similar tools (disambiguation failure)
2. Exponential search complexity
3. Hallucinated tool combinations

Common causes: Uncontrolled tool registration, insufficient taxonomy management, 
poor tool clustering, or missing tool aliasing strategies.""",

        "DELEGATION_CHAIN_FAILURE": """The system failed to complete a delegated task or maintain
state across a call chain:
1. Partial/incomplete outputs
2. Lost context between steps
3. Unhandled cascading failures

Common causes: Missing error handling, insufficient state management, 
incomplete function implementations, or circular dependencies.""",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        use_langgraph: bool = True,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        if self.api_key and self.api_key != "dummy":
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"⚠️ Failed to initialize Claude client: {e}")
                self.client = None
        self.model = model
        self.use_langgraph = use_langgraph and HAS_LANGGRAPH

        # Initialize classifiers for signal extraction
        self.signal_extractor = ErrorSignalExtractor()
        self.classifier = RuleBasedErrorClassifier()

        # Build LangGraph if available
        if self.use_langgraph:
            self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Construct the LangGraph workflow."""
        if not HAS_LANGGRAPH:
            return None

        workflow = StateGraph(ErrorExplanationState)

        # Add nodes
        workflow.add_node("gather_context", self.node_gather_context)
        workflow.add_node("extract_signals", self.node_extract_signals)
        workflow.add_node("reason", self.node_reason)
        workflow.add_node("generate_remediation", self.node_generate_remediation)

        # Define edges (linear flow)
        workflow.add_edge(START, "gather_context")
        workflow.add_edge("gather_context", "extract_signals")
        workflow.add_edge("extract_signals", "reason")
        workflow.add_edge("reason", "generate_remediation")
        workflow.add_edge("generate_remediation", END)

        return workflow.compile()

    # ========================================================================
    # NODE IMPLEMENTATIONS
    # ========================================================================

    def node_gather_context(self, state: ErrorExplanationState) -> ErrorExplanationState:
        """Node 1: Gather context about the error and similar past errors."""
        state.execution_trace.append("📋 Gathering context...")

        # Pseudo-implementation: In production, query PostgreSQL for similar errors
        # For now, provide structured context based on error type
        error_type = state.error_type or "UNKNOWN"
        template = self.ERROR_TYPE_PROMPTS.get(error_type, "Unknown error type.")

        state.context_summary = f"""
Error Type: {error_type}
Description: {template}

Log Details:
- Message: {state.raw_log.get('message', 'N/A') if state.raw_log else 'N/A'}
- Timestamp: {state.raw_log.get('timestamp', 'N/A') if state.raw_log else 'N/A'}
- Source: {state.raw_log.get('source', 'N/A') if state.raw_log else 'N/A'}
"""

        # Simulate fetching similar errors (would query DB in production)
        state.similar_errors = [
            {
                "timestamp": (datetime.utcnow()).isoformat(),
                "error_type": error_type,
                "similarity_score": 0.92,
                "message": f"Similar {error_type} detected",
            }
        ]

        state.execution_trace.append("✅ Context gathered")
        return state

    def node_extract_signals(self, state: ErrorExplanationState) -> ErrorExplanationState:
        """Node 2: Extract domain-specific signals from the error."""
        state.execution_trace.append("🔍 Extracting signals...")

        if not state.raw_log:
            state.execution_trace.append("⚠️ No raw log to extract signals from")
            return state

        # Use existing signal extractor
        try:
            signals = self.signal_extractor.extract_signals(state.raw_log)
            state.extracted_signals = {
                "tool_signals": signals.get("tool_signals", {}),
                "retrieval_signals": signals.get("retrieval_signals", {}),
                "registry_signals": signals.get("registry_signals", {}),
                "delegation_signals": signals.get("delegation_signals", {}),
            }
            state.signal_confidence = sum(
                v.get("score", 0) for v in state.extracted_signals.values() if isinstance(v, dict)
            ) / len(state.extracted_signals)
        except Exception as e:
            state.execution_trace.append(f"⚠️ Signal extraction failed: {str(e)}")
            state.extracted_signals = {}
            state.signal_confidence = 0.0

        state.execution_trace.append("✅ Signals extracted")
        return state

    def node_reason(self, state: ErrorExplanationState) -> ErrorExplanationState:
        """Node 3: Call Claude for structured reasoning about root cause."""
        state.execution_trace.append("🧠 Reasoning about root cause...")

        if not self.client:
            state.execution_trace.append("⚠️ Claude API not configured; skipping reasoning")
            state.root_cause_analysis = "Claude API key not available."
            return state

        # Build reasoning prompt
        error_context = self.ERROR_TYPE_PROMPTS.get(state.error_type, "Unknown error.")

        signals_summary = json.dumps(state.extracted_signals, indent=2)

        prompt = f"""You are an expert SRE analyzing a detected system error.

**Error Type**: {state.error_type}

**Error Context**:
{error_context}

**Extracted Signals**:
{signals_summary}

**Task**: Provide a concise root cause analysis in the following format:

1. **Immediate Cause** (1 sentence): What directly triggered the error?
2. **Predisposing Factors** (2-3 bullets): What system conditions enabled this?
3. **Signal Correlation** (1-2 sentences): How do the extracted signals support this analysis?
4. **Confidence**: Low/Medium/High - justify briefly

Keep the analysis technical but concise."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            state.root_cause_analysis = response.content[0].text

            # Simple heuristic for confidence
            if "high" in response.content[0].text.lower():
                state.confidence_level = "high"
            elif "low" in response.content[0].text.lower():
                state.confidence_level = "low"
            else:
                state.confidence_level = "medium"

        except Exception as e:
            state.execution_trace.append(f"⚠️ Claude API error: {str(e)}")
            state.root_cause_analysis = f"Error during reasoning: {str(e)}"
            state.confidence_level = "low"

        state.execution_trace.append("✅ Reasoning complete")
        return state

    def node_generate_remediation(self, state: ErrorExplanationState) -> ErrorExplanationState:
        """Node 4: Generate actionable remediation and prevention steps."""
        state.execution_trace.append("🛠️ Generating remediation steps...")

        if not self.client:
            state.execution_trace.append("⚠️ Claude API not configured; using heuristic remediations")
            state.remediation_steps = self._get_heuristic_remediations(state.error_type)
            return state

        remediation_prompt = f"""Based on this error analysis, provide immediate remediation and long-term prevention steps.

**Error Type**: {state.error_type}
**Root Cause Analysis**:
{state.root_cause_analysis}

**Task**: Provide in this exact format:

**Immediate Actions** (3-5 steps for operators):
- Step 1
- Step 2
- etc.

**Prevention Measures** (2-3 systemic improvements):
- Measure 1
- Measure 2
- etc.

Be specific and actionable."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": remediation_prompt}],
            )

            # Parse remediation steps from response
            text = response.content[0].text
            lines = text.split("\n")

            immediate_actions = []
            prevention_measures = []
            current_section = None

            for line in lines:
                line = line.strip()
                if "immediate" in line.lower():
                    current_section = "immediate"
                elif "prevention" in line.lower():
                    current_section = "prevention"
                elif line.startswith("- ") and current_section == "immediate":
                    immediate_actions.append(line[2:])
                elif line.startswith("- ") and current_section == "prevention":
                    prevention_measures.append(line[2:])

            state.remediation_steps = immediate_actions or self._get_heuristic_remediations(
                state.error_type
            )
            state.prevention_measures = (
                prevention_measures
                or [
                    "Improve error taxonomy and signal detection",
                    "Enhance monitoring and alerting",
                ]
            )

        except Exception as e:
            state.execution_trace.append(f"⚠️ Remediation generation error: {str(e)}")
            state.remediation_steps = self._get_heuristic_remediations(state.error_type)
            state.prevention_measures = [
                "Improve error taxonomy and signal detection",
                "Enhance monitoring and alerting",
            ]

        state.execution_trace.append("✅ Remediation steps generated")
        return state

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def explain_error(
        self,
        anomaly_id: str,
        error_type: str,
        raw_log: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point: Explain a detected error through the workflow.

        Args:
            anomaly_id: Unique identifier for the anomaly
            error_type: 'TOOL_HALLUCINATION', 'CONTEXT_POISONING', etc.
            raw_log: Raw log entry dict with structure: {message, timestamp, source, pattern, etc.}

        Returns:
            Dictionary with:
            {
                "anomaly_id": str,
                "error_type": str,
                "root_cause_analysis": str,
                "remediation_steps": list[str],
                "prevention_measures": list[str],
                "confidence_level": str,
                "execution_trace": list[str],
                "created_at": str,
            }
        """
        # Initialize state
        state = ErrorExplanationState(
            anomaly_id=anomaly_id,
            error_type=error_type,
            raw_log=raw_log or {},
        )

        # Execute workflow
        if self.use_langgraph and self.workflow:
            # Use LangGraph compiled workflow
            try:
                final_state_dict = self.workflow.invoke(state)
                # Extract values from LangGraph's AddableValuesDict
                state = self._dict_to_state(final_state_dict)
            except Exception as e:
                state.execution_trace.append(f"❌ LangGraph execution failed: {str(e)}")
                # Fallback to sequential node execution
                state = self._execute_fallback(state)
        else:
            # Fallback: sequential node execution
            state = self._execute_fallback(state)

        # Format output
        return {
            "anomaly_id": state.anomaly_id,
            "error_type": state.error_type,
            "root_cause_analysis": state.root_cause_analysis,
            "remediation_steps": state.remediation_steps,
            "prevention_measures": state.prevention_measures,
            "confidence_level": state.confidence_level,
            "signal_confidence": round(state.signal_confidence, 2),
            "execution_trace": state.execution_trace,
            "created_at": state.created_at,
        }

    def _execute_fallback(self, state: ErrorExplanationState) -> ErrorExplanationState:
        """Fallback: Execute nodes sequentially without LangGraph."""
        state = self.node_gather_context(state)
        state = self.node_extract_signals(state)
        state = self.node_reason(state)
        state = self.node_generate_remediation(state)
        return state

    @staticmethod
    def _dict_to_state(state_dict: Dict[str, Any]) -> ErrorExplanationState:
        """Convert LangGraph's AddableValuesDict to ErrorExplanationState."""
        return ErrorExplanationState(
            anomaly_id=state_dict.get("anomaly_id", ""),
            error_type=state_dict.get("error_type", ""),
            raw_log=state_dict.get("raw_log"),
            context_summary=state_dict.get("context_summary", ""),
            similar_errors=state_dict.get("similar_errors", []),
            extracted_signals=state_dict.get("extracted_signals", {}),
            signal_confidence=state_dict.get("signal_confidence", 0.0),
            root_cause_analysis=state_dict.get("root_cause_analysis", ""),
            confidence_level=state_dict.get("confidence_level", "medium"),
            remediation_steps=state_dict.get("remediation_steps", []),
            prevention_measures=state_dict.get("prevention_measures", []),
            created_at=state_dict.get("created_at", datetime.utcnow().isoformat()),
            execution_trace=state_dict.get("execution_trace", []),
        )

    @staticmethod
    def _get_heuristic_remediations(error_type: str) -> List[str]:
        """Heuristic remediation steps when Claude API unavailable."""
        remediations = {
            "TOOL_HALLUCINATION": [
                "1. Verify tool registry completeness and accuracy",
                "2. Review and update tool availability indices",
                "3. Check schema definitions for all registered tools",
                "4. Implement tool existence validation before invocation",
                "5. Add fallback mechanism for unknown tool requests",
            ],
            "CONTEXT_POISONING": [
                "1. Sanitize all external data inputs (retrieval results)",
                "2. Validate context sources and trust scores",
                "3. Implement content hash verification for critical data",
                "4. Add anomaly detection on unusual context patterns",
                "5. Review access controls for context/metadata systems",
            ],
            "REGISTRY_OVERFLOW": [
                "1. Audit tool registry for duplicates and dead entries",
                "2. Implement tool taxonomy/categorization",
                "3. Add tool aliasing and grouping strategies",
                "4. Optimize tool search and disambiguation",
                "5. Set limits on registry growth and complexity",
            ],
            "DELEGATION_CHAIN_FAILURE": [
                "1. Add comprehensive error handling in delegation chains",
                "2. Implement state checkpointing between steps",
                "3. Review task completion criteria and validation",
                "4. Add timeout and retry mechanisms",
                "5. Improve logging and tracing for chain execution",
            ],
        }
        return remediations.get(error_type, [
            "1. Investigate error type classification",
            "2. Review system logs for context",
            "3. Check for recent system changes",
            "4. Validate data and configuration",
            "5. Consider escalation to operations team",
        ])


# ============================================================================
# TESTING & UTILITIES
# ============================================================================


def test_explain_error():
    """Test the error explanation agent."""
    agent = ErrorExplanationAgent()

    # Sample error logs
    test_cases = [
        {
            "anomaly_id": "log_001",
            "error_type": "TOOL_HALLUCINATION",
            "raw_log": {
                "message": "Attempted to call undefined tool: weather_realtime_v999",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "agent_orchestrator",
                "pattern": {
                    "tools_called": ["weather_realtime_v999"],
                    "registry_size": 42,
                    "tool_entropy": 3.2,
                },
            },
        },
        {
            "anomaly_id": "log_002",
            "error_type": "CONTEXT_POISONING",
            "raw_log": {
                "message": "Suspicious instruction injection detected in retrieval results",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "rag_validator",
                "pattern": {
                    "injection_score": 0.87,
                    "schema_violations": 2,
                    "trust_degradation": 0.45,
                },
            },
        },
    ]

    for test_case in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {test_case['error_type']}")
        print(f"{'='*70}")

        result = agent.explain_error(
            anomaly_id=test_case["anomaly_id"],
            error_type=test_case["error_type"],
            raw_log=test_case["raw_log"],
        )

        print(f"\n📊 Execution Trace:")
        for step in result["execution_trace"]:
            print(f"  {step}")

        print(f"\n📋 Root Cause Analysis:")
        print(result["root_cause_analysis"])

        print(f"\n🛠️ Remediation Steps:")
        for step in result["remediation_steps"]:
            print(f"  • {step}")

        print(f"\n🛡️ Prevention Measures:")
        for measure in result["prevention_measures"]:
            print(f"  • {measure}")

        print(f"\n📈 Confidence: {result['confidence_level']}")


if __name__ == "__main__":
    test_explain_error()
