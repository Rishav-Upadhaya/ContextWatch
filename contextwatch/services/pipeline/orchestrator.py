"""Enterprise Orchestrator: Complete ContextWatch pipeline from anomaly to RCA report.

Orchestrates:
1. Normalization → protocol plumbing removal
2. LogBERT encoding → VHM anomaly scoring
3. Multi-agent RCA → graph propagation
4. LLM-as-Judge validation
5. PDF report generation
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from contextwatch.services.detection.anomaly_detector import AnomalyDetector, DetectionMethod
from contextwatch.services.graph.graph_propagation import GraphPropagation
from contextwatch.services.reasoning.llm_judge import ReasoningJudge, ReasoningTrace
from contextwatch.services.ai.logbert_inference import LogBERTInference
from contextwatch.services.reasoning.mar_cra import MARCRA, MARCATrace
from contextwatch.services.reasoning.report_generator import generate_full_report
from contextwatch.services.detection.signal_filter import filter_log_signal
from contextwatch.services.detection.vhm_core import VHMEngine
from contextwatch.utils.tokenizer import LogTokenizer
from contextwatch.utils.config import Settings
from contextwatch.adapters.vector_store import VectorStore
from contextwatch.services.pipeline.vhm_bootstrap import bootstrap_vhm

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from the full pipeline."""
    anomaly_id: str
    is_anomalous: bool
    anomaly_score: float
    anomaly_confidence: float
    method: str
    rca_trace: Optional[MARCATrace] = None
    judge_verdict: Optional[dict] = None
    report_files: Optional[Dict[str, str]] = None
    execution_time_ms: float = 0.0
    errors: List[str] = None

    def to_dict(self) -> dict:
        result = {
            "anomaly_id": self.anomaly_id,
            "is_anomalous": self.is_anomalous,
            "anomaly_score": self.anomaly_score,
            "anomaly_confidence": self.anomaly_confidence,
            "method": self.method,
            "execution_time_ms": self.execution_time_ms,
        }
        if self.rca_trace:
            result["rca_trace"] = self.rca_trace.to_dict()
        if self.judge_verdict:
            result["judge_verdict"] = self.judge_verdict
        if self.report_files:
            result["report_files"] = self.report_files
        if self.errors:
            result["errors"] = self.errors
        return result



class ContextWatchOrchestrator:
    """Enterprise-grade log anomaly detection and RCA orchestrator.

    Pipeline:
    1. **Normalization** → Clean log signal, remove protocol overhead
    2. **Embedding** → LogBERT semantic encoding
    3. **Detection** → VHM-based anomaly scoring
    4. **RCA** → Multi-agent root cause analysis with graph propagation
    5. **Validation** → LLM-as-Judge quality assurance
    6. **Reporting** → PDF compliance-ready report generation
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        vector_store: Optional[VectorStore] = None,
        knowledge_graph: Optional[Any] = None,
    ):
        self.settings = settings or Settings()
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

        # Initialize components
        self.tokenizer = LogTokenizer()
        self.anomaly_detector = AnomalyDetector(
            method=DetectionMethod.ENSEMBLE,  # Use ensemble for robustness
        )
        
        # Create VHMEngine for calibration
        # Default dimension is 1024 (matching LogBERT d_model)
        self.vhm_engine = VHMEngine(metric="cosine", dimensions=1024)
        self.vhm_engine.decision_scale = 3.0  # Level 1: Immediate buffer fix
        self.vhm_engine._refresh_public_state()

        # LogBERT model
        self.logbert_inference: Optional[LogBERTInference] = None
        self.logbert_version = "v1"

        # RCA engines
        self.marcra: Optional[MARCRA] = None
        self.graph_propagation: Optional[GraphPropagation] = None

        # Judge
        self.judge = ReasoningJudge()

    def initialize(self) -> bool:
        """Initialize all components.
        
        Returns:
            True if successful, False if critical init fails
        """
        try:
            # Load or train LogBERT
            from contextwatch.utils.weight_manager import WeightManager, get_weight_manager
            from contextwatch.services.ai.logbert_inference import load_or_train_model

            weight_manager = get_weight_manager(self.logbert_version)
            if weight_manager.has_weights():
                logger.info(f"✓ Loading LogBERT {self.logbert_version}")
                self.logbert_inference = LogBERTInference(weight_manager)
            else:
                logger.warning("LogBERT weights not found. Will train on first batch.")

            # Initialize RCA if graph available
            if self.knowledge_graph is not None:
                self.marcra = MARCRA(
                    knowledge_graph=self.knowledge_graph,
                    vector_store=self.vector_store,
                )
                self.graph_propagation = GraphPropagation(
                    knowledge_graph=self.knowledge_graph,
                )
                logger.info("✓ MA-RCA and graph propagation initialized")

            # Update detector with LogBERT
            if self.logbert_inference:
                self.anomaly_detector.logbert_inference = self.logbert_inference

            # Level 2 & 3: Calibrate VHM with golden dataset for F1-optimal thresholds
            logger.info("💡 Attempting VHM calibration with golden dataset...")
            calibration_result = self._calibrate_vhm_from_golden_dataset()
            if calibration_result:
                logger.info(
                    f"✓ VHM calibrated: F1={calibration_result.get('best_f1', 0):.4f}, "
                    f"Scale={calibration_result.get('decision_scale', 3.0):.4f}"
                )
            else:
                logger.warning(
                    "⚠️  VHM calibration failed or skipped. Using fallback scale=3.0"
                )

            logger.info("✓ ContextWatch orchestrator initialized")
            return True

        except (ImportError, ValueError, RuntimeError) as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Initialization unexpected failure: {e}", exc_info=True)
            return False

    def _load_golden_calibration_data(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load normal and anomalous logs from golden dataset JSONL files.
        
        Returns:
            Tuple of (normal_logs, anomaly_logs) lists
        """
        golden_dir = Path(__file__).parent.parent.parent / "data" / "golden_dataset"
        normal_file = golden_dir / "golden_normal.jsonl"
        anomaly_file = golden_dir / "golden_anomalies.jsonl"

        normal_logs = []
        anomaly_logs = []

        try:
            if normal_file.exists():
                with open(normal_file, "r") as f:
                    for line in f:
                        if line.strip():
                            normal_logs.append(json.loads(line))
            else:
                logger.warning(f"Golden normal dataset not found: {normal_file}")

            if anomaly_file.exists():
                with open(anomaly_file, "r") as f:
                    for line in f:
                        if line.strip():
                            anomaly_logs.append(json.loads(line))
            else:
                logger.warning(f"Golden anomaly dataset not found: {anomaly_file}")

            logger.info(
                f"✓ Loaded golden datasets: {len(normal_logs)} normal, {len(anomaly_logs)} anomalies"
            )
            return normal_logs, anomaly_logs

        except Exception as e:
            logger.error(f"Failed to load golden datasets: {e}", exc_info=True)
            return [], []

    def _calibrate_vhm_from_golden_dataset(self) -> Optional[Dict[str, Any]]:
        """Calibrate VHM using golden dataset for production-grade thresholds.
        
        This implements Level 2 & 3 of the VHM calibration strategy:
        - Level 2: Call calibrate() after fit() with labeled baseline sets
        - Level 3: Validate calibration and fallback if F1 is poor
        
        Returns:
            Calibration result dict with decision_scale, precision, recall, F1
            Returns None if calibration fails or is skipped
        """
        if not self.logbert_inference:
            logger.warning("LogBERT not initialized; skipping VHM calibration")
            return None

        if not self.vhm_engine:
            logger.warning("VHM engine not available; skipping calibration")
            return None

        # Load golden datasets
        normal_logs, anomaly_logs = self._load_golden_calibration_data()

        if len(normal_logs) < 10 or len(anomaly_logs) < 5:
            logger.warning(
                f"Insufficient golden data for calibration. "
                f"Normal={len(normal_logs)}, Anomaly={len(anomaly_logs)}. "
                f"Need at least 10 normal and 5 anomalies. Using fallback scale=3.0."
            )
            return None

        try:
            # Use bootstrap_vhm which handles encoding, fit, calibrate, and validation
            result = bootstrap_vhm(
                vhm=self.vhm_engine,
                logbert=self.logbert_inference,
                normal_logs=normal_logs,
                anomaly_logs=anomaly_logs,
                tokenizer=self.tokenizer,  # Pass tokenizer
                min_samples=10,
            )
            
            # Sync calibrated VHM parameters into logbert_inference for ensemble detection
            if result and "best_f1" in result and result["best_f1"] >= 0.5:
                if hasattr(self.vhm_engine, 'decision_radius') and self.logbert_inference:
                    # Use decision_radius (calibrated) not radius (pre-calibration)
                    self.logbert_inference.vhm_radius = self.vhm_engine.decision_radius
                    self.logbert_inference.vhm_center = np.array(self.vhm_engine.center, dtype=np.float32)
                    logger.info(
                        f"🔄 Synced calibrated VHM to LogBERT: "
                        f"center_dim={len(self.logbert_inference.vhm_center)}, "
                        f"radius={self.logbert_inference.vhm_radius:.6f} (decision_radius), "
                        f"scale={self.vhm_engine.decision_scale:.4f}"
                    )
            
            return result

        except Exception as e:
            logger.error(
                f"VHM calibration failed with exception: {e}",
                exc_info=True,
            )
            return None

    def process_log(
        self,
        log_id: str,
        log_dict: Dict[str, Any],
        protocol: str = "mcp",
    ) -> PipelineResult:
        """Process a single log event through the full pipeline.
        
        Args:
            log_id: Unique log identifier
            log_dict: Log event dictionary
            protocol: Protocol type ("mcp" or "a2a")
            
        Returns:
            PipelineResult with anomaly score, RCA, and report
        """
        start_time = datetime.utcnow()
        errors = []

        try:
            # Step 1: Normalize & filter signal
            logger.debug(f"Processing {log_id}: Normalization...")
            filtered = filter_log_signal(
                log_dict,
                protocol=protocol.upper(),
            )
            signal_text = filtered.signal_text

            # Step 2: Tokenize
            logger.debug(f"Processing {log_id}: Tokenization...")
            tokens = self.tokenizer.tokenize(signal_text)

            # Step 3: LogBERT embedding & anomaly detection
            logger.debug(f"Processing {log_id}: Embedding & detection...")
            anomaly_score = 0.0
            anomaly_confidence = 0.0
            is_anomalous = False
            method = "zscore"  # fallback

            if self.logbert_inference:
                try:
                    embedding = self.logbert_inference.encode_sequence(tokens)
                    self.anomaly_detector.add_embedding(embedding)
                    anomaly_score, anomaly_confidence = self.logbert_inference.compute_anomaly_score(
                        embedding
                    )
                    method = "logbert_vhm"
                except (ValueError, RuntimeError, KeyError) as e:
                    logger.warning(f"LogBERT inference failed: {e}", exc_info=True)
                    errors.append(f"LogBERT inference error: {str(e)}")
                except Exception as e:
                    logger.warning(f"LogBERT inference unexpected failure: {e}", exc_info=True)
                    errors.append(f"LogBERT inference unexpected error: {str(e)}")

            # Add to detector
            self.anomaly_detector.add_score(anomaly_score)
            result = self.anomaly_detector.detect()

            is_anomalous = result.is_anomalous
            anomaly_score = result.score
            anomaly_confidence = result.confidence
            method = result.method

            logger.info(
                f"Anomaly detection: {log_id} → Score={anomaly_score:.3f}, "
                f"Conf={anomaly_confidence:.1%}, Method={method}, Anomalous={is_anomalous}"
            )

            # Step 4: RCA if anomalous
            rca_trace = None
            judge_verdict = None
            report_files = None

            if is_anomalous and self.marcra:
                logger.debug(f"Processing {log_id}: RCA analysis...")

                # Create anomaly node (simplified)
                anomaly_node_id = f"anomaly_{log_id}"

                # Run MA-RCA
                try:
                    rca_trace = self.marcra.analyze(
                        anomaly_node_id=anomaly_node_id,
                        context_logs=[signal_text],
                        max_hypotheses=5,
                    )
                    logger.info(f"RCA complete: Root={rca_trace.root_cause_id}, Conf={rca_trace.confidence:.1%}")

                    # Step 5: Run graph propagation
                    if self.graph_propagation and rca_trace.causal_path:
                        logger.debug(f"Processing {log_id}: Graph propagation...")
                        self.graph_propagation.build_from_graph(
                            anomaly_node_ids=[anomaly_node_id]
                        )
                        ranked_roots = self.graph_propagation.rank_root_candidates()
                        if ranked_roots:
                            rca_trace.root_cause_id = ranked_roots[0]
                            logger.info(f"Graph propagation: Adjusted root to {ranked_roots[0]}")

                    # Step 6: Judge validation
                    logger.debug(f"Processing {log_id}: Judge validation...")
                    trace = ReasoningTrace(
                        root_cause_id=rca_trace.root_cause_id,
                        explanation=f"Root cause: {rca_trace.root_cause_type}",
                        evidence=rca_trace.validation_notes,
                        causal_chain=rca_trace.causal_path,
                        timestamps=[],
                    )
                    judge_result = self.judge.evaluate(trace)
                    judge_verdict = judge_result.to_dict()
                    logger.info(f"Judge verdict: {judge_verdict['passed']} (Conf={judge_verdict['confidence']:.1%})")

                    # Step 7: Generate report
                    logger.debug(f"Processing {log_id}: Report generation...")
                    report_files = generate_full_report(
                        anomaly_id=log_id,
                        rca_trace=rca_trace.to_dict(),
                        judge_verdict=judge_verdict,
                        output_dir="/tmp/contextwatch_reports",
                    )
                    logger.info(f"Reports generated: {report_files}")

                except (ValueError, KeyError, RuntimeError) as e:
                    logger.error(f"RCA pipeline failed: {e}", exc_info=True)
                    errors.append(f"RCA error: {str(e)}")
                except Exception as e:
                    logger.error(f"RCA pipeline unexpected failure: {e}", exc_info=True)
                    errors.append(f"RCA unexpected error: {str(e)}")

            # Compute execution time
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0

            return PipelineResult(
                anomaly_id=log_id,
                is_anomalous=is_anomalous,
                anomaly_score=anomaly_score,
                anomaly_confidence=anomaly_confidence,
                method=method,
                rca_trace=rca_trace,
                judge_verdict=judge_verdict,
                report_files=report_files,
                execution_time_ms=elapsed_ms,
                errors=errors if errors else None,
            )

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error(f"Pipeline failed for {log_id}: {e}", exc_info=True)
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0
            return PipelineResult(
                anomaly_id=log_id,
                is_anomalous=False,
                anomaly_score=0.0,
                anomaly_confidence=0.0,
                method="error",
                execution_time_ms=elapsed_ms,
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Pipeline unexpected error for {log_id}: {e}", exc_info=True)
            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000.0
            return PipelineResult(
                anomaly_id=log_id,
                is_anomalous=False,
                anomaly_score=0.0,
                anomaly_confidence=0.0,
                method="error",
                execution_time_ms=elapsed_ms,
                errors=[str(e)],
            )

    def process_batch(
        self,
        logs: List[Tuple[str, Dict[str, Any], str]],
    ) -> List[PipelineResult]:
        """Process a batch of logs in parallel or sequentially.
        
        Args:
            logs: List of (log_id, log_dict, protocol) tuples
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        for log_id, log_dict, protocol in logs:
            result = self.process_log(log_id, log_dict, protocol)
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics for monitoring."""
        return {
            "timestamps": datetime.utcnow().isoformat(),
            "logbert": {
                "version": self.logbert_version,
                "initialized": self.logbert_inference is not None,
                "stats": self.logbert_inference.get_statistics() if self.logbert_inference else {},
            },
            "detector": self.anomaly_detector.get_statistics(),
            "marcra": {"initialized": self.marcra is not None},
            "graph_propagation": {
                "initialized": self.graph_propagation is not None,
                "stats": self.graph_propagation.get_statistics() if self.graph_propagation else {},
            },
        }
