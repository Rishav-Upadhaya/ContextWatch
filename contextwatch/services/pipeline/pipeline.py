"""contextwatch.services.pipeline.pipeline
==========================================
Primary orchestration pipeline for ContextWatch detection and RCA.

Coordinates normalization, tokenization, LogBERT embedding, VHM scoring,
optional MA-RCA reasoning, and persistence of normal/anomalous outcomes.

Used by: contextwatch.controllers.api
Depends on: services.ai.logbert, services.detection.vhm_engine
"""

from __future__ import annotations

import uuid
import logging
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

from contextwatch.services.detection.signal_filter import filter_log_signal  # noqa: E402
from contextwatch.utils.redaction import redact_log
from contextwatch.adapters.trace_mapper import map_to_otel_trace
from contextwatch.adapters.postgres import PostgresLogStore, StoredAnomaly, StoredNormalLog
from contextwatch.utils.tokenizer import LogTokenizer
from contextwatch.services.ai.logbert import LogBERTEncoder, LogBERTConfig
from contextwatch.services.ai.training_data import infer_log_label, load_system_training_corpus, log_to_training_text
from contextwatch.services.detection.error_classifier import RuleBasedErrorClassifier
from contextwatch.services.detection.error_type_classifier import EnsembleErrorClassifier, ErrorTypeClassifierML
from contextwatch.services.detection.vhm_engine import VHMEngine, cosine_distance
from contextwatch.services.reasoning.mar_cra import MARCRA, MARCATrace
from contextwatch.services.reasoning.llm_judge import ReasoningJudge, ReasoningTrace
from contextwatch.adapters.vector_store import VectorStore
from contextwatch.services.graph.knowledge_graph import KnowledgeGraph


# ── Explicit anomaly taxonomy (MCP/A2A protocol events) ──────────────────────

_EXPLICIT_ANOMALY_EVENTS = {
    "TOOL_HALLUCINATION",
    "CONTEXT_POISONING",
    "REGISTRY_OVERFLOW",
    "DELEGATION_CHAIN_FAILURE",
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DetectionOutcome:
    log_id: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    anomaly_type: Optional[str]
    hypersphere_distance: Optional[float]
    cosine_distance: Optional[float]
    marca_trace: Optional[MARCATrace]
    judge_verdict: Optional[dict]
    explanation: str
    token_efficiency: Optional[dict]
    trace_context: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "hypersphere_distance": self.hypersphere_distance,
            "cosine_distance": self.cosine_distance,
            "marca_trace": self.marca_trace.to_dict() if self.marca_trace else None,
            "judge_verdict": self.judge_verdict,
            "explanation": self.explanation,
            "token_efficiency": self.token_efficiency,
            "trace_context": self.trace_context,
        }


@dataclass
class BatchDetectionOutcome:
    batch_size: int
    anomaly_count: int
    normal_count: int
    results: List[DetectionOutcome]

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "anomaly_count": self.anomaly_count,
            "normal_count": self.normal_count,
            "results": [r.to_dict() for r in self.results],
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ContextWatchPipeline:
    """Runs end-to-end anomaly detection and RCA for log events.

    Attributes:
        config: LogBERT model configuration.
        encoder: NumPy-based LogBERT embedding model.
        vhm: Hypersphere anomaly detector.
        tokenizer: Text-to-token encoder.
        vector_store: In-memory baseline embedding store.
        kg: Knowledge graph for anomaly context.
        normal_store: PostgreSQL adapter for persistence.
    """

    def __init__(
        self,
        model_config: Optional[LogBERTConfig] = None,
        vhm_engine: Optional[VHMEngine] = None,
        tokenizer: Optional[LogTokenizer] = None,
        vector_store: Optional[VectorStore] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        marcra: Optional[MARCRA] = None,
        normal_store: Optional[PostgresLogStore] = None,
    ):
        self.config = model_config or LogBERTConfig()
        self.encoder = LogBERTEncoder(config=self.config)
        self.vhm = vhm_engine or VHMEngine(dimensions=self.config.d_model)
        self.tokenizer = tokenizer or LogTokenizer()
        self.vector_store = vector_store or VectorStore()
        self.kg = knowledge_graph or KnowledgeGraph()
        self.normal_store = normal_store or PostgresLogStore()
        self.marcra = marcra or MARCRA(
            knowledge_graph=self.kg,
            vector_store=self.vector_store,
        )
        self.judge = ReasoningJudge()
        self.error_classifier = EnsembleErrorClassifier(ml_classifier=ErrorTypeClassifierML())
        self.calibration_metrics: dict[str, Any] = {}
        self.vhm_metrics: dict[str, Any] = {}
        self._bootstrap_normal_baseline()
        self._warmup_vhm_from_system_data()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_log_level(self, raw_log: dict) -> str:
        params = raw_log.get("params") or {}
        if isinstance(params, dict):
            level = params.get("level")
            if isinstance(level, str):
                return level.lower()
        return ""

    def _extract_event_name(self, raw_log: dict) -> str:
        params = raw_log.get("params") or {}
        if not isinstance(params, dict):
            return ""
        data = params.get("data") or {}
        if not isinstance(data, dict):
            return ""
        event = data.get("event", "")
        return str(event).strip().upper() if event else ""

    def _detect_explicit_anomaly(self, raw_log: dict) -> Tuple[bool, Optional[str], str]:
        event = self._extract_event_name(raw_log)
        if event in _EXPLICIT_ANOMALY_EVENTS:
            return True, event, f"explicit anomaly event '{event}'"
        return False, None, "no explicit anomaly marker"

    def _bootstrap_normal_baseline(self) -> None:
        """Pre-load saved normal embeddings into VHM at startup."""
        baseline = self.normal_store.fetch_embeddings(limit=1000)
        if not baseline:
            return
        
        # Filter for proper dimensionality (in case of legacy data)
        valid_baseline = [emb for emb in baseline if len(emb) == self.config.d_model]
        
        for i, emb in enumerate(valid_baseline):
            self.vector_store.add([emb], [{"doc_id": f"boot_{i}", "type": "normal"}])
        if len(valid_baseline) >= 10:
            self.vhm.fit(valid_baseline)

    def _warmup_vhm_from_system_data(self, max_normal: int = 256, max_anomaly: int = 128) -> None:
        """Cold-start fit/calibration from bundled datasets when baseline DB is empty.

        This prevents model-info values from staying at zero on first startup.
        """
        if self.vhm.is_fitted:
            return

        data_root = Path(__file__).resolve().parents[2] / "data"
        normal_paths = [
            data_root / "golden_dataset" / "golden_normal.jsonl",
            data_root / "synthetic" / "mcp" / "mcp_normal_logs.jsonl",
            data_root / "synthetic" / "a2a" / "a2a_normal_logs.jsonl",
        ]
        anomaly_paths = [
            data_root / "golden_dataset" / "golden_anomalies.jsonl",
            data_root / "synthetic" / "mcp" / "mcp_anomaly_logs.jsonl",
            data_root / "synthetic" / "a2a" / "a2a_anomaly_logs.jsonl",
        ]

        def load_embeddings(paths: List[Path], limit: int) -> List[List[float]]:
            out: List[List[float]] = []
            for path in paths:
                if not path.exists():
                    continue
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if len(out) >= limit:
                                return out
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                payload = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            text = log_to_training_text(payload)
                            token_ids = self.tokenizer.encode(text, max_length=self.config.max_seq_len)
                            emb = self.encoder.forward(token_ids)
                            if len(emb) == self.config.d_model:
                                out.append(emb)
                except OSError as exc:
                    logger.warning("Failed reading warmup dataset %s: %s", path, exc)
            return out

        normal_embs = load_embeddings(normal_paths, max_normal)
        if len(normal_embs) < 10:
            logger.info(
                "VHM warmup skipped: insufficient normal embeddings from system datasets (%s)",
                len(normal_embs),
            )
            return

        self.vector_store.add(
            normal_embs,
            [{"doc_id": f"sys_warm_{i}", "type": "normal", "protocol": "SYSTEM"} for i in range(len(normal_embs))],
        )

        try:
            self.vhm.fit(normal_embs)
            logger.info("VHM warmup fit complete from system datasets: normal=%s radius=%.6f", len(normal_embs), self.vhm.radius)
        except Exception as exc:
            logger.warning("VHM warmup fit failed: %s", exc, exc_info=True)
            return

        anomaly_embs = load_embeddings(anomaly_paths, max_anomaly)
        if len(anomaly_embs) < 5:
            logger.info("VHM warmup calibration skipped: insufficient anomaly embeddings (%s)", len(anomaly_embs))
            return

        try:
            calibration = self.vhm.calibrate(normal_embs, anomaly_embs)
            calibration["training_normals"] = len(normal_embs)
            calibration["training_anomalies"] = len(anomaly_embs)
            self.calibration_metrics = calibration
            logger.info(
                "VHM warmup calibration complete: decision_radius=%.6f f1=%.4f",
                self.vhm.decision_radius,
                float(calibration.get("best_f1", 0.0)),
            )
        except Exception as exc:
            logger.warning("VHM warmup calibration failed: %s", exc, exc_info=True)

    def _get_baseline_embeddings(self) -> List[List[float]]:
        """Return all normal embeddings currently in the vector store."""
        embeddings = []
        for item in self.vector_store.items():
            emb = item.get("embedding", []) if isinstance(item, dict) else getattr(item, "embedding", [])
            if len(emb) == self.config.d_model:
                embeddings.append(emb)
        return embeddings

    def _encode_texts(self, texts: List[str]) -> List[List[int]]:
        return [
            self.tokenizer.encode(text, max_length=self.config.max_seq_len)
            for text in texts
        ]

    def _calibrate_detector(
        self,
        normal_sequences: List[List[int]],
        anomaly_sequences: List[List[int]],
    ) -> dict[str, Any]:
        normal_embeddings = [self.encoder.forward(seq) for seq in normal_sequences]
        anomaly_embeddings = [self.encoder.forward(seq) for seq in anomaly_sequences]

        all_baseline = self._get_baseline_embeddings() + normal_embeddings
        if len(all_baseline) >= 1:
            self.vhm.fit(all_baseline)

        calibration = self.vhm.calibrate(all_baseline, anomaly_embeddings)
        calibration["training_normals"] = len(normal_sequences)
        calibration["training_anomalies"] = len(anomaly_sequences)
        self.calibration_metrics = calibration
        return calibration

    def _persist_graph(self) -> None:
        snap = self.kg.serialize()
        self.normal_store.upsert_graph_snapshot(
            nodes=snap.get("nodes", []),
            edges=snap.get("edges", []),
        )

    # ── Core processing ───────────────────────────────────────────────────────

    def process(
        self,
        raw_log: dict,
        protocol: str,
        context_logs: Optional[List[str]] = None,
        include_rca: bool = True,
        update_baseline: bool = True,
        enqueue_task: Optional[Any] = None,
    ) -> DetectionOutcome:
        """
        End-to-end processing of a single log entry.

        Steps:
          1. Normalize: signal filter → secret redaction
          2. Tokenize
          3. LogBERT embedding (finetuned transformer)
          4. Vector distance: cosine + euclidean to hypersphere
          5. VHM anomaly check
          6. IF anomaly → MA-RCA + LLM judge
          7. Store result in PostgreSQL
        """
        log_id = raw_log.get("log_id") or str(uuid.uuid4())

        # ── Step 1: Normalize ─────────────────────────────────────────────────
        try:
            signal = filter_log_signal(raw_log, protocol)
            redacted = redact_log(signal.signal_text)
            clean_text = redacted.data
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Normalization failed for {log_id}: {e}", exc_info=True)
            return DetectionOutcome(
                log_id=log_id, is_anomaly=False, anomaly_score=0.0, confidence=0.0,
                anomaly_type="ERROR", hypersphere_distance=0.0, cosine_distance=0.0,
                marca_trace=None, judge_verdict=None, explanation=f"Normalization error: {e}",
                token_efficiency=None, trace_context=None
            )
        except Exception as e:
            logger.error(f"Normalization unexpected error for {log_id}: {e}", exc_info=True)
            return DetectionOutcome(
                log_id=log_id, is_anomaly=False, anomaly_score=0.0, confidence=0.0,
                anomaly_type="ERROR", hypersphere_distance=0.0, cosine_distance=0.0,
                marca_trace=None, judge_verdict=None, explanation=str(e),
                token_efficiency=None, trace_context=None
            )

        # Trace mapping
        trace_map = map_to_otel_trace(raw_log, protocol)
        trace_dict: Optional[dict] = None
        if trace_map:
            trace_dict = {
                "trace_id": trace_map.trace_context.trace_id,
                "span_id": trace_map.trace_context.span_id,
                "original_system": trace_map.original_system,
            }
            if trace_map.parent_span_id:
                trace_dict["parent_span_id"] = trace_map.parent_span_id

        # ── Step 2–3: Tokenize + LogBERT embed ───────────────────────────────
        try:
            token_ids = self.tokenizer.encode(clean_text, max_length=self.config.max_seq_len)
            embedding = self.encoder.forward(token_ids)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Embedding failed for {log_id}: {e}", exc_info=True)
            embedding = []  # Fallback to empty context
        except Exception as e:
            logger.error(f"Embedding unexpected error for {log_id}: {e}", exc_info=True)
            embedding = []

        # ── Step 4: Explicit taxonomy check ──────────────────────────────────
        explicit_anomaly, explicit_type, explicit_reason = self._detect_explicit_anomaly(raw_log)

        # ── Step 5: VHM anomaly check (vector distance) ───────────────────────
        # Fit VHM if we have enough baseline
        baseline = self._get_baseline_embeddings()
        if not self.vhm.is_fitted and len(baseline) >= 10:
            self.vhm.fit(baseline)

        vhm_anomaly, vhm_distance = False, 0.0
        cos_dist = 0.0
        cluster_id = -1
        cluster_threshold = 0.0
        normalized_distance = 0.0

        if self.vhm.is_fitted and embedding:
            try:
                vhm_result = self.vhm.score_details(embedding)
                vhm_anomaly = bool(vhm_result["is_anomaly"])
                vhm_distance = float(vhm_result["distance"])
                cluster_id = int(vhm_result["cluster_id"])
                cluster_threshold = float(vhm_result["threshold"])
                normalized_distance = float(vhm_result["normalized_distance"])
                if self.vhm.center:
                    cos_dist = cosine_distance(embedding, self.vhm.center)
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.error(f"VHM scoring failed for {log_id}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"VHM scoring unexpected error for {log_id}: {e}", exc_info=True)

        # Explicit taxonomy always takes priority
        is_anomaly = explicit_anomaly or vhm_anomaly

        # Confidence
        if explicit_anomaly:
            level = self._extract_log_level(raw_log)
            confidence = 0.98 if level == "error" else 0.90
        elif is_anomaly:
            confidence = min(1.0, max(0.0, normalized_distance - 1.0)) if self.vhm.is_fitted else 0.75
        else:
            threshold_radius = cluster_threshold or (self.vhm.decision_radius if self.vhm.decision_radius > 0 else self.vhm.radius)
            if self.vhm.is_fitted and threshold_radius > 0:
                confidence = 1.0 - min(1.0, vhm_distance / threshold_radius)
            else:
                confidence = 0.90

        # ── Step 5.5: Error type classification (when anomaly detected) ──────
        classified_error_type: Optional[str] = explicit_type  # Default to explicit if found
        classification_description: str = ""
        
        if is_anomaly and not explicit_type:
            # Try ensemble classifier for deeper error type classification
            try:
                embedding_array = np.array(embedding, dtype=np.float32) if embedding else None
                classified_type, class_conf, class_reason = self.error_classifier.classify(raw_log, embedding_array)
                if classified_type:
                    classified_error_type = classified_type
                    classification_description = f"{class_reason} (conf={class_conf:.2f})"
                    logger.debug(f"Error classified: {log_id} -> {classified_type} ({class_conf:.2f})")
            except Exception as e:
                logger.debug(f"Error classification failed for {log_id}: {e}")
                classification_description = f"Classification error: {e}"

        # ── Step 6: MA-RCA (only for anomalies) ──────────────────────────────
        marca_trace = None
        judge_dict: Optional[dict] = None
        anomaly_node_id: Optional[str] = None

        if is_anomaly:
            event_node_id = f"event_{uuid.uuid4()}"
            self.kg.add_node(event_node_id, "Event", {
                "log_id": log_id,
                "protocol": protocol,
                "embedding_dim": len(embedding),
                "timestamp": trace_dict.get("trace_id", "") if trace_dict else "",
            })
            anomaly_node = self.kg.add_node(f"anomaly_{event_node_id}", "Anomaly", {
                "log_id": log_id,
                "score": vhm_distance,
                "confidence": confidence,
                "anomaly_type": classified_error_type or "ANOMALY",
            })
            anomaly_node_id = anomaly_node.id
            self.kg.add_edge(event_node_id, anomaly_node_id, "HAS_ANOMALY")

            if include_rca:
                try:
                    marca_trace = self.marcra.analyze(
                        anomaly_node_id=anomaly_node_id,
                        context_logs=context_logs or [],
                    )
                    reasoning_trace = ReasoningTrace(
                        root_cause_id=marca_trace.root_cause_id,
                        explanation=marca_trace.root_cause_type,
                        evidence=[],
                        causal_chain=marca_trace.causal_path,
                        timestamps=[],
                    )
                    verdict = self.judge.evaluate(reasoning_trace)
                    judge_dict = verdict.to_dict()
                except (ValueError, KeyError, RuntimeError) as e:
                    logger.error(f"RCA generation failed for {log_id}: {e}", exc_info=True)
                    judge_dict = {"passed": False, "confidence": 0.0, "notes": f"RCA error: {e}"}
                except Exception as e:
                    logger.error(f"RCA unexpected error for {log_id}: {e}", exc_info=True)
                    judge_dict = {"passed": False, "confidence": 0.0, "notes": "RCA unexpected error"}
            else:
                judge_dict = {
                    "passed": True,
                    "confidence": confidence,
                    "notes": "RCA skipped for batch performance",
                }

        # ── Step 7: Explanation ───────────────────────────────────────────────
        if is_anomaly:
            root_cause = marca_trace.root_cause_type if marca_trace else "RCA skipped"
            if explicit_anomaly:
                explanation = (
                    f"Anomaly via explicit taxonomy: {explicit_type}. "
                    f"Reason: {explicit_reason}. Root cause: {root_cause}. "
                    f"VHM dist={vhm_distance:.4f}, cosine dist={cos_dist:.4f}."
                )
            else:
                explanation = (
                    f"Anomaly via VHM: distance {vhm_distance:.4f} > threshold {cluster_threshold:.4f} "
                    f"(cluster={cluster_id}, normalized={normalized_distance:.4f}). "
                    f"Type: {classified_error_type}. {classification_description} "
                    f"Cosine dist={cos_dist:.4f}. Root cause: {root_cause}."
                )
        else:
            explanation = (
                f"Normal log. VHM dist={vhm_distance:.4f} <= threshold={cluster_threshold:.4f} "
                f"(cluster={cluster_id}, normalized={normalized_distance:.4f}). Cosine dist={cos_dist:.4f}."
            )

        # ── Step 8: Persist ───────────────────────────────────────────────────
        from functools import partial
        if not is_anomaly:
            task1 = partial(self.normal_store.insert_normal_log, StoredNormalLog(
                log_id=log_id,
                protocol=protocol,
                normalized_text=str(clean_text),
                embedding=embedding,
                trace_context=trace_dict,
            ))
            if enqueue_task: enqueue_task(task1)
            else: task1()

            if update_baseline and embedding:
                self.vector_store.add(
                    [embedding],
                    [{"doc_id": log_id, "type": "normal", "protocol": protocol}],
                )
                self.vhm_metrics = self.vhm.update_buffer([embedding], auto_refit=True)
        else:
            task2 = partial(self.normal_store.upsert_anomaly, StoredAnomaly(
                anomaly_id=anomaly_node_id or f"anomaly_{log_id}",
                log_id=log_id,
                anomaly_type=classified_error_type or "ANOMALY",
                score=float(vhm_distance),
                confidence=float(confidence),
                explanation=explanation,
                details={
                    "cosine_distance": cos_dist,
                    "vhm_distance": vhm_distance,
                    "cluster_id": cluster_id,
                    "cluster_threshold": cluster_threshold,
                    "normalized_distance": normalized_distance,
                    "token_efficiency": {
                        "raw_tokens": signal.raw_tokens_preserved,
                        "overhead_removed": signal.overhead_tokens_removed,
                    },
                    "trace_context": trace_dict or {},
                    "judge_verdict": judge_dict or {},
                },
            ))
            if enqueue_task: enqueue_task(task2)
            else: task2()

        snap = self.kg.serialize()
        task3 = partial(self.normal_store.upsert_graph_snapshot, snap.get("nodes", []), snap.get("edges", []))
        if enqueue_task: enqueue_task(task3)
        else: task3()

        return DetectionOutcome(
            log_id=log_id,
            is_anomaly=is_anomaly,
            anomaly_score=vhm_distance,
            confidence=confidence,
            anomaly_type=classified_error_type or (
                marca_trace.root_cause_type if marca_trace else ("ANOMALY" if is_anomaly else None)
            ),
            hypersphere_distance=vhm_distance,
            cosine_distance=cos_dist,
            marca_trace=marca_trace,
            judge_verdict=judge_dict,
            explanation=explanation,
            token_efficiency={
                "raw_tokens": signal.raw_tokens_preserved,
                "overhead_removed": signal.overhead_tokens_removed,
            },
            trace_context=trace_dict,
        )

    def process_batch(
        self,
        raw_logs: List[dict],
        protocol: Optional[str] = None,
        context_logs: Optional[List[str]] = None,
        include_rca: bool = False,
        enqueue_task: Optional[Any] = None,
    ) -> BatchDetectionOutcome:
        """Processes a log batch and returns aggregate detection outcomes.

        Args:
            raw_logs: Collection of raw protocol log payloads.
            protocol: Optional protocol override for the whole batch.
            context_logs: Optional contextual logs for RCA enrichment.
            include_rca: Whether RCA should run for detected anomalies.
            enqueue_task: Optional async task enqueue callback for writes.

        Returns:
            BatchDetectionOutcome with per-log results and summary counts.
        """
        results: List[DetectionOutcome] = []
        for raw_log in raw_logs:
            log_protocol = protocol or raw_log.get("protocol", "MCP")
            results.append(self.process(
                raw_log=raw_log,
                protocol=log_protocol,
                context_logs=context_logs,
                include_rca=include_rca,
                update_baseline=True,
                enqueue_task=enqueue_task,
            ))

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        return BatchDetectionOutcome(
            batch_size=len(results),
            anomaly_count=anomaly_count,
            normal_count=len(results) - anomaly_count,
            results=results,
        )

    # ── Finetuning ────────────────────────────────────────────────────────────

    def finetune(
        self,
        raw_logs: List[dict],
        epochs: int = 5,
        learning_rate: float = 1e-3,
        use_system_data: bool = True,
    ) -> List[dict]:
        """Finetunes LogBERT weights and refreshes VHM baseline.

        Args:
            raw_logs: Training logs used to derive token sequences.
            epochs: Number of finetuning epochs.
            learning_rate: Optimizer step size for parameter updates.

        Returns:
            Per-epoch training metrics including MLKP, VHM, and total loss.
        """
        from contextwatch.services.ai.logbert_trainer import LogBERTFinetuner

        if not use_system_data:
            normal_logs = [log for log in raw_logs if infer_log_label(log, "user_input") == 0]
            anomaly_logs = [log for log in raw_logs if infer_log_label(log, "user_input") == 1]
            from contextwatch.services.ai.training_data import TrainingCorpus
            corpus = TrainingCorpus(normal_logs=normal_logs, anomaly_logs=anomaly_logs, labeled_records=[])
        else:
            corpus = load_system_training_corpus(extra_logs=raw_logs)
        if use_system_data and raw_logs:
            logger.info("Finetune requested with %s extra user logs on top of system datasets", len(raw_logs))

        if not corpus.normal_logs:
            raise ValueError("Finetuning requires at least one normal log")

        normal_sequences = self._encode_texts([log_to_training_text(log) for log in corpus.normal_logs])
        anomaly_sequences = self._encode_texts([log_to_training_text(log) for log in corpus.anomaly_logs])

        finetuner = LogBERTFinetuner(
            config=self.config,
            learning_rate=learning_rate,
        )
        # Copy current encoder weights into finetuner
        finetuner.encoder.weights = self.encoder.weights

        metrics = finetuner.finetune(normal_sequences, epochs=epochs)

        # Apply updated weights back to the pipeline encoder
        self.encoder.weights = finetuner.get_encoder().weights

        calibration = self._calibrate_detector(normal_sequences, anomaly_sequences)

        results = [
            {
                "epoch": m.epoch,
                "loss_mlkp": m.loss_mlkp,
                "loss_vhm": m.loss_vhm,
                "loss_total": m.loss_total,
                "vhm_radius": m.vhm_radius,
                "n_sequences": m.n_sequences,
                "decision_radius": round(self.vhm.decision_radius, 6),
            }
            for m in metrics
        ]
        if results:
            results[-1]["calibration_f1"] = round(float(calibration.get("best_f1", 0.0)), 6)
            results[-1]["normal_samples"] = int(calibration.get("normal_samples", 0))
            results[-1]["anomaly_samples"] = int(calibration.get("anomaly_samples", 0))
        return results

    def add_baseline_log(self, log_text: str) -> None:
        """Add a single normal log text to the baseline."""
        ids = self.tokenizer.encode(log_text, max_length=self.config.max_seq_len)
        emb = self.encoder.forward(ids)
        self.vector_store.add([emb], [{"doc_id": str(uuid.uuid4()), "type": "normal"}])
        baseline = self._get_baseline_embeddings()
        if len(baseline) >= 10:
            self.vhm.fit(baseline)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_default_pipeline() -> ContextWatchPipeline:
    """Construct pipeline from environment settings."""
    from contextwatch.utils.config import load_settings
    settings = load_settings()

    config = LogBERTConfig(
        vocab_size=settings.LOGBERT_VOCAB_SIZE,
        d_model=settings.LOGBERT_D_MODEL,
        n_heads=settings.LOGBERT_N_HEADS,
        d_k=max(8, settings.LOGBERT_D_MODEL // max(settings.LOGBERT_N_HEADS, 1)),
        d_v=max(8, settings.LOGBERT_D_MODEL // max(settings.LOGBERT_N_HEADS, 1)),
        d_ff=max(64, settings.LOGBERT_D_MODEL * 4),
        max_seq_len=settings.LOGBERT_SEQ_LEN_MAX,
        n_layers=settings.LOGBERT_N_LAYERS,
    )
    return ContextWatchPipeline(
        model_config=config,
        normal_store=PostgresLogStore(settings.DATABASE_URL),
    )


"""End of pipeline module."""
