from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from config.settings import Settings
from core.schema import AnomalyResult, RCAResult


class KnowledgeGraph:
    def __init__(self, settings: Settings):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        self._ensure_indexes()

    def close(self) -> None:
        self.driver.close()

    def _ensure_indexes(self) -> None:
        queries = [
            "CREATE INDEX event_log_id IF NOT EXISTS FOR (e:Event) ON (e.log_id)",
            "CREATE INDEX session_id_idx IF NOT EXISTS FOR (s:Session) ON (s.session_id)",
            "CREATE INDEX agent_id_idx IF NOT EXISTS FOR (a:Agent) ON (a.agent_id)",
            "CREATE INDEX anomaly_log_id_idx IF NOT EXISTS FOR (a:Anomaly) ON (a.log_id)",
        ]
        with self.driver.session() as session:
            for query in queries:
                session.run(query)

    def upsert_event(self, normalized_log: dict[str, Any], anomaly: AnomalyResult | None = None) -> None:
        metadata = normalized_log["metadata"]
        query = """
        MERGE (e:Event {log_id: $log_id})
        SET e.timestamp = datetime($timestamp),
            e.agent_id = $agent_id,
            e.protocol = $protocol,
            e.session_id = $session_id
        MERGE (s:Session {session_id: $session_id})
        MERGE (ag:Agent {agent_id: $agent_id})
        MERGE (e)-[:PART_OF]->(s)
        MERGE (e)-[:EXECUTED_BY]->(ag)
        """
        params = {
            "log_id": normalized_log["log_id"],
            "timestamp": str(normalized_log["timestamp"]),
            "agent_id": normalized_log["agent_id"],
            "protocol": normalized_log["protocol"],
            "session_id": normalized_log["session_id"],
        }
        tool_name = metadata.get("tool_name")
        if tool_name:
            query += """
            MERGE (t:Tool {name: $tool_name})
            SET t.is_known = $tool_is_known
            MERGE (e)-[:USED_TOOL]->(t)
            """
            params["tool_name"] = tool_name
            params["tool_is_known"] = not str(tool_name).startswith("agent_")

        reasoning = metadata.get("reasoning_step") or metadata.get("message_content")
        intent = metadata.get("intent") or metadata.get("task_intent")
        if reasoning:
            query += """
            MERGE (r:ReasoningStep {text: $reasoning_text})
            SET r.intent = $intent,
                r.coherence_score = $coherence_score
            MERGE (e)-[:HAS_REASONING]->(r)
            """
            params["reasoning_text"] = reasoning
            params["intent"] = intent
            params["coherence_score"] = 0.2 if metadata.get("is_anomaly") else 0.8

        if anomaly and anomaly.is_anomaly:
            query += """
            MERGE (a:Anomaly {log_id: $anomaly_log_id})
            SET a.anomaly_type = $anomaly_type,
                a.confidence = $confidence,
                a.anomaly_score = $anomaly_score
            MERGE (a)-[:DETECTED_IN]->(e)
            """
            params["anomaly_log_id"] = anomaly.log_id
            params["anomaly_type"] = anomaly.anomaly_type or "UNKNOWN"
            params["confidence"] = anomaly.confidence
            params["anomaly_score"] = anomaly.anomaly_score

        with self.driver.session() as session:
            session.run(query, params)

    def create_temporal_link(self, current_log_id: str, previous_log_id: str, lag_ms: int) -> None:
        query = """
        MATCH (c:Event {log_id: $current_log_id}), (p:Event {log_id: $previous_log_id})
        MERGE (c)-[:PRECEDED_BY {lag_ms: $lag_ms}]->(p)
        """
        with self.driver.session() as session:
            session.run(query, current_log_id=current_log_id, previous_log_id=previous_log_id, lag_ms=lag_ms)

    def create_causal_link(self, current_log_id: str, previous_log_id: str, relation: str = "CAUSED_BY") -> None:
        if relation not in {"CAUSED_BY", "SEQUENTIAL_CAUSAL_HINT"}:
            relation = "CAUSED_BY"
        query = f"""
        MATCH (a:Anomaly {{log_id: $current_log_id}}), (b:Anomaly {{log_id: $previous_log_id}})
        MERGE (a)-[:{relation}]->(b)
        """
        with self.driver.session() as session:
            session.run(query, current_log_id=current_log_id, previous_log_id=previous_log_id)

    def create_delegation_link(self, from_log_id: str, to_log_id: str) -> None:
        query = """
        MATCH (f:Event {log_id: $from_log_id}), (t:Event {log_id: $to_log_id})
        MERGE (f)-[:DELEGATED_TO]->(t)
        """
        with self.driver.session() as session:
            session.run(query, from_log_id=from_log_id, to_log_id=to_log_id)

    def trace_rca(self, log_id: str) -> RCAResult:
        query = """
        MATCH path = (a:Anomaly {log_id: $log_id})-[:CAUSED_BY*1..3]-(root)
        RETURN [n IN nodes(path) | coalesce(n.log_id, n.name)] AS chain,
               length(path) AS hops
        ORDER BY hops ASC
        LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(query, log_id=log_id).single()
        if not record:
            return RCAResult(
                root_cause_log_id=log_id,
                causal_chain=[log_id],
                hop_count=0,
                explanation="No explicit CAUSED_BY chain found; anomaly treated as root.",
            )
        chain = [item for item in record["chain"] if item]
        return RCAResult(
            root_cause_log_id=chain[-1],
            causal_chain=chain,
            hop_count=int(record["hops"]),
            explanation="Root cause identified from anomaly causal graph.",
        )

    def delegation_trace(self, log_id: str) -> list[dict[str, Any]]:
        query = """
        MATCH (e:Event {log_id: $log_id})<-[:DELEGATED_TO*1..5]-(chain:Event)
        RETURN chain.log_id AS log_id, chain.timestamp AS timestamp
        ORDER BY chain.timestamp
        """
        with self.driver.session() as session:
            result = session.run(query, log_id=log_id)
            return [dict(row) for row in result]

    def trigger_graph(self, log_id: str) -> dict[str, Any]:
        query = """
        MATCH p=(a:Anomaly {log_id: $log_id})-[:CAUSED_BY|SEQUENTIAL_CAUSAL_HINT*0..3]->(b:Anomaly)
        WITH p
        ORDER BY length(p) DESC
        LIMIT 1
        UNWIND nodes(p) AS n
        WITH collect(DISTINCT n) AS nodes, p
        UNWIND relationships(p) AS r
        RETURN nodes,
               collect(DISTINCT {
                 source: startNode(r).log_id,
                 target: endNode(r).log_id,
                 type: type(r)
               }) AS edges
        """
        with self.driver.session() as session:
            row = session.run(query, log_id=log_id).single()

        if not row:
            return {
                "root_log_id": log_id,
                "nodes": [{"id": log_id, "label": log_id[:8], "kind": "anomaly"}],
                "edges": [],
            }

        nodes = [
            {
                "id": n.get("log_id"),
                "label": str(n.get("log_id", ""))[:8],
                "kind": "anomaly",
                "anomaly_type": n.get("anomaly_type"),
                "confidence": n.get("confidence"),
                "anomaly_score": n.get("anomaly_score"),
            }
            for n in row["nodes"]
            if n.get("log_id")
        ]
        edges = row["edges"]
        root_log_id = edges[-1]["target"] if edges else log_id
        return {"root_log_id": root_log_id, "nodes": nodes, "edges": edges}

    def session_anomaly_density(self, session_id: str) -> dict[str, Any]:
        query = """
        MATCH (s:Session {session_id: $session_id})<-[:PART_OF]-(e:Event)<-[:DETECTED_IN]-(a:Anomaly)
        RETURN count(a) as anomaly_count, collect(a.anomaly_type) as types
        """
        with self.driver.session() as session:
            row = session.run(query, session_id=session_id).single()
        if not row:
            return {"session_id": session_id, "anomaly_count": 0, "types": []}
        return {
            "session_id": session_id,
            "anomaly_count": int(row["anomaly_count"]),
            "types": row["types"],
        }
