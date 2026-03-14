from __future__ import annotations

from random import sample
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from core.knowledge_graph import KnowledgeGraph


def evaluate_rca() -> dict:
    kg = KnowledgeGraph(get_settings())
    tested = 0
    hits = 0
    try:
        with kg.driver.session() as session:
            result = session.run("MATCH (a:Anomaly) RETURN a.log_id as log_id LIMIT 100")
            anomaly_ids = [row["log_id"] for row in result]
        for log_id in sample(anomaly_ids, k=min(100, len(anomaly_ids))):
            tested += 1
            rca = kg.trace_rca(log_id)
            if rca.hop_count <= 3:
                hits += 1
    finally:
        kg.close()
    precision = (hits / tested) if tested else 0.0
    print(f"precision@3hops={precision:.4f} tested={tested}")
    assert precision >= 0.75, f"RCA precision below target: {precision:.4f}"
    return {"precision_at_3": precision, "tested": tested}


if __name__ == "__main__":
    evaluate_rca()
