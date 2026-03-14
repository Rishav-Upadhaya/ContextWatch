from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.middleware import timing_middleware
from api.routes.analyze import router as analyze_router
from api.routes.anomalies import router as anomalies_router
from api.routes.graph import router as graph_router
from api.routes.health import router as health_router
from api.routes.ingest import router as ingest_router
from api.routes.mcp_analysis import router as mcp_analysis_router
from api.store import DurableStore
from config.logging_config import setup_logging
from config.settings import get_settings
from core.classifier import AnomalyClassifier
from core.detector import AnomalyDetector
from core.embedder import LogEmbedder
from core.knowledge_graph import KnowledgeGraph
from core.llm_explainer import LLMExplainer
from core.mcp_ml_assist import DistilBERTSignalAssist
from core.normalizer import LogNormalizer

# Initialize logging
logger = setup_logging()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_runtime_security()
    logger.info("Starting ContextWatch API...")
    logger.info(f"Configuration: threshold={settings.ANOMALY_THRESHOLD}, model={settings.EMBEDDING_MODEL}")
    
    app.state.settings = settings
    app.state.normalizer = LogNormalizer()
    app.state.embedder = LogEmbedder(settings)
    app.state.detector = AnomalyDetector(app.state.embedder, settings)
    app.state.classifier = AnomalyClassifier(settings)
    app.state.kg = KnowledgeGraph(settings)
    app.state.explainer = LLMExplainer(settings)
    app.state.store = DurableStore(settings.STORE_DB_PATH)
    app.state.mcp_ml_assist = None
    if settings.MCP_HYBRID_ENABLED and not settings.MCP_ML_KILLSWITCH and settings.MCP_HYBRID_PHASE >= 2:
        try:
            app.state.mcp_ml_assist = DistilBERTSignalAssist(settings.MCP_ML_MODEL_NAME)
            logger.info("MCP DistilBERT signal assist initialized for hybrid phase >=2")
        except Exception as exc:
            logger.warning(f"Failed to initialize MCP DistilBERT assist, continuing with rules-only: {exc}")
    app.state.metrics = {
        "requests_total": 0,
        "requests_failed": 0,
        "latency_sum_ms": 0.0,
        "latency_count": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    
    logger.info("All services initialized successfully")
    yield
    
    logger.info("Shutting down ContextWatch API...")
    app.state.store.close()
    app.state.kg.close()


app = FastAPI(title="ContextWatch API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

settings = get_settings()
allow_origins = [x.strip() for x in settings.CORS_ORIGINS.split(",") if x.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(timing_middleware)

app.include_router(ingest_router)
app.include_router(analyze_router)
app.include_router(mcp_analysis_router)
app.include_router(anomalies_router)
app.include_router(graph_router)
app.include_router(health_router)
