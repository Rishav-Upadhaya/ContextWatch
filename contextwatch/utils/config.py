"""Configuration management using environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _build_database_url() -> str:
    """Resolve DATABASE_URL from env or compose-style postgres components."""
    direct = os.getenv("DATABASE_URL", "").strip()
    if direct:
        return direct

    host = os.getenv("POSTGRES_HOST", "").strip()
    db = os.getenv("POSTGRES_DB", "").strip()
    user = os.getenv("POSTGRES_USER", "").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()
    port = os.getenv("POSTGRES_PORT", "5432").strip() or "5432"

    if host and db and user and password:
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    return ""


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    return float(value) if value else None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() == "true"


def _load_dotenv_if_present() -> None:
    """Load project .env into process env for local scripts.

    Existing environment variables take precedence.
    """
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    """Application settings sourced from environment variables."""

    # Anomaly detection thresholds
    ANOMALY_THRESHOLD: float
    ANOMALY_THRESHOLD_MCP: Optional[float]
    ANOMALY_THRESHOLD_A2A: Optional[float]

    # Vector store & embedding
    EMBEDDING_DIM: int
    NORMAL_BASELINE_PATH: str

    # Graph
    GRAPH_MAX_DEPTH: int

    # API
    API_HOST: str
    API_PORT: int

    # Postgres
    DATABASE_URL: str
    VECTOR_STORE_BACKEND: str

    # LogBERT Configuration
    LOGBERT_VERSION: str
    LOGBERT_D_MODEL: int
    LOGBERT_N_HEADS: int
    LOGBERT_N_LAYERS: int
    LOGBERT_SEQ_LEN_MAX: int
    LOGBERT_VOCAB_SIZE: int
    LOGBERT_WEIGHTS_DIR: str

    # MA-RCA Configuration
    MARCRA_MAX_HYPOTHESES: int
    VHM_MARGIN: float

    # Judge Configuration
    JUDGE_TARGET_FAILURES: str
    JUDGE_USE_LLM: bool
    ANTHROPIC_API_KEY: Optional[str]

    # Detection Method
    DETECTION_METHOD: str  # zscore, logbert_vhm, ensemble

    # Logging
    LOG_LEVEL: str


def load_settings() -> Settings:
    """Load the application settings."""
    _load_dotenv_if_present()
    return Settings(
        ANOMALY_THRESHOLD=_env_float("ANOMALY_THRESHOLD", 0.20),
        ANOMALY_THRESHOLD_MCP=_env_optional_float("ANOMALY_THRESHOLD_MCP"),
        ANOMALY_THRESHOLD_A2A=_env_optional_float("ANOMALY_THRESHOLD_A2A"),
        EMBEDDING_DIM=_env_int("EMBEDDING_DIM", 384),
        NORMAL_BASELINE_PATH=_env_str("NORMAL_BASELINE_PATH", "./data/chroma_db"),
        GRAPH_MAX_DEPTH=_env_int("GRAPH_MAX_DEPTH", 3),
        API_HOST=_env_str("API_HOST", "0.0.0.0"),
        API_PORT=_env_int("API_PORT", 8000),
        DATABASE_URL=_build_database_url(),
        VECTOR_STORE_BACKEND=_env_str("VECTOR_STORE_BACKEND", "memory"),
        LOGBERT_VERSION=_env_str("LOGBERT_VERSION", "v1"),
        LOGBERT_D_MODEL=_env_int("LOGBERT_D_MODEL", 64),
        LOGBERT_N_HEADS=_env_int("LOGBERT_N_HEADS", 4),
        LOGBERT_N_LAYERS=_env_int("LOGBERT_N_LAYERS", 2),
        LOGBERT_SEQ_LEN_MAX=_env_int("LOGBERT_SEQ_LEN_MAX", 64),
        LOGBERT_VOCAB_SIZE=_env_int("LOGBERT_VOCAB_SIZE", 5000),
        LOGBERT_WEIGHTS_DIR=_env_str("LOGBERT_WEIGHTS_DIR", "./contextwatch/weights"),
        MARCRA_MAX_HYPOTHESES=_env_int("MARCRA_MAX_HYPOTHESES", 10),
        VHM_MARGIN=_env_float("VHM_MARGIN", 1.0),
        JUDGE_TARGET_FAILURES=_env_str("JUDGE_TARGET_FAILURES", "RF-01,RF-03,RF-04,RF-05"),
        JUDGE_USE_LLM=_env_bool("JUDGE_USE_LLM", False),
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY"),
        DETECTION_METHOD=_env_str("DETECTION_METHOD", "ensemble"),
        LOG_LEVEL=_env_str("LOG_LEVEL", "INFO"),
    )

