from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    ANOMALY_THRESHOLD: float = 0.20
    ANOMALY_THRESHOLD_MCP: Optional[float] = 0.20
    ANOMALY_THRESHOLD_A2A: Optional[float] = 0.20
    DETECTOR_DECISION_MODE: str = "rule_first"
    DETECTOR_SHADOW_MODE: bool = True

    MCP_HYBRID_ENABLED: bool = False
    MCP_HYBRID_PHASE: int = 1
    MCP_ML_KILLSWITCH: bool = False
    MCP_ML_SHADOW_MODE: bool = True
    MCP_ML_PROMOTION_THRESHOLD: float = 0.82
    MCP_ML_MODEL_NAME: str = "distilbert-base-uncased"

    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NORMAL: str = "normal_logs"
    STORE_DB_PATH: str = "./data/contextwatch.db"

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "contextwatch"

    LLM_API_KEY: str = ""
    LLM_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_PROVIDER: str = "anthropic"  # "anthropic" or "openai"
    OPENAI_API_KEY: str = ""
    LLM_MAX_TOKENS: int = 500

    BATCH_SIZE: int = 512
    MAX_LATENCY_MS: int = 500
    MIN_BASELINE_LOGS: int = 200
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_FILE_PER_MINUTE: int = 120
    AUTH_ENABLED: bool = False
    API_KEY: str = ""
    CORS_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    def validate_runtime_security(self) -> None:
        placeholder_values = {
            "",
            "change-me-strong-api-key",
            "change-me-strong-neo4j-password",
            "contextwatch-dev-key-change-me",
            "contextwatch",
        }

        if self.AUTH_ENABLED and self.API_KEY.strip() in placeholder_values:
            raise ValueError("AUTH_ENABLED is true but API_KEY is unset or placeholder.")

        if self.NEO4J_PASSWORD.strip() in placeholder_values:
            raise ValueError("NEO4J_PASSWORD is unset or placeholder. Set a strong password before startup.")

        if self.MCP_HYBRID_PHASE < 1 or self.MCP_HYBRID_PHASE > 4:
            raise ValueError("MCP_HYBRID_PHASE must be between 1 and 4.")

        if self.MCP_ML_PROMOTION_THRESHOLD < 0.0 or self.MCP_ML_PROMOTION_THRESHOLD > 1.0:
            raise ValueError("MCP_ML_PROMOTION_THRESHOLD must be between 0.0 and 1.0.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
