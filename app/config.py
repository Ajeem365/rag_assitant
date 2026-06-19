
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    groq_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0

    # Vector Store
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG behaviour
    top_k_retrieval: int = 5
    max_retry_count: int = 2

    # Bonus: web search
    tavily_api_key: str = ""
    enable_web_fallback: bool = False

    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — avoids re-reading .env on every call."""
    return Settings()