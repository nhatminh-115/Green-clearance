from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- LLM ------------------------------------------------------------
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # --- ChromaDB -------------------------------------------------------
    chroma_persist_path: str = "./backend/knowledge_base/chroma_db"
    chroma_collection_glec: str = "glec_framework"
    chroma_collection_epa: str = "epa_emission_factors"

    # --- RAG ------------------------------------------------------------
    rag_top_k: int = 5
    confidence_threshold: float = 0.75

    # --- Supabase -------------------------------------------------------
    supabase_url: str
    supabase_key: str
    supabase_table_reports: str = "green_clearance_reports"

    # --- App ------------------------------------------------------------
    app_env: str = "development"
    log_level: str = "INFO"
    poppler_path: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]