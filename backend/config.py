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
    gemini_api_key: str

    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-1.5-flash"

    # --- ChromaDB -------------------------------------------------------
    chroma_persist_path: str = "./backend/knowledge_base/chroma_db"

    chroma_collection_glec: str = "glec_framework"
    chroma_collection_epa: str = "epa_emission_factors"

    # --- RAG ------------------------------------------------------------
    rag_top_k: int = 5

    confidence_threshold: float = 0.75

    # --- External APIs --------------------------------------------------
    searoutes_api_key: str = ""

    # --- Supabase -------------------------------------------------------
    supabase_url: str
    supabase_key: str
    supabase_table_reports: str = "green_clearance_reports"

    # --- App ------------------------------------------------------------
    app_env: str = "development"   # development | production
    log_level: str = "INFO"
    poppler_path: str = ""

# NOTE: dung lru_cache de Settings chi duoc khoi tao mot lan
# trong suot vong doi cua app — tranh doc .env nhieu lan.
# Cach dung: from config import get_settings; s = get_settings()
@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]