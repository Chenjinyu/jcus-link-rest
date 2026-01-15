# src/config/settings.py
"""
Application configuration and settings.
BaseSettings automatically reads from (non-case-sensitive by default.)
1. OS environment variables (os.environ)
2. .env file (if env_file is configured)
3. Default values in code
as the order. 
AS LONG AS THE ENV EXISTS IN OS ENV, IT WILL BE USED, AND CANNOT
OVERWRITE BY THE .ENV FILE.
"""

from functools import lru_cache
import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Base application settings loaded from environment variables."""

    app_env: str = "development"

    # Application
    app_name: str = "jcus-link-rest"
    app_version: str = "1.0.0"
    debug: bool = False
    # Profile/Resume Data
    author_user_id: str | None = "jinyu.chen"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # CORS
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # MCP Protocol
    mcp_protocol_version: str = "2024-11-05"
    
    # Vector Database - Primary: Supabase
    vector_db_type: str = "supabase"  # supabase, chromadb
    supabase_url: str | None = None
    supabase_key: str | None = None
    supabase_postgres_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_POSTGRES_URL", "POSTGRES_URL_NON_POOLING"),
    )
    postgres_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_POSTGRES_USER", "POSTGRES_USER"),
    )
    # Vector Database - Alternative: ChromaDB
    chromadb_host: str | None = None
    chromadb_port: int | None = None
    chromadb_collection: str = "resumes"
    
    openai_api_key: str | None = None
    google_api_key: str | None = None
    default_llm_provider: str | None = None # openai, google, ollama
    default_embedding_model_name: str | None = None
    
    # LLM Models
    openai_model: str = "gpt-4o-mini"
    google_model: str = "gemini-1.5-flash"
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 180.0
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 24 * 60 * 60  # 24 hours
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Upload & Document Parsing
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list[str] = [".txt", ".pdf", ".doc", ".docx", ".html", ".md"]
    allow_url_uploads: bool = True  # Allow uploading job descriptions from URLs
    allowed_url_schemes: list[str] = ["http", "https"]
    
    # Document parsing timeouts
    url_fetch_timeout: int = 30  # seconds
    pdf_max_pages: int = 100
    
    # Resume Generation
    default_top_k: int = 5
    min_similarity_threshold: float = 0.5

    # Resume Cache
    resume_cache_ttl_seconds: int = 24 * 60 * 60
    resume_cache_max_entries: int = 200
    resume_cache_path: str = ".cache/resume_cache.json"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


class DevSettings(AppSettings):
    """Development defaults."""
    app_env: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    rate_limit_enabled: bool = False
    # LLM Service
    default_llm_provider: str = "ollama"  # openai, google, ollama. ollama uses local model
    default_embedding_model_name: str = "nomic-embed-text"

class ProdSettings(AppSettings):
    """Production defaults."""

    app_env: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    # LLM Service
    default_llm_provider: str = "google"  # openai, google, ollama. ollama uses local model
    default_embedding_model_name: str = "text-embedding-004"


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached settings instance."""
    app_env = os.environ.get("APP_ENV", "development").strip().lower()
    if app_env == "production":
        return ProdSettings()
    return DevSettings()


# Convenience function to get settings
settings = get_settings()
