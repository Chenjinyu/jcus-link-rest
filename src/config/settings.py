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

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
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
    
    # LLM Service
    default_llm_provider: str = "ollama"  # openai, google, ollama. ollama uses local model
    openai_api_key: str | None = None
    google_api_key: str | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience function to get settings
settings = get_settings()
