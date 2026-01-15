"""
MCP tool registrations for resume matching and generation.
For questions in the chat window, the Next.js will connect to supabase vector DB
and complete the conversation.

This module defines tools for:
- Listing matched job skills from a job description.
- Generating an updated resume based on a job description.
- Downloading the latest resume without a job description.
- Checking resume cache status.
"""

from __future__ import annotations

import base64
import json
import logging

from fastmcp import FastMCP
from fastmcp.server import Context

from src.config import settings
from src.libs.document_parser import get_document_parser
from src.services import get_profile_service, get_resume_service
from src.libs.vector_database import EmbeddingModel, VectorDatabase
from src.libs.exceptions import FileUploadException

logger = logging.getLogger(__name__)
_vector_db: VectorDatabase | None = None


def _get_vector_database() -> VectorDatabase:
    global _vector_db
    if _vector_db is None:
        if not settings.supabase_url or not settings.supabase_key:
            raise ValueError("Supabase credentials are required")
        _vector_db = VectorDatabase(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_key,
            postgres_url=settings.supabase_postgres_url or "",
        )
    return _vector_db


def _get_embedding_model_by_name(name: str) -> EmbeddingModel | None:
    return _get_vector_database().get_embedding_model(name)


def _get_default_embedding_model(provider: str) -> EmbeddingModel | None:
    models = _get_vector_database().list_embedding_models(provider=provider)
    return models[0] if models else None


def _resolve_user_id(user_id: str | None) -> str:
    resolved = user_id or settings.author_user_id
    if not resolved:
        raise ValueError("user_id is required (set author_user_id or pass user_id)")
    return resolved


def _resolve_provider_and_embedding_model(
    provider: str | None,
    embedding_model_name: str | None,
) -> tuple[str, EmbeddingModel]:
    resolved_provider = (provider or settings.default_llm_provider or "").strip().lower()
    if not resolved_provider:
        raise ValueError("LLM provider is required")

    resolved_model_name = embedding_model_name
    if not resolved_model_name:
        if provider is not None:
            default_model = _get_default_embedding_model(resolved_provider)
            if not default_model:
                raise ValueError("No embedding model configured for the provider")
            resolved_model_name = default_model.name
        else:
            resolved_model_name = settings.default_embedding_model_name
            if not resolved_model_name:
                default_model = _get_default_embedding_model(resolved_provider)
                if not default_model:
                    raise ValueError("No embedding model configured for the provider")
                resolved_model_name = default_model.name

    model = _get_embedding_model_by_name(resolved_model_name)
    if not model:
        raise ValueError("Embedding model not found in Supabase")
    if model.provider != resolved_provider:
        raise ValueError("Embedding model provider does not match requested provider")

    return resolved_provider, model


async def _extract_job_description(
    input_data: str,
    input_type: str,
    filename: str | None,
) -> str:
    # init doc parser
    parser = get_document_parser()

    if input_type == "url":
        if not settings.allow_url_uploads:
            raise ValueError("URL uploads are not enabled")
        return await parser.parse(input_data, is_url=True)

    if input_type == "file":
        if not filename:
            raise ValueError("filename is required for file input")
        try:
            file_bytes = base64.b64decode(input_data)
        except Exception as exc:
            raise ValueError(f"Invalid base64 encoding: {exc}") from exc
        if len(file_bytes) > settings.max_upload_size:
            raise ValueError(
                f"File size exceeds maximum of {settings.max_upload_size} bytes"
            )
        file_type = parser.detect_file_type(filename)
        return await parser.parse(file_bytes, file_type, False)

    return input_data


def register_tools(mcp: FastMCP) -> None:
    resume_service = get_resume_service()

    @mcp.tool()
    async def list_matched_job_skills(
        input_data: str,
        input_type: str = "file",
        filename: str | None = None,
        user_id: str | None = None,
        top_k: int = 10,
        threshold: float = settings.min_similarity_threshold,
        provider: str | None = None,
        embedding_model_name: str | None = None,
        ctx: Context | None = None,
    ) -> str:
        """
        Parse a job description file and return matched chunks with similarity rates.
        Arguments:
        - input_data: Base64-encoded file content, URL, or raw text.
        - input_type: "file", "url", or "text".
        - filename: Original filename (required for "file" type).
        - user_id: User identifier for profile data.
        - top_k: Number of top matches to return.
        - threshold: Minimum similarity threshold (0.0 to 1.0).
        - provider: Optional LLM provider name (openai, google, ollama).
        - embedding_model_name: Optional embedding model name.
        - ctx: MCP Context for logging.
        Returns:
            JSON string with matched chunks and similarity rates.
        """
        try:
            if ctx:
                # sends info message tied to the current MCP request and send to MCP client.
                await ctx.info("Parsing job description for skill matching")

            job_description = await _extract_job_description(
                input_data=input_data,
                input_type=input_type,
                filename=filename,
            )

            if not job_description.strip():
                raise ValueError("Job description cannot be empty")

            resolved_user_id = _resolve_user_id(user_id)

            if ctx:
                await ctx.info("Running similarity search against profile data")

            resolved_provider, resolved_model = _resolve_provider_and_embedding_model(
                provider=provider,
                embedding_model_name=embedding_model_name,
            )
            try:
                profile_service = get_profile_service()
            except Exception as exc:
                raise ValueError(f"Profile service unavailable: {exc}") from exc

            matches = await profile_service.search_job_matches(
                job_description=job_description,
                user_id=resolved_user_id,
                top_k=top_k,
                threshold=threshold,
                embedding_model=resolved_model,
            )

            match_items = []
            similarity_total = 0.0
            for match in matches:
                similarity = float(match.get("similarity") or 0.0)
                similarity_total += similarity
                match_items.append(
                    {
                        "chunk_text": match.get("chunk_text") or "",
                        "similarity": similarity,
                        "match_rate_percent": int(round(similarity * 100)),
                        "title": match.get("title"),
                        "content_type": match.get("content_type"),
                        "metadata": match.get("metadata"),
                    }
                )

            match_rate = similarity_total / len(match_items) if match_items else 0.0

            return json.dumps(
                {
                    "status": "success",
                    "job_description_preview": job_description[:200]
                    + ("..." if len(job_description) > 200 else ""),
                    "match_rate": match_rate,
                    "match_rate_percent": int(round(match_rate * 100)),
                    "matches": match_items,
                    "total_matches": len(match_items),
                },
                indent=2,
                ensure_ascii=True,
            )
        except Exception as exc:
            if ctx:
                await ctx.error(f"Skill match failed: {exc}")
            logger.error("Skill match failed: %s", exc)
            raise FileUploadException(str(exc)) from exc

    @mcp.tool()
    async def generate_updated_resume(
        job_description: str,
        provider: str | None = None,
        embedding_model_name: str | None = None,
        top_k: int = settings.default_top_k,
        user_id: str | None = None,
        use_cache: bool = True,
        ctx: Context | None = None,
    ) -> str:
        """Generate an updated resume based on a job description."""
        try:
            if not job_description.strip():
                raise ValueError("Job description cannot be empty")

            if ctx:
                await ctx.info("Generating updated resume")

            resolved_provider, resolved_model = _resolve_provider_and_embedding_model(
                provider=provider,
                embedding_model_name=embedding_model_name,
            )
            result = await resume_service.generate_updated_resume(
                job_description=job_description,
                provider=resolved_provider,
                embedding_model=resolved_model,
                top_k=top_k,
                user_id=user_id,
                use_cache=use_cache,
            )

            return json.dumps(
                {
                    "status": "success",
                    "resume": result.get("resume", ""),
                    "match_summary": result.get("match_summary"),
                    "matches": result.get("matches"),
                    "cache_hit": result.get("cache_hit"),
                },
                indent=2,
                default=str,
                ensure_ascii=True,
            )
        except Exception as exc:
            if ctx:
                await ctx.error(f"Resume generation failed: {exc}")
            logger.error("Resume generation failed: %s", exc)
            return json.dumps(
                {"status": "error", "message": str(exc)},
                indent=2,
                ensure_ascii=True,
            )

    @mcp.tool()
    async def download_latest_resume(
        provider: str | None = None,
        user_id: str | None = None,
        use_cache: bool = True,
        ctx: Context | None = None,
    ) -> str:
        """Return the latest resume without a job description, using cache."""
        try:
            if ctx:
                await ctx.info("Fetching latest resume")

            resolved_provider = (provider or settings.default_llm_provider or "").strip().lower()
            if not resolved_provider:
                raise ValueError("LLM provider is required")
            result = await resume_service.generate_latest_resume(
                provider=resolved_provider,
                user_id=user_id,
                use_cache=use_cache,
            )

            return json.dumps(
                {
                    "status": "success",
                    "resume": result.get("resume", ""),
                    "match_summary": result.get("match_summary"),
                    "cache_hit": result.get("cache_hit"),
                },
                indent=2,
                default=str,
                ensure_ascii=True,
            )
        except Exception as exc:
            if ctx:
                await ctx.error(f"Resume download failed: {exc}")
            logger.error("Resume download failed: %s", exc)
            return json.dumps(
                {"status": "error", "message": str(exc)},
                indent=2,
                ensure_ascii=True,
            )

    @mcp.tool()
    async def resume_cache_status(ctx: Context | None = None) -> str:
        """Return resume cache statistics for debugging/health checks."""
        try:
            if ctx:
                await ctx.info("Fetching resume cache stats")

            stats = resume_service.resume_cache.stats()
            return json.dumps(
                {"status": "success", "cache": stats},
                indent=2,
                ensure_ascii=True,
            )
        except Exception as exc:
            if ctx:
                await ctx.error(f"Cache status failed: {exc}")
            logger.error("Cache status failed: %s", exc)
            return json.dumps(
                {"status": "error", "message": str(exc)},
                indent=2,
                ensure_ascii=True,
            )
