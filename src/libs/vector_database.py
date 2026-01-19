# vector_database.py
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Union, Literal

import asyncpg
from asyncpg.pool import PoolConnectionProxy
from supabase import Client, create_client

@dataclass
class EmbeddingModel:
    id: str
    name: str
    provider: str
    model_identifier: str
    dimensions: int
    is_local: bool
    cost_per_token: float | None = None


class VectorDatabase:
    """
    Comprehensive vector database interface for managing documents,
    embeddings, articles, profile data, and personal attributes.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        postgres_url: str,
    ):
        self.supabase: Client = create_client(supabase_url, supabase_key)

        # PostgreSQL connection pool for transactions
        self.postgres_url = postgres_url
        self.pg_pool: asyncpg.Pool | None = None
        self.embedding_model: str | None = None

    async def init_pool(self):
        """Initialize PostgreSQL connection pool. Call this before using transaction methods."""
        if not self.pg_pool:
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                statement_cache_size=0,  # Disable prepared statement caching.
            )

    async def close_pool(self):
        """Close PostgreSQL connection pool."""
        if self.pg_pool:
            await self.pg_pool.close()
            self.pg_pool = None


    def _generate_searchable_text_from_profile_data(self, data: dict[str, Any]) -> str:
        """
        Generate searchable text from profile_data JSONB data.
        Replicates the logic from update_profile_data_searchable_text() trigger.

        Args:
            data: Profile data dictionary

        Returns:
            Searchable text string
        """
        text_parts = []

        if "job_title" in data:
            text_parts.append(str(data["job_title"]))

        if "company" in data:
            text_parts.append(str(data["company"]))

        if "positions" in data:
            text_parts.append(str(data["positions"]))

        if "description" in data:
            text_parts.append(str(data["description"]))

        if "skill_keywords" in data:
            skills = data["skill_keywords"]
            if isinstance(skills, list):
                text_parts.append("Skills: " + ", ".join(map(str, skills)))
            else:
                text_parts.append("Skills: " + str(skills))

        if "achievements" in data:
            achievements = data["achievements"]
            if isinstance(achievements, list):
                text_parts.append(". ".join(map(str, achievements)))
            else:
                text_parts.append(str(achievements))

        if "responsibilities" in data:
            responsibilities = data["responsibilities"]
            if isinstance(responsibilities, list):
                text_parts.append(". ".join(map(str, responsibilities)))
            else:
                text_parts.append(str(responsibilities))

        # Add other common fields
        for key in ["institution", "degree", "field_of_study", "name", "level"]:
            if key in data:
                text_parts.append(str(data[key]))

        return ". ".join(text_parts) if text_parts else ""

    def _load_models(self) -> None:
        """Load embedding models from database."""
        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _safe_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        result = (
            self.supabase.table("embedding_models")
            .select("*")
            .eq("is_active", True)
            .execute()
        )
        for model_data in result.data or []:
            if not isinstance(model_data, dict):
                continue
            model_name = str(model_data.get("name", ""))
            provider = str(model_data.get("provider", ""))
            if provider not in ["openai", "ollama", "google"]:
                raise ValueError(f"Unsupported provider: {provider}")

            model_id = str(model_data.get("id", ""))
            self._models_cache[model_name] = EmbeddingModel(
                id=model_id,
                name=model_name,
                provider=provider,
                model_identifier=str(model_data.get("model_identifier", "")),
                dimensions=_safe_int(model_data.get("dimensions", 0)),
                is_local=bool(model_data.get("is_local", False)),
                cost_per_token=_safe_float(model_data.get("cost_per_token")),
            )

    def list_embedding_models(self, provider: str | None = None) -> List[EmbeddingModel]:
        """Return active embedding models, optionally filtered by provider."""
        if provider is None:
            raise ValueError("provider is required")
        
        normalized = provider.strip().lower()
        return [
            model
            for model in self._models_cache.values()
            if model.provider == normalized
        ]

    def get_embedding_model(self, name: str) -> EmbeddingModel | None:
        """Get an embedding model by name."""
        return self._models_cache.get(name)

    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================

    @staticmethod
    def _embedding_to_vector_str(embedding: List[float]) -> str:
        return "[" + ",".join(map(str, embedding)) + "]"


    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """Get document by ID"""
        result = (
            self.supabase.table("documents").select("*").eq("id", document_id).execute()
        )
        # replace `return result.data[0] if result.data else None` by below
        # to fix static type check with Pylance
        data = result.data
        if not data:
            return None
        first = data[0]
        return first if isinstance(first, dict) else None

    def get_documents(
        self,
        user_id: str,
        tags: List[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """Get documents with optional filtering"""
        query = (
            self.supabase.table("documents")
            .select("*")
            .eq("user_id", user_id)
            .eq("is_current", True)
            .is_("deleted_at", "null")
        )

        if tags:
            query = query.overlaps("tags", tags)

        result = (
            query.order("created_at", desc=True)
            .limit(limit)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data # type: ignore

    # ========================================================================
    # ARTICLE OPERATIONS
    # ========================================================================

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        """Get article by ID"""
        result = (
            self.supabase.table("articles").select("*").eq("id", article_id).execute()
        )
        data = result.data
        if not data:
            return None
        first = data[0]
        return first if isinstance(first, dict) else None

    def get_article_by_slug(self, slug: str) -> dict[str, Any] | None:
        """Get article by slug"""
        result = self.supabase.table("articles").select("*").eq("slug", slug).execute()
        data = result.data
        if not data:
            return None
        first = data[0]
        return first if isinstance(first, dict) else None

    def get_articles(
        self,
        user_id: str,
        status: str | None = None,
        category: str | None = None,
        tags: List[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """Get articles with optional filtering"""
        query = self.supabase.table("articles").select("*").eq("user_id", user_id)

        if status:
            query = query.eq("status", status)
        if category:
            query = query.eq("category", category)
        if tags:
            query = query.overlaps("tags", tags)

        result = (
            query.order("created_at", desc=True)
            .limit(limit)
            .range(offset, offset + limit - 1)
            .execute()
        )
        data = result.data
        if not data:
            return []
        return [item for item in data if isinstance(item, dict)]

    # ========================================================================
    # PROFILE DATA OPERATIONS
    # ========================================================================

    def get_profile_data(self, profile_id: str) -> dict[str, Any] | None:
        """Get profile data by ID"""
        result = (
            self.supabase.table("profile_data")
            .select("*")
            .eq("id", profile_id)
            .execute()
        )
        data = result.data
        if not data:
            return None
        first = data[0]
        return first if isinstance(first, dict) else None

    def get_profile_data_list(
        self,
        user_id: str,
        category: str | None = None,
        is_current: bool | None = None,
        is_featured: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """Get profile data with optional filtering"""
        query = self.supabase.table("profile_data").select("*").eq("user_id", user_id)

        if category:
            query = query.eq("category", category)
        if is_current is not None:
            query = query.eq("is_current", is_current)
        if is_featured is not None:
            query = query.eq("is_featured", is_featured)

        result = (
            query.order("display_order", desc=False)
            .order("created_at", desc=True)
            .limit(limit)
            .range(offset, offset + limit - 1)
            .execute()
        )
        data = result.data
        if not data:
            return []
        return [item for item in data if isinstance(item, dict)]

    # ========================================================================
    # PERSONAL ATTRIBUTES OPERATIONS
    # ========================================================================

    def get_personal_attribute(self, attribute_id: str) -> dict[str, Any] | None:
        """Get personal attribute by ID"""
        result = (
            self.supabase.table("personal_attributes")
            .select("*")
            .eq("id", attribute_id)
            .execute()
        )
        data = result.data
        if not data:
            return None
        first = data[0]
        return first if isinstance(first, dict) else None

    def get_personal_attributes(
        self,
        user_id: str,
        attribute_type: str | None = None,
        min_importance: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict[str, Any]]:
        """Get personal attributes with optional filtering"""
        query = (
            self.supabase.table("personal_attributes")
            .select("*")
            .eq("user_id", user_id)
        )

        if attribute_type:
            query = query.eq("attribute_type", attribute_type)
        if min_importance is not None:
            query = query.gte("importance_score", min_importance)

        result = (
            query.order("importance_score", desc=True)
            .order("created_at", desc=True)
            .limit(limit)
            .range(offset, offset + limit - 1)
            .execute()
        )
        data = result.data
        if not data:
            return []
        return [item for item in data if isinstance(item, dict)]

    # ========================================================================
    # SEARCH OPERATIONS (Using SQL Functions)
    # ========================================================================

    async def search_rpc_function(
        self,
        query: str,
        user_id: str,
        content_types: List[str] | None = None,
        tags: List[str] | None = None,
        threshold: float = 0.7,
        limit: int = 10,
        model_name: str | None = None,
        query_embedding: List[float] | None = None,
    ) -> List[dict[str, Any]]:
        """
        Search across all content using vector similarity.
        Uses the search_documents SQL function via Supabase RPC.

        Returns:
            List of Dict with fields including:
            - document_id, article_id, profile_data_id, personal_attribute_id
            - title, content, chunk_text
            - content_type, category, attribute_type
            - similarity (FLOAT): Cosine similarity score (0-1, higher = more similar)
            - metadata, tags, created_at

        Note: Uses Supabase client (single operation) - no transaction needed.
        For multi-operation atomicity, use asyncpg pool with transactions.
        """
        if model_name is None:
            raise ValueError("model_name is required for search_rpc_function")

        if query_embedding is None:
            raise ValueError("query_embedding is required for search_rpc_function")

        model = self._models_cache.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found")

        # Supabase RPC accepts List[float] directly - it handles conversion to vector type
        results = self.supabase.rpc(
            "search_documents",
            {
                "query_embedding": query_embedding,  # List[float] - Supabase converts to vector
                "model_id": model.id,
                "match_threshold": threshold,
                "match_count": limit,
                "filter_user_id": user_id,
                "filter_content_types": content_types,
                "filter_tags": tags,
            },
        ).execute()

        # Results include 'similarity' field from SQL function
        data = results.data
        if not data:
            return []
        return [item for item in data if isinstance(item, dict)] # type: ignore

    async def search_all_similar_content_rpc_function(
        self,
        query: str,
        user_id: str,
        threshold: float = 0.7,
        limit: int = 10,
        model_name: str | None = None,
        query_embedding: List[float] | None = None,
    ) -> List[dict[str, Any]]:
        """
        Simplified search across all content.
        Uses the search_similar_content SQL function.
        """
        if model_name is None:
            raise ValueError(
                "model_name is required for search_all_similar_content_rpc_function"
            )

        if query_embedding is None:
            raise ValueError(
                "query_embedding is required for search_all_similar_content_rpc_function"
            )

        results = self.supabase.rpc(
            "search_similar_content",
            {
                "query_embedding": query_embedding,
                "user_id_filter": user_id,
                "match_threshold": threshold,
                "match_count": limit,
            },
        ).execute()

        data = results.data
        if not data:
            return []
        return [item for item in data if isinstance(item, dict)] # type: ignore
