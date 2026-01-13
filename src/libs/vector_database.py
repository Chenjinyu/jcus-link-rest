# vector_database.py
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Union

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

        self._models_cache: dict[str, EmbeddingModel] = {}
        self._models_by_id: dict[str, EmbeddingModel] = {}
        self._load_models()

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
            # Fix profile_data trigger if needed (disable searchable_text assignment)
            await self._fix_profile_data_trigger()

    async def close_pool(self):
        """Close PostgreSQL connection pool."""
        if self.pg_pool:
            await self.pg_pool.close()
            self.pg_pool = None

    async def _fix_profile_data_trigger(self):
        """
        Fix profile_data trigger to not set searchable_text (column doesn't exist).
        This is a one-time fix that runs when pool is initialized.
        Note: DDL statements (DROP, CREATE) don't require a transaction.
        """
        if not self.pg_pool:
            return  # Pool not initialized, skip trigger fix

        try:
            # Acquire a connection from the pool (DDL statements auto-commit)
            async with self.pg_pool.acquire() as conn:
                # Drop and recreate trigger function as no-op
                await conn.execute("""
                    DROP TRIGGER IF EXISTS update_profile_data_searchable_text_trigger ON profile_data;
                    DROP FUNCTION IF EXISTS update_profile_data_searchable_text();
                    
                    CREATE OR REPLACE FUNCTION update_profile_data_searchable_text()
                    RETURNS TRIGGER
                    LANGUAGE plpgsql
                    SET search_path = public
                    AS $$
                    BEGIN
                      -- No-op: searchable_text is generated in Python, not by trigger
                      RETURN NEW;
                    END;
                    $$;
                    
                    CREATE TRIGGER update_profile_data_searchable_text_trigger
                      BEFORE INSERT OR UPDATE ON profile_data
                      FOR EACH ROW
                      EXECUTE FUNCTION update_profile_data_searchable_text();
                """)
        except Exception as e:
            # If trigger fix fails, log but don't fail initialization
            print(f"Warning: Could not fix profile_data trigger: {e}")
            print(
                "You may need to run the SQL migration manually: db_migration/supabase/fix_profile_data_trigger.sql"
            )

    def _parse_date(self, date_value: Union[str, date] | None) -> date | None:
        """
        Convert string date to Python date object for asyncpg.

        Args:
            date_value: String in 'YYYY-MM-DD' format or date object or None

        Returns:
            date object or None
        """
        if date_value is None:
            return None
        if isinstance(date_value, date):
            return date_value
        if isinstance(date_value, str):
            # Parse 'YYYY-MM-DD' format
            try:
                return datetime.strptime(date_value, "%Y-%m-%d").date()
            except ValueError:
                # Try other common formats
                try:
                    return datetime.strptime(date_value, "%Y/%m/%d").date()
                except ValueError:
                    raise ValueError(
                        f"Invalid date format: {date_value}. Expected 'YYYY-MM-DD' or 'YYYY/MM/DD'"
                    )
                    
        raise TypeError(
            f"date_value must be str, date, or None, got {type(date_value)}"
        )

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

        if "title" in data:
            text_parts.append(str(data["title"]))

        if "company" in data:
            text_parts.append(str(data["company"]))

        if "position" in data:
            text_parts.append(str(data["position"]))

        if "description" in data:
            text_parts.append(str(data["description"]))

        if "skills" in data:
            skills = data["skills"]
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
            self._models_by_id[model_id] = self._models_cache[
                model_name
            ]
        print("--" * 20)
        print(self._models_cache)

    def list_embedding_models(self, provider: str | None = None) -> List[EmbeddingModel]:
        """Return active embedding models, optionally filtered by provider."""
        if provider is None:
            return list(self._models_cache.values())
        normalized = provider.strip().lower()
        return [
            model
            for model in self._models_cache.values()
            if model.provider == normalized
        ]

    def get_embedding_model(self, name: str) -> EmbeddingModel | None:
        """Get an embedding model by name."""
        return self._models_cache.get(name)

    def get_embedding_model_by_id(self, model_id: str) -> EmbeddingModel | None:
        """Get an embedding model by ID."""
        return self._models_by_id.get(model_id)

    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================

    @staticmethod
    def _embedding_to_vector_str(embedding: List[float]) -> str:
        return "[" + ",".join(map(str, embedding)) + "]"

    async def _insert_embeddings_for_models(
        self,
        conn: asyncpg.Connection | PoolConnectionProxy,
        document_id: str,
        chunks: List[str],
        model_names: List[str],
        embeddings_by_model: dict[str, List[List[float]]],
    ) -> None:
        if not chunks:
            return
        total_chunks = len(chunks)
        for model_name in model_names:
            model = self._models_cache.get(model_name)
            if not model:
                raise ValueError(f"Model {model_name} not found or not active")
            embeddings = embeddings_by_model.get(model_name)
            if embeddings is None:
                raise ValueError(f"Missing embeddings for model {model_name}")
            if len(embeddings) != total_chunks:
                raise ValueError(
                    f"Embeddings count mismatch for {model_name}: "
                    f"expected {total_chunks}, got {len(embeddings)}"
                )

            for chunk_index, chunk_text in enumerate(chunks):
                embedding_str = self._embedding_to_vector_str(embeddings[chunk_index])

                await conn.execute(
                    """
                    INSERT INTO embeddings
                    (document_id, embedding_model_id, embedding, chunk_text, chunk_index, total_chunks)
                    VALUES ($1, $2, $3::vector, $4, $5, $6)
                    """,
                    document_id,
                    model.id,
                    embedding_str,
                    chunk_text,
                    chunk_index,
                    total_chunks,
                )

    def _get_model_names_from_ids(self, model_ids: List[str]) -> List[str]:
        model_names = []
        for model_id in model_ids:
            model = self._models_by_id.get(str(model_id))
            if model:
                model_names.append(model.name)
        return model_names

    async def add_document(
        self,
        user_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: List[str] | None = None,
        model_names: List[str] | None = None,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> str:
        """
        Add document with embeddings from multiple models using database transaction.
        Automatically chunks long content.
        If any error occurs, entire operation is rolled back.
        """
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        if model_names is None:
            model_names = list(self._models_cache.keys())

        if embeddings_by_model is None:
            raise ValueError("embeddings_by_model is required to add a document")
        if chunks is None:
            raise ValueError("chunks is required to add a document")

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_id = await conn.fetchval(
                    """
                    INSERT INTO documents (user_id, title, content, metadata, tags)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    user_id,
                    title,
                    content,
                    json.dumps(metadata or {}),
                    tags or [],
                )

                # Create and insert embeddings for each model
                await self._insert_embeddings_for_models(
                    conn, document_id, chunks, model_names, embeddings_by_model
                )

        return document_id

    async def update_document(
        self,
        document_id: str,
        title: str | None = None,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: List[str] | None = None,
        recreate_embeddings: bool = True,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> bool:
        """Update document and optionally recreate embeddings using database transaction"""
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Build update query dynamically
                update_parts = []
                params = []
                param_count = 1

                if title is not None:
                    update_parts.append(f"title = ${param_count}")
                    params.append(title)
                    param_count += 1

                if content is not None:
                    update_parts.append(f"content = ${param_count}")
                    params.append(content)
                    param_count += 1

                if metadata is not None:
                    update_parts.append(f"metadata = ${param_count}")
                    params.append(json.dumps(metadata))
                    param_count += 1

                if tags is not None:
                    update_parts.append(f"tags = ${param_count}")
                    params.append(tags)
                    param_count += 1

                if update_parts:
                    update_parts.append("updated_at = NOW()")
                    params.append(document_id)

                    await conn.execute(
                        f"""
                        UPDATE documents
                        SET {", ".join(update_parts)}
                        WHERE id = ${param_count}
                        """,
                        *params,
                    )

                # Recreate embeddings if content changed
                if recreate_embeddings and content is not None:
                    if embeddings_by_model is None:
                        raise ValueError(
                            "embeddings_by_model is required to recreate embeddings"
                        )
                    if chunks is None:
                        raise ValueError("chunks is required to recreate embeddings")
                    # Get existing embedding model IDs
                    existing = await conn.fetch(
                        "SELECT DISTINCT embedding_model_id FROM embeddings WHERE document_id = $1",
                        document_id,
                    )
                    model_ids = [row["embedding_model_id"] for row in existing]

                    # Delete old embeddings
                    await conn.execute(
                        "DELETE FROM embeddings WHERE document_id = $1", document_id
                    )

                    model_names = self._get_model_names_from_ids(model_ids)
                    if model_names:
                        await self._insert_embeddings_for_models(
                            conn,
                            document_id,
                            chunks,
                            model_names,
                            embeddings_by_model,
                        )

        return True

    def delete_document(self, document_id: str, soft_delete: bool = True) -> bool:
        """Delete document (soft or hard delete)"""
        if soft_delete:
            self.supabase.table("documents").update(
                {"deleted_at": datetime.now().isoformat(), "is_current": False}
            ).eq("id", document_id).execute()
        else:
            self.supabase.table("documents").delete().eq("id", document_id).execute()

        return True

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
    # SQL FUNCTION: upsert_document_with_embedding
    # ========================================================================

    async def upsert_document_with_embedding_rpc_function(
        self,
        user_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tags: List[str] | None = None,
        model_name: str | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Call SQL function to upsert document with embeddings.

        Args:
            chunks: List of dicts with 'text', 'embedding', 'chunk_index'
        """
        if model_name is None:
            raise ValueError("model_name is required for upsert_document_with_embedding_rpc_function")

        chunks_json = chunks or []

        result = self.supabase.rpc(
            "upsert_document_with_embedding",
            {
                "p_user_id": user_id,
                "p_title": title,
                "p_content": content,
                "p_metadata": metadata or {},
                "p_tags": tags or [],
                "p_embedding_model_name": model_name,
                "p_chunks": chunks_json,
            },
        ).execute()
        return result.data # type: ignore

    # ========================================================================
    # ARTICLE OPERATIONS
    # ========================================================================

    async def add_article(
        self,
        user_id: str,
        title: str,
        content: str,
        subtitle: str | None = None,
        excerpt: str | None = None,
        tags: List[str] | None = None,
        category: str | None = None,
        status: str = "draft",
        seo_title: str | None = None,
        seo_description: str | None = None,
        og_image: str | None = None,
        model_names: List[str] | None = None,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> dict[str, Any]:
        """Add article with document and embeddings using database transaction"""
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        if model_names is None:
            model_names = list(self._models_cache.keys())

        if embeddings_by_model is None:
            raise ValueError("embeddings_by_model is required to add an article")
        if chunks is None:
            raise ValueError("chunks is required to add an article")

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Generate unique slug (check for duplicates within transaction)
                slug = await self._create_unique_slug(title, conn, user_id)

                # Insert document
                document_id = await conn.fetchval(
                    """
                    INSERT INTO documents (user_id, title, content, metadata, tags)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    user_id,
                    title,
                    content,
                    json.dumps({"source": "article", "category": category}),
                    tags or [],
                )

                # Insert article
                article_id = await conn.fetchval("""
                    INSERT INTO articles
                    (user_id, document_id, title, subtitle, content, excerpt, tags, category,
                    status, slug, seo_title, seo_description, og_image, published_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    RETURNING id
                    """,
                    user_id,
                    document_id,
                    title,
                    subtitle,
                    content,
                    excerpt or content[:200],
                    tags or [],
                    category,
                    status,
                    slug,
                    seo_title,
                    seo_description,
                    og_image,
                    datetime.now() if status == "published" else None,
                )

                # Create embeddings
                await self._insert_embeddings_for_models(
                    conn, document_id, chunks, model_names, embeddings_by_model
                )

        return {"article_id": article_id, "document_id": document_id}

    async def update_article(
        self,
        article_id: str,
        title: str | None = None,
        content: str | None = None,
        subtitle: str | None = None,
        excerpt: str | None = None,
        tags: List[str] | None = None,
        category: str | None = None,
        status: str | None = None,
        seo_title: str | None = None,
        seo_description: str | None = None,
        og_image: str | None = None,
        recreate_embeddings: bool = True,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> bool:
        """Update article and optionally recreate embeddings using database transaction"""
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Get document_id
                result = await conn.fetchrow(
                    "SELECT document_id FROM articles WHERE id = $1", article_id
                )
                if not result:
                    raise ValueError(f"Article {article_id} not found")

                document_id = result["document_id"]
                # Update article
                article_update_parts = []
                article_params = []
                param_count = 1

                if title is not None:
                    article_update_parts.append(f"title = ${param_count}")
                    article_params.append(title)
                    param_count += 1

                if content is not None:
                    article_update_parts.append(f"content = ${param_count}")
                    article_params.append(content)
                    param_count += 1

                if subtitle is not None:
                    article_update_parts.append(f"subtitle = ${param_count}")
                    article_params.append(subtitle)
                    param_count += 1

                if excerpt is not None:
                    article_update_parts.append(f"excerpt = ${param_count}")
                    article_params.append(excerpt)
                    param_count += 1

                if tags is not None:
                    article_update_parts.append(f"tags = ${param_count}")
                    article_params.append(tags)
                    param_count += 1

                if category is not None:
                    article_update_parts.append(f"category = ${param_count}")
                    article_params.append(category)
                    param_count += 1

                if status is not None:
                    article_update_parts.append(f"status = ${param_count}")
                    article_params.append(status)
                    param_count += 1
                    if status == "published":
                        article_update_parts.append(f"published_at = ${param_count}")
                        article_params.append(datetime.now())
                        param_count += 1

                if seo_title is not None:
                    article_update_parts.append(f"seo_title = ${param_count}")
                    article_params.append(seo_title)
                    param_count += 1

                if seo_description is not None:
                    article_update_parts.append(f"seo_description = ${param_count}")
                    article_params.append(seo_description)
                    param_count += 1

                if og_image is not None:
                    article_update_parts.append(f"og_image = ${param_count}")
                    article_params.append(og_image)
                    param_count += 1

                if article_update_parts:
                    article_update_parts.append("updated_at = NOW()")
                    article_params.append(article_id)

                    await conn.execute(
                        f"""
                        UPDATE articles
                        SET {", ".join(article_update_parts)}
                        WHERE id = ${param_count}
                        """,
                        *article_params,
                    )

                # Update document
                doc_update_parts = []
                doc_params = []
                param_count = 1

                if title is not None:
                    doc_update_parts.append(f"title = ${param_count}")
                    doc_params.append(title)
                    param_count += 1

                if content is not None:
                    doc_update_parts.append(f"content = ${param_count}")
                    doc_params.append(content)
                    param_count += 1

                if tags is not None:
                    doc_update_parts.append(f"tags = ${param_count}")
                    doc_params.append(tags)
                    param_count += 1

                if doc_update_parts:
                    doc_update_parts.append("updated_at = NOW()")
                    doc_params.append(document_id)

                    await conn.execute(
                        f"""
                        UPDATE documents
                        SET {", ".join(doc_update_parts)}
                        WHERE id = ${param_count}
                        """,
                        *doc_params,
                    )

                # Recreate embeddings if needed
                if recreate_embeddings and content is not None:
                    if embeddings_by_model is None:
                        raise ValueError(
                            "embeddings_by_model is required to recreate embeddings"
                        )
                    if chunks is None:
                        raise ValueError("chunks is required to recreate embeddings")
                    existing = await conn.fetch(
                        "SELECT DISTINCT embedding_model_id FROM embeddings WHERE document_id = $1",
                        document_id,
                    )
                    model_ids = [row["embedding_model_id"] for row in existing]

                    await conn.execute(
                        "DELETE FROM embeddings WHERE document_id = $1", document_id
                    )

                    model_names = self._get_model_names_from_ids(model_ids)
                    if model_names:
                        await self._insert_embeddings_for_models(
                            conn,
                            document_id,
                            chunks,
                            model_names,
                            embeddings_by_model,
                        )

        return True

    def delete_article(self, article_id: str, soft_delete: bool = True) -> bool:
        """Delete article and associated document"""
        article = (
            self.supabase.table("articles")
            .select("document_id")
            .eq("id", article_id)
            .single()
            .execute()
        )
        article_data = article.data if isinstance(article.data, dict) else None
        document_id = article.data.get("document_id", "") if article_data else None # type: ignore
        if document_id:
            self.delete_document(str(document_id), soft_delete=soft_delete)

        return True

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

    def increment_article_views(self, article_id: str) -> bool:
        """Increment article view count"""
        article = self.get_article(article_id)
        if article:
            self.supabase.table("articles").update(
                {"view_count": article.get("view_count", 0) + 1}
            ).eq("id", article_id).execute()
            return True
        return False

    def increment_article_likes(self, article_id: str) -> bool:
        """Increment article like count"""
        article = self.get_article(article_id)
        if article:
            self.supabase.table("articles").update(
                {"like_count": article.get("like_count", 0) + 1}
            ).eq("id", article_id).execute()
            return True
        return False

    # ========================================================================
    # PROFILE DATA OPERATIONS
    # ========================================================================

    async def add_profile_data(
        self,
        user_id: str,
        category: Literal[
            "work_experience",
            "education",
            "certification",
            "skill",
            "value",
            "goal",
            "project",
            "volunteering",
            "event",
        ],
        data: dict[str, Any],
        searchable: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
        is_current: bool = False,
        is_featured: bool = False,
        display_order: int | None = None,
        model_names: List[str] | None = None,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> dict[str, Any]:
        """Add profile data with optional document/embedding using database transaction"""
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        if model_names is None:
            model_names = list(self._models_cache.keys())

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Insert profile data
                # Convert string dates to date objects for asyncpg
                parsed_start_date = self._parse_date(
                    start_date or data.get("start_date")
                )
                parsed_end_date = self._parse_date(end_date or data.get("end_date"))

                profile_id = await conn.fetchval(
                    """
                    INSERT INTO profile_data
                    (user_id, category, data, start_date, end_date, is_current, is_featured, display_order)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                    """,
                    user_id,
                    category,
                    json.dumps(data),
                    parsed_start_date,
                    parsed_end_date,
                    is_current or data.get("current", False),
                    is_featured,
                    display_order,
                )

                document_id = None

                if searchable:
                    if embeddings_by_model is None:
                        raise ValueError(
                            "embeddings_by_model is required to add searchable profile data"
                        )
                    if chunks is None:
                        raise ValueError(
                            "chunks is required to add searchable profile data"
                        )
                    # Generate searchable_text in Python (profile_data table doesn't have searchable_text column)
                    # This replicates the trigger logic from update_profile_data_searchable_text()
                    searchable_text = self._generate_searchable_text_from_profile_data(
                        data
                    )

                    # Insert document
                    document_id = await conn.fetchval(
                        """
                        INSERT INTO documents (user_id, title, content, metadata, tags)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id
                        """,
                        user_id,
                        data.get("title", category),
                        searchable_text,
                        json.dumps(
                            {
                                "source": "profile_data",
                                "category": category,
                                "profile_data": data,
                            }
                        ),
                        [category],
                    )

                    # Create embeddings
                    await self._insert_embeddings_for_models(
                        conn, document_id, chunks, model_names, embeddings_by_model
                    )

                    # Update profile_data with document_id
                    await conn.execute(
                        "UPDATE profile_data SET document_id = $1 WHERE id = $2",
                        document_id,
                        profile_id,
                    )

                return {"profile_id": profile_id, "document_id": document_id}

    async def update_profile_data(
        self,
        profile_id: str,
        data: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        is_current: bool | None = None,
        is_featured: bool | None = None,
        display_order: int | None = None,
        recreate_embeddings: bool = True,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> bool:
        """Update profile data and optionally recreate embeddings using database transaction"""
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Get current profile data
                profile = await conn.fetchrow(
                    "SELECT data, document_id FROM profile_data WHERE id = $1",
                    profile_id,
                )

                if not profile:
                    raise ValueError(f"Profile data {profile_id} not found")

                current_data = (
                    json.loads(profile["data"])
                    if isinstance(profile["data"], str)
                    else profile["data"]
                )
                document_id = profile["document_id"]

                # Build update query
                update_parts = []
                params = []
                param_count = 1

                if data:
                    updated_data = {**current_data, **data}
                    update_parts.append(f"data = ${param_count}")
                    params.append(json.dumps(updated_data))
                    param_count += 1

                if start_date is not None:
                    update_parts.append(f"start_date = ${param_count}")
                    params.append(self._parse_date(start_date))
                    param_count += 1

                if end_date is not None:
                    update_parts.append(f"end_date = ${param_count}")
                    params.append(self._parse_date(end_date))
                    param_count += 1

                if is_current is not None:
                    update_parts.append(f"is_current = ${param_count}")
                    params.append(is_current)
                    param_count += 1

                if is_featured is not None:
                    update_parts.append(f"is_featured = ${param_count}")
                    params.append(is_featured)
                    param_count += 1

                if display_order is not None:
                    update_parts.append(f"display_order = ${param_count}")
                    params.append(display_order)
                    param_count += 1

                if update_parts:
                    update_parts.append(f"updated_at = NOW()")
                    params.append(profile_id)

                    await conn.execute(
                        f"""
                        UPDATE profile_data
                        SET {", ".join(update_parts)}
                        WHERE id = ${param_count}
                        """,
                        *params,
                    )

                # Recreate embeddings if needed
                if document_id and recreate_embeddings and data:
                    if embeddings_by_model is None:
                        raise ValueError(
                            "embeddings_by_model is required to recreate embeddings"
                        )
                    if chunks is None:
                        raise ValueError("chunks is required to recreate embeddings")
                    # Get updated data and generate searchable_text in Python
                    updated_profile = await conn.fetchrow(
                        "SELECT data FROM profile_data WHERE id = $1", profile_id
                    )
                    updated_data = (
                        json.loads(updated_profile["data"])
                        if isinstance(updated_profile["data"], str)
                        else updated_profile["data"]
                    )
                    searchable_text = self._generate_searchable_text_from_profile_data(
                        updated_data
                    )

                    # Update document content
                    await conn.execute(
                        "UPDATE documents SET content = $1, updated_at = NOW() WHERE id = $2",
                        searchable_text,
                        document_id,
                    )

                    # Get existing embedding model IDs
                    existing = await conn.fetch(
                        "SELECT DISTINCT embedding_model_id FROM embeddings WHERE document_id = $1",
                        document_id,
                    )
                    model_ids = [row["embedding_model_id"] for row in existing]

                    # Delete old embeddings
                    await conn.execute(
                        "DELETE FROM embeddings WHERE document_id = $1", document_id
                    )

                    model_names = self._get_model_names_from_ids(model_ids)
                    if model_names:
                        await self._insert_embeddings_for_models(
                            conn,
                            document_id,
                            chunks,
                            model_names,
                            embeddings_by_model,
                        )

        return True

    def delete_profile_data(self, profile_id: str) -> bool:
        """Delete profile data"""
        profile = (
            self.supabase.table("profile_data")
            .select("document_id")
            .eq("id", profile_id)
            .single()
            .execute()
        )

        if profile.data and profile.data["document_id"]: # type: ignore
            self.delete_document(profile.data["document_id"], soft_delete=False) # type: ignore
        else:
            self.supabase.table("profile_data").delete().eq("id", profile_id).execute()

        return True

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

    async def add_personal_attribute(
        self,
        user_id: str,
        attribute_type: Literal[
            "soft_skill", "value", "worldview", "aspiration", "principle"
        ],
        title: str,
        description: str,
        examples: List[str] | None = None,
        importance_score: int | None = None,
        confidence_level: int | None = None,
        related_articles: List[str] | None = None,
        related_experiences: List[str] | None = None,
        searchable: bool = True,
        model_names: List[str] | None = None,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> dict[str, Any]:
        """Add personal attribute with optional document/embedding using database transaction"""
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        if model_names is None:
            model_names = list(self._models_cache.keys())

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Insert personal attribute
                attribute_id = await conn.fetchval(
                    """
                    INSERT INTO personal_attributes
                    (user_id, attribute_type, title, description, examples, importance_score,
                     confidence_level, related_articles, related_experiences)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                    """,
                    user_id,
                    attribute_type,
                    title,
                    description,
                    examples or [],
                    importance_score,
                    confidence_level,
                    related_articles or [],
                    related_experiences or [],
                )

                document_id = None

                if searchable:
                    if embeddings_by_model is None:
                        raise ValueError(
                            "embeddings_by_model is required to add searchable personal attributes"
                        )
                    if chunks is None:
                        raise ValueError(
                            "chunks is required to add searchable personal attributes"
                        )
                    # the SQL function: update_personal_attributes_searchable_text_trigger will be triggered before insert or update on personal_attributes table
                    # Get searchable_text
                    searchable_text_row = await conn.fetchrow(
                        "SELECT searchable_text FROM personal_attributes WHERE id = $1",
                        attribute_id,
                    )
                    searchable_text = searchable_text_row["searchable_text"]

                    # Insert document
                    document_id = await conn.fetchval(
                        """
                        INSERT INTO documents (user_id, title, content, metadata, tags)
                        VALUES ($1, $2, $3, $4, $5)
                        RETURNING id
                        """,
                        user_id,
                        title,
                        searchable_text,
                        json.dumps(
                            {
                                "source": "personal_attribute",
                                "attribute_type": attribute_type,
                                "importance_score": importance_score,
                                "confidence_level": confidence_level,
                            }
                        ),
                        [attribute_type],
                    )

                    # Create embeddings
                    await self._insert_embeddings_for_models(
                        conn, document_id, chunks, model_names, embeddings_by_model
                    )

                    # Update personal_attributes with document_id
                    await conn.execute(
                        "UPDATE personal_attributes SET document_id = $1 WHERE id = $2",
                        document_id,
                        attribute_id,
                    )

                return {"attribute_id": attribute_id, "document_id": document_id}

    # ========================================================================
    # SQL FUNCTION: add_personal_attribute
    # ========================================================================

    def add_personal_attribute_rpc_function(
        self,
        user_id: str,
        attribute_type: str,
        title: str,
        description: str,
        examples: List[str] | None = None,
        importance_score: int | None = None,
        confidence_level: int | None = None,
        related_articles: List[str] | None = None,
        related_experiences: List[str] | None = None,
        model_name: str | None = None,
        create_searchable: bool = True,
    ) -> str:
        """
        Call SQL function to add personal attribute.
        Returns attribute_id.
        """
        if model_name is None:
            raise ValueError("model_name is required for add_personal_attribute_rpc_function")

        result = self.supabase.rpc(
            "add_personal_attribute",
            {
                "p_user_id": user_id,
                "p_attribute_type": attribute_type,
                "p_title": title,
                "p_description": description,
                "p_examples": examples or [],
                "p_importance_score": importance_score,
                "p_confidence_level": confidence_level,
                "p_related_articles": related_articles or [],
                "p_related_experiences": related_experiences or [],
                "p_embedding_model_name": model_name,
                "p_create_searchable": create_searchable,
            },
        ).execute()

        return result.data # type: ignore

    # ========================================================================
    # SQL FUNCTION: update_personal_attribute
    # ========================================================================

    def update_personal_attribute_rpc_function(
        self,
        attribute_id: str,
        title: str | None = None,
        description: str | None = None,
        examples: List[str] | None = None,
        importance_score: int | None = None,
        confidence_level: int | None = None,
        related_articles: List[str] | None = None,
        related_experiences: List[str] | None = None,
        recreate_embedding: bool = True,
    ) -> bool:
        """
        Call SQL function to update personal attribute.
        Returns success boolean.
        """
        result = self.supabase.rpc(
            "update_personal_attribute",
            {
                "p_attribute_id": attribute_id,
                "p_title": title,
                "p_description": description,
                "p_examples": examples,
                "p_importance_score": importance_score,
                "p_confidence_level": confidence_level,
                "p_related_articles": related_articles,
                "p_related_experiences": related_experiences,
                "p_recreate_embedding": recreate_embedding,
            },
        ).execute()

        return result.data # type: ignore

    async def update_personal_attribute(
        self,
        attribute_id: str,
        title: str | None = None,
        description: str | None = None,
        examples: List[str] | None = None,
        importance_score: int | None = None,
        confidence_level: int | None = None,
        related_articles: List[str] | None = None,
        related_experiences: List[str] | None = None,
        recreate_embeddings: bool = True,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> bool:
        """Update personal attribute using database transaction"""
        if not self.pg_pool:
            raise RuntimeError(
                "PostgreSQL pool not initialized. Call init_pool() first."
            )

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                # Get current attribute
                attr = await conn.fetchrow(
                    "SELECT document_id FROM personal_attributes WHERE id = $1",
                    attribute_id,
                )

                if not attr:
                    raise ValueError(f"Personal attribute {attribute_id} not found")

                document_id = attr["document_id"]

                # Build update query
                update_parts = []
                params = []
                param_count = 1

                if title is not None:
                    update_parts.append(f"title = ${param_count}")
                    params.append(title)
                    param_count += 1

                if description is not None:
                    update_parts.append(f"description = ${param_count}")
                    params.append(description)
                    param_count += 1

                if examples is not None:
                    update_parts.append(f"examples = ${param_count}")
                    params.append(examples)
                    param_count += 1

                if importance_score is not None:
                    update_parts.append(f"importance_score = ${param_count}")
                    params.append(importance_score)
                    param_count += 1

                if confidence_level is not None:
                    update_parts.append(f"confidence_level = ${param_count}")
                    params.append(confidence_level)
                    param_count += 1

                if related_articles is not None:
                    update_parts.append(f"related_articles = ${param_count}")
                    params.append(related_articles)
                    param_count += 1

                if related_experiences is not None:
                    update_parts.append(f"related_experiences = ${param_count}")
                    params.append(related_experiences)
                    param_count += 1

                if update_parts:
                    update_parts.append(f"updated_at = NOW()")
                    params.append(attribute_id)

                    await conn.execute(
                        f"""
                        UPDATE personal_attributes
                        SET {", ".join(update_parts)}
                        WHERE id = ${param_count}
                        """,
                        *params,
                    )

                # Recreate embeddings if needed
                if (
                    document_id
                    and recreate_embeddings
                    and (title or description or examples)
                ):
                    if embeddings_by_model is None:
                        raise ValueError(
                            "embeddings_by_model is required to recreate embeddings"
                        )
                    if chunks is None:
                        raise ValueError("chunks is required to recreate embeddings")
                    # Get updated searchable_text and title
                    updated_attr = await conn.fetchrow(
                        "SELECT searchable_text, title FROM personal_attributes WHERE id = $1",
                        attribute_id,
                    )
                    searchable_text = updated_attr["searchable_text"]
                    new_title = updated_attr["title"]

                    # Update document
                    await conn.execute(
                        "UPDATE documents SET title = $1, content = $2, updated_at = NOW() WHERE id = $3",
                        new_title,
                        searchable_text,
                        document_id,
                    )

                    # Get existing embedding model IDs
                    existing = await conn.fetch(
                        "SELECT DISTINCT embedding_model_id FROM embeddings WHERE document_id = $1",
                        document_id,
                    )
                    model_ids = [row["embedding_model_id"] for row in existing]

                    # Delete old embeddings
                    await conn.execute(
                        "DELETE FROM embeddings WHERE document_id = $1", document_id
                    )

                    model_names = self._get_model_names_from_ids(model_ids)
                    if model_names:
                        await self._insert_embeddings_for_models(
                            conn,
                            document_id,
                            chunks,
                            model_names,
                            embeddings_by_model,
                        )

                return True

    def delete_personal_attribute(self, attribute_id: str) -> bool:
        """Delete personal attribute"""
        attr = (
            self.supabase.table("personal_attributes")
            .select("document_id")
            .eq("id", attribute_id)
            .single()
            .execute()
        )

        attr_data = attr.data if isinstance(attr.data, dict) else None
        document_id = attr_data.get("document_id") if attr_data else None
        if document_id:
            self.delete_document(str(document_id), soft_delete=False)
        else:
            self.supabase.table("personal_attributes").delete().eq(
                "id", attribute_id
            ).execute()

        return True

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

    # ========================================================================
    # SMART UPDATE OPERATIONS
    # ========================================================================

    async def smart_update(
        self,
        user_id: str,
        update_description: str,
        new_content: str,
        content_type: str | None = None,
        similarity_threshold: float = 0.85,
        model_name: str | None = None,
        query_embedding: List[float] | None = None,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> dict[str, Any]:
        """Intelligently find and update documents based on semantic similarity"""
        if model_name is None:
            raise ValueError("model_name is required for smart_update")
        if query_embedding is None:
            raise ValueError("query_embedding is required for smart_update")
        if embeddings_by_model is None:
            raise ValueError("embeddings_by_model is required for smart_update")
        if chunks is None:
            raise ValueError("chunks is required for smart_update")

        matches = await self.search_rpc_function(
            query=update_description,
            user_id=user_id,
            content_types=[content_type] if content_type else None,
            threshold=similarity_threshold,
            limit=3,
            model_name=model_name,
            query_embedding=query_embedding,
        )

        if not matches:
            return {
                "success": False,
                "message": "No matching documents found",
                "matches": [],
            }

        best_match = matches[0]
        document_id = best_match["document_id"]

        document = self.get_document(document_id)
        if not document:
            return {
                "success": False,
                "message": f"Document {document_id} not found",
                "matches": matches,
            }

        update_result = await self._apply_smart_update(
            document=document,
            best_match=best_match,
            new_content=new_content,
            embeddings_by_model=embeddings_by_model,
            chunks=chunks,
        )

        return {
            "success": True,
            "matched_document_id": document_id,
            "similarity": best_match["similarity"],
            "content_type": best_match["content_type"],
            "article_id": best_match.get("article_id"),
            "profile_data_id": best_match.get("profile_data_id"),
            "personal_attribute_id": best_match.get("personal_attribute_id"),
            "update_result": update_result,
            "all_matches": matches,
        }

    async def _apply_smart_update(
        self,
        document: dict[str, Any],
        best_match: dict[str, Any],
        new_content: str,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> dict[str, Any]:
        """Apply update to the appropriate table based on match type"""
        if embeddings_by_model is None:
            raise ValueError("embeddings_by_model is required for smart update")
        if chunks is None:
            raise ValueError("chunks is required for smart update")

        if best_match.get("article_id"):
            await self.update_article(
                article_id=best_match["article_id"],
                content=new_content,
                recreate_embeddings=True,
                embeddings_by_model=embeddings_by_model,
                chunks=chunks,
            )
            return {"type": "article", "id": best_match["article_id"]}

        elif best_match.get("profile_data_id"):
            profile = self.get_profile_data(best_match["profile_data_id"])
            if not profile:
                raise ValueError(
                    f"Profile data {best_match['profile_data_id']} not found"
                )
            updated_data = self._merge_profile_data(profile["data"], new_content)

            await self.update_profile_data(
                profile_id=best_match["profile_data_id"],
                data=updated_data,
                recreate_embeddings=True,
                embeddings_by_model=embeddings_by_model,
                chunks=chunks,
            )
            return {"type": "profile_data", "id": best_match["profile_data_id"]}

        elif best_match.get("personal_attribute_id"):
            await self.update_personal_attribute(
                attribute_id=best_match["personal_attribute_id"],
                description=new_content,
                recreate_embeddings=True,
                embeddings_by_model=embeddings_by_model,
                chunks=chunks,
            )
            return {
                "type": "personal_attribute",
                "id": best_match["personal_attribute_id"],
            }

        else:
            await self.update_document(
                document_id=document["id"],
                content=new_content,
                recreate_embeddings=True,
                embeddings_by_model=embeddings_by_model,
                chunks=chunks,
            )
            return {"type": "document", "id": document["id"]}

    async def propose_updates(
        self,
        user_id: str,
        update_request: str,
        model_name: str | None = None,
        query_embedding: List[float] | None = None,
    ) -> List[dict[str, Any]]:
        """Find potential updates but don't apply them yet"""
        if model_name is None:
            raise ValueError("model_name is required for propose_updates")
        if query_embedding is None:
            raise ValueError("query_embedding is required for propose_updates")

        matches = await self.search_rpc_function(
            query=update_request,
            user_id=user_id,
            threshold=0.75,
            limit=5,
            model_name=model_name,
            query_embedding=query_embedding,
        )

        proposals = []
        for match in matches:
            proposal = {
                "document_id": match["document_id"],
                "current_content": match["chunk_text"],
                "similarity": match["similarity"],
                "content_type": match["content_type"],
                "title": match["title"],
                "article_id": match.get("article_id"),
                "profile_data_id": match.get("profile_data_id"),
                "personal_attribute_id": match.get("personal_attribute_id"),
                "proposed_action": "update",
            }
            proposals.append(proposal)

        return proposals

    async def apply_confirmed_update(
        self,
        document_id: str,
        new_content: str,
        article_id: str | None = None,
        profile_data_id: str | None = None,
        personal_attribute_id: str | None = None,
        embeddings_by_model: dict[str, List[List[float]]] | None = None,
        chunks: List[str] | None = None,
    ) -> bool:
        """Apply a confirmed update after user approval"""
        if embeddings_by_model is None:
            raise ValueError("embeddings_by_model is required for apply_confirmed_update")
        if chunks is None:
            raise ValueError("chunks is required for apply_confirmed_update")

        if article_id:
            await self.update_article(
                article_id=article_id,
                content=new_content,
                recreate_embeddings=True,
                embeddings_by_model=embeddings_by_model,
                chunks=chunks,
            )
        elif profile_data_id:
            profile = self.get_profile_data(profile_data_id)
            if profile and "data" in profile:
                updated_data = self._merge_profile_data(profile["data"], new_content)
                await self.update_profile_data(
                    profile_id=profile_data_id,
                    data=updated_data,
                    recreate_embeddings=True,
                    embeddings_by_model=embeddings_by_model,
                    chunks=chunks,
                )
        elif personal_attribute_id:
            await self.update_personal_attribute(
                attribute_id=personal_attribute_id,
                description=new_content,
                recreate_embeddings=True,
                embeddings_by_model=embeddings_by_model,
                chunks=chunks,
            )
        else:
            await self.update_document(
                document_id=document_id,
                content=new_content,
                recreate_embeddings=True,
                embeddings_by_model=embeddings_by_model,
                chunks=chunks,
            )

        return True

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _flatten_dict_to_text(self, data: dict[str, Any]) -> str:
        """Convert dictionary to searchable text"""
        parts = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                parts.append(f"{key}: {', '.join(map(str, value))}")
            elif isinstance(value, dict):
                parts.append(f"{key}: {self._flatten_dict_to_text(value)}")
        return ". ".join(parts)

    def _merge_profile_data(self, existing_data: dict[str, Any], new_content: str) -> dict[str, Any]:
        """Merge new content with existing profile data"""
        updated_data = existing_data.copy()

        if "description" in updated_data:
            updated_data["description"] = new_content
        elif "details" in updated_data:
            updated_data["details"] = new_content
        else:
            updated_data["updated_info"] = new_content

        return updated_data

    def _create_slug(self, title: str) -> str:
        """Create URL-friendly slug from title"""
        import re

        slug = title.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return slug[:100]

    async def _create_unique_slug(
        self, title: str, conn: asyncpg.Connection | PoolConnectionProxy, user_id: str
    ) -> str:
        """
        Create a unique slug from title, checking for duplicates and appending suffix if needed.

        Args:
            title: Article title
            conn: Database connection
            user_id: User ID to check for duplicates within user's articles

        Returns:
            Unique slug string
        """
        base_slug = self._create_slug(title)
        slug = base_slug
        counter = 1

        # Check if slug exists for this user
        while True:
            existing = await conn.fetchval(
                "SELECT id FROM articles WHERE slug = $1 AND user_id = $2",
                slug,
                user_id,
            )
            if existing is None:
                break  # Slug is unique
            # Append counter to make it unique
            slug = f"{base_slug}-{counter}"
            counter += 1
            # Prevent infinite loop (max 1000 attempts)
            if counter > 1000:
                # Fallback: add timestamp
                import time

                slug = f"{base_slug}-{int(time.time())}"
                break

        return slug
