"""Profile data access and matching via Supabase-backed VectorDatabase."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from src.config import settings
from src.libs.vector_database import EmbeddingModel, VectorDatabase
from src.services import llm_embedding_service


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str, ensure_ascii=True)


class ProfileService:
    """Service for fetching profile data and running vector search."""

    def __init__(self, user_id: str) -> None:
        if not settings.supabase_url or not settings.supabase_key:
            raise ValueError("Supabase credentials are required")
        self.user_id = user_id
        self._db = VectorDatabase(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_key,
            postgres_url=settings.supabase_postgres_url or "",
        )

    async def search_job_matches(
        self,
        job_description: str,
        top_k: int,
        threshold: float,
        embedding_model: EmbeddingModel,
    ) -> list[dict[str, Any]]:
        query_embedding = await llm_embedding_service.embed_text(
            job_description,
            model=embedding_model,
        )
        return await self._db.search_rpc_function(
            query=job_description,
            user_id=self.user_id,
            threshold=threshold,
            limit=top_k,
            model_name=embedding_model.name,
            query_embedding=query_embedding,
        )

    def get_profile_data_by_ids(self, profile_ids: Iterable[str]) -> list[dict[str, Any] | Any]:
        ids = [pid for pid in profile_ids if pid]
        if not ids:
            return []
        result = (
            self._db.supabase.table("profile_data")
            .select("*")
            .in_("id", ids)
            .execute()
        )
        data = result.data or []
        # sort empty display_order to the end
        data.sort(
            key=lambda item: (item.get("display_order") is None, item.get("display_order", 0)) # type: ignore
        )
        return data

    def get_resume_source(
        self,
        profile_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if profile_ids:
            profile_data = self.get_profile_data_by_ids(profile_ids)
        else:
            profile_data = self._db.get_profile_data_list(user_id=self.user_id)
        personal_attributes = self._db.get_personal_attributes(user_id=self.user_id)
        return {
            "user_id": self.user_id,
            "profile_data": profile_data,
            "personal_attributes": personal_attributes,
        }

    def fingerprint_resume_source(self, resume_source: dict[str, Any]) -> str:
        payload = _stable_json(
            {
                "profile_data": resume_source.get("profile_data", []),
                "personal_attributes": resume_source.get("personal_attributes", []),
            }
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_profile_service: ProfileService | None = None


def get_profile_service(user_id: str | None = None) -> ProfileService:
    global _profile_service
    user_id = user_id or settings.author_user_id or "default_user"
    if _profile_service is None:
        _profile_service = ProfileService(user_id=user_id)
    return _profile_service
