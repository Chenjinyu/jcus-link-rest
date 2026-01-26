from pyexpat import model
from typing import Any, Dict, List, Union
from uuid import UUID
from supabase import Client
from src.config.settings import settings

async def get_model_id(
    vector_db_client,
    provider: str | None = None, 
    model_identifier: str | None = None
) -> UUID:
    if provider is None:
        provider = settings.default_llm_provider
    if model_identifier is None:
        model_name = settings.default_embedding_model_name

    if not provider or not model_identifier:
        raise ValueError("provider and model_name are required")

    query = (
        vector_db_client.table("embedding_models")
        .select("id")
        .eq("provider", provider)
        .eq("model_identifier", model_identifier)
    )
    result = query.execute()
    if not result.data:
        raise ValueError(f"Model not found: {provider}, {model_identifier}")
    return result.data[0]["id"]

def _normalize_execute_result(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []
    # supabase-py style: object with .data
    data = getattr(result, "data", None)
    if isinstance(data, list):
        return data
    # sometimes library returns plain list
    if isinstance(result, list):
        return result
    # sometimes result is dict with "data" key
    if isinstance(result, dict) and isinstance(result.get("data"), list):
        return result["data"]
    return []

async def profile_similarity_search_rpc(
    vector_db_client,
    query_embedding: list[float],
    model_id: UUID,
    embedding_dimensions: int,
    user_id: str,
    top_k: int = 10,
    threshold: float = settings.min_similarity_threshold,
) -> list[dict[str, Any]]:
    similarity_query_payload = {
        "query_embedding": query_embedding,
        "model_id": model_id,
        "embedding_dimensions": embedding_dimensions,
        "user_id_filter": user_id,
        "match_count": top_k,
        "match_threshold": threshold,
    }
    print("="*30 + "similarity_query_payload"+ "="*30)
    print(similarity_query_payload)
    response = (
        vector_db_client.rpc(
            "profile_similarity_search",
            similarity_query_payload,
        )
        .execute()
    )
    # {
    #   "profile_id": uuid,
    #   "data": JSONB,
    #   "tag": list[str],
    #   "similarity": float
    # }
    print("=" * 30 + "profile_similarity_search" + "=" * 30)
    print(type(response))
    print(response)
    return response
    # return response.data or []
    
async def search_similar_content_rpc(
    vector_db_client,
    query_embedding: list[float],
    model_id: UUID,
    embedding_dimensions: int,
    user_id: str,
    top_k: int = 10,
    threshold: float = settings.min_similarity_threshold,
):
    response = (
        vector_db_client.rpc(
            "search_similar_content",
            {
                "query_embedding": query_embedding,
                "model_id": model_id,
                "embedding_dimensions": embedding_dimensions,
                "user_id_filter": user_id,
                "match_count": top_k,
                "match_threshold": threshold,
            },
        )
        .execute()
    )
    return response
    # return response.data or []