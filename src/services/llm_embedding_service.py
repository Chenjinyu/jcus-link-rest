"""
Embedding service utilities for generating vector representations.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
import openai
from google import genai

from src.config import settings
from src.libs.vector_database import EmbeddingModel


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation providers."""

    async def generate_embeddings(self, text: str, model: EmbeddingModel) -> list[float]:
        """Generate embeddings for input text using the given model."""
        ...


_embedding_provider: EmbeddingProvider | None = None


class LLMEmbbingService:
    """Embedding generator backed by vendor APIs."""

    def __init__(
        self,
        openai_api_key: str | None,
        google_api_key: str | None,
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> None:
        """Initialize embedding clients based on available API keys."""
        self._openai_client: openai.OpenAI | None = None
        self._google_client: genai.Client | None = None
        self._ollama_url: str = ollama_url

        if openai_api_key:
            self._openai_client = openai.OpenAI(api_key=openai_api_key)
        if google_api_key:
            self._google_client = genai.Client(api_key=google_api_key)

    async def generate_embeddings(self, text: str, model: EmbeddingModel) -> list[float]:
        """Generate embeddings from the provider configured on the model."""
        dimensions = model.dimensions if model.dimensions <= 2000 else None
        if model.provider == "openai":
            if not self._openai_client:
                raise ValueError("OpenAI client not initialized")
            request_params: dict[str, Any] = {
                "model": model.model_identifier,
                "dimensions": dimensions,
                "input": text,
            }
            response = self._openai_client.embeddings.create(**request_params)
            return response.data[0].embedding
        if model.provider == "ollama":
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._ollama_url}/api/embeddings",
                    json={"model": model.model_identifier, "prompt": text},
                    timeout=30.0,
                )
                return response.json()["embedding"]
        if model.provider == "google":
            if not self._google_client:
                raise ValueError("Google client not initialized")
            result = self._google_client.models.embed_content(
                model=model.model_identifier,
                contents=[text],
                config={"output_dimensionality": dimensions},
            )
            embs = getattr(result, "embeddings", None)
            if not embs:
                raise ValueError("No embeddings returned from Google API")
            embs_value = embs.values
            if isinstance(embs_value, list) and len(embs_value) > 0:
                return [float(x) for x in embs_value]
            return []
        raise ValueError(f"Provider {model.provider} not supported")


def get_embedding_provider() -> EmbeddingProvider:
    """Return a singleton embedding provider instance."""
    global _embedding_provider
    if _embedding_provider is not None:
        return _embedding_provider
    _embedding_provider = LLMEmbbingService(
        openai_api_key=settings.openai_api_key,
        google_api_key=settings.google_api_key,
        ollama_url=settings.ollama_url,
    )
    return _embedding_provider


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


async def embed_text(text: str, model: EmbeddingModel | None) -> list[float]:
    """Generate embeddings for text using the provided embedding model."""
    if not model:
        raise ValueError("embedding model is required")
    return await get_embedding_provider().generate_embeddings(text, model)
