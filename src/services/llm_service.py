"""
#### this file has overlap with vector_database.py ####
# LLM Service for resume generation and text processing.
llm_service.py initializes AI models for generation and embeddings.
vector_database.py fetches embedding models from Supabase and manages vector queries.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, AsyncGenerator, AsyncIterator, Protocol

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


class LLMEmbeddingProvider:
    """Embedding generator backed by vendor APIs."""

    def __init__(
        self,
        openai_api_key: str | None,
        google_api_key: str | None,
        ollama_url: str = "http://localhost:11434",
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
    _embedding_provider = LLMEmbeddingProvider(
        openai_api_key=settings.openai_api_key,
        google_api_key=settings.google_api_key,
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


class BaseLLMService(ABC):
    """
    Abstract base class for LLM services by following the dependency inversion principle,
    depending on abstractions rather than concrete implementations."""

    @abstractmethod
    async def _stream_generate_text(self, prompt: str) -> AsyncIterator[str]:
        """Stream text generation output for a prompt."""
        raise NotImplementedError

    @abstractmethod
    async def _generate_text(self, prompt: str) -> str:
        """Generate a full text response for a prompt."""
        raise NotImplementedError

    async def _yield_prompt(self, prompt: str, stream: bool) -> AsyncIterator[str]:
        """Yield text chunks or a single response depending on stream mode."""
        if stream:
            async for chunk in self._stream_generate_text(prompt): # type: ignore
                yield chunk
        else:
            yield await self._generate_text(prompt)

    async def generate_stream_text(
        self, prompt: str, stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate/Answer text from a prompt, optionally streaming."""
        async for chunk in self._yield_prompt(prompt, stream):
            yield chunk

    async def generate_text_response(self, prompt: str) -> str:
        """Generate a full response for a prompt."""
        return await self._generate_text(prompt)


LLMServiceFactoryCallable = Callable[
    [], # input args
    BaseLLMService # return type
]
_LLM_SERVICE_REGISTRY: dict[str, LLMServiceFactoryCallable] = {}


def register_llm_service(provider: str, factory: LLMServiceFactoryCallable) -> None:
    """Register an LLM provider without modifying the factory implementation."""
    key = provider.strip().lower()
    if not key:
        raise ValueError("LLM provider name cannot be empty")
    _LLM_SERVICE_REGISTRY[key] = factory


def available_llm_providers() -> list[str]:
    """Return registered LLM provider names."""
    return sorted(_LLM_SERVICE_REGISTRY.keys())


class GoogleLLMService(BaseLLMService):
    """Google Gemini LLM service implementation"""

    def __init__(self) -> None:
        """Initialize Google LLM client settings."""
        self.api_key: str | None = settings.google_api_key
        self.model: str = settings.google_model
        self.max_tokens: int = 8192  # Default for Gemini models
        self.temperature: float = 0.7  # Default temperature

        if not self.api_key:
            raise ValueError("Google API key not set")

        self.client: genai.Client = genai.Client(api_key=self.api_key)

    async def _stream_generate_text(self, prompt: str) -> AsyncIterator[str]:
        """Stream Google model text generation output."""
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
        )
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def _generate_text(self, prompt: str) -> str:
        """Generate a full response using the Google model."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        return text or ""


class OpenAILLMService(BaseLLMService):
    """OpenAI GPT service implementation"""

    def __init__(self) -> None:
        """Initialize OpenAI client settings."""
        self.api_key: str | None = settings.openai_api_key
        self.model: str = settings.openai_model
        self.max_tokens: int = 4000  # Default for OpenAI models
        self.temperature: float = 0.7  # Default temperature

        if not self.api_key:
            raise ValueError("OpenAI API key not set")

        self.client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=self.api_key)

    async def _stream_generate_text(self, prompt: str) -> AsyncIterator[str]:
        """Stream OpenAI model text generation output."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        async for event in response:
            delta = event.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    async def _generate_text(self, prompt: str) -> str:
        """Generate a full response using the OpenAI model."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        message = response.choices[0].message
        return message.content or ""


class OllamaLLMService(BaseLLMService):
    """Ollama LLM service implementation"""

    def __init__(self) -> None:
        """Initialize Ollama client settings."""
        self.base_url: str = settings.ollama_url
        self.model: str = settings.ollama_model
        self.max_tokens: int = 4096
        self.temperature: float = 0.7

    async def _stream_generate_text(self, prompt: str) -> AsyncIterator[str]:
        """Stream Ollama model text generation output."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("response")
                    if chunk:
                        yield chunk

    async def _generate_text(self, prompt: str) -> str:
        """Generate a full response using the Ollama model."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")


class LLMServiceFactory:
    """Factory for creating LLM service instances using a provider registry."""

    @staticmethod
    def create(provider: str | None = None) -> BaseLLMService:
        """Create LLM service based on configuration or explicit provider."""
        provider_key = (provider or settings.default_llm_provider).strip().lower()
        try:
            factory = _LLM_SERVICE_REGISTRY[provider_key]
        except KeyError as exc:
            supported = ", ".join(available_llm_providers()) or "none"
            raise ValueError(
                f"Unsupported LLM provider: {provider_key}. Supported: {supported}"
            ) from exc
        return factory()


register_llm_service("openai", OpenAILLMService)
register_llm_service("google", GoogleLLMService)
register_llm_service("ollama", OllamaLLMService)


_llm_service_cache: dict[str, BaseLLMService] = {}


def get_llm_service(provider: str | None) -> BaseLLMService:
    """Get LLM service instance (singleton per provider)."""
    if not provider:
        raise ValueError("provider is required")
    key = provider.strip().lower()
    service = _llm_service_cache.get(key)
    if service is None:
        service = LLMServiceFactory.create(provider=key)
        _llm_service_cache[key] = service
    return service
