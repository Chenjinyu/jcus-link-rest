"""
#### this file has overlap with vector_database.py ####
# LLM Service for text generation.
llm_service.py initializes AI models for generation.
vector_database.py fetches embedding models from Supabase and manages vector queries.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import AsyncGenerator, AsyncIterator

import httpx
import openai
from google import genai

from src.config import settings


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

    async def extract_jobs_insights(self, prompt: str) -> str:
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
                timeout=settings.ollama_timeout_seconds,
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
                timeout=settings.ollama_timeout_seconds,
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
