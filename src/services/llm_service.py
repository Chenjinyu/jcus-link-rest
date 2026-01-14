"""
#### this file has overlap with vector_database.py ####
# LLM Service for resume generation and text processing.
llm_service.py init ai model and use it to call supabase.
vector_database.py gets ai embedding models from supabase database, init the ai model and call vectors from supabase.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable # runtime behavior check
from typing import Any, AsyncGenerator, AsyncIterator, List, Protocol # static type hints

import httpx
import openai
from google import genai

from src.config import settings
from src.libs.exceptions import LLMServiceException
from src.libs.vector_database import EmbeddingModel, VectorDatabase
from src.mcp.prompts import (
    build_analysis_prompt,
    build_resume_from_source_prompt,
    build_resume_prompt,
)
from src.schemas import ResumeMatch

logger = logging.getLogger(__name__)


_SIMULATED_RESUME_PARTS = [
    "# Professional Resume\n\n",
    "## Professional Summary\n",
    "Experienced software engineer with strong background in Python, TypeScript, and cloud technologies. ",
    "Proven track record of building scalable applications and leading development teams.\n\n",
    "## Key Skills\n",
    "- **Programming Languages:** Python, TypeScript, JavaScript\n",
    "- **Frameworks:** FastAPI, React, Node.js\n",
    "- **Cloud Platforms:** AWS, GCP\n",
    "- **Tools:** Docker, Kubernetes, Git\n\n",
    "## Work Experience\n\n",
    "### Senior Software Engineer | Tech Company\n",
    "*2020 - Present*\n\n",
    "- Developed microservices architecture serving 1M+ users\n",
    "- Led team of 5 engineers in implementing CI/CD pipeline\n",
    "- Reduced deployment time by 60% through automation\n\n",
    "## Education\n",
    "**Bachelor of Science in Computer Science**\n",
    "University Name, 2018\n\n",
    "## Achievements\n",
    "- Architected system handling 10K requests/second\n",
    "- Published 3 technical articles on Medium\n",
    "- Contributed to 5+ open source projects\n",
]


def _default_analysis_result() -> dict:
    return {
        "required_skills": ["Python", "FastAPI", "React", "AWS"],
        "experience_level": "Senior",
        "key_responsibilities": [
            "Design and implement scalable systems",
            "Lead technical projects",
            "Mentor junior developers",
        ],
        "estimated_match_threshold": 0.7,
    }


async def _stream_simulated_resume() -> AsyncIterator[str]:
    for part in _SIMULATED_RESUME_PARTS:
        await asyncio.sleep(0.1)
        yield part


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _format_date_range(entry: dict[str, Any]) -> str:
    start = entry.get("start_date")
    end = entry.get("end_date")
    is_current = entry.get("is_current")
    if start and (end or is_current):
        return f"{start} - {end or 'Present'}"
    if start:
        return str(start)
    return ""


def _render_resume_from_source(resume_source: dict[str, Any]) -> str:
    profile_data = resume_source.get("profile_data", []) or []
    personal_attributes = resume_source.get("personal_attributes", []) or []

    sections: list[str] = ["# Resume\n"]

    summary_items = [
        attr
        for attr in personal_attributes
        if attr.get("attribute_type") in {"summary", "bio", "headline"}
    ]
    if summary_items:
        sections.append("## Professional Summary\n")
        for item in summary_items:
            description = item.get("description") or item.get("value")
            if description:
                sections.append(f"- {description}\n")
        sections.append("\n")

    skills: list[str] = []
    for entry in profile_data:
        if entry.get("category") == "skill":
            data = entry.get("data") or {}
            skills.extend(_ensure_list(data.get("skills") or data.get("name")))
    if skills:
        unique_skills = sorted({skill for skill in skills if skill})
        sections.append("## Key Skills\n")
        sections.append("- " + ", ".join(unique_skills) + "\n\n")

    work_items = [entry for entry in profile_data if entry.get("category") == "work_experience"]
    if work_items:
        sections.append("## Work Experience\n")
        for entry in work_items:
            data = entry.get("data") or {}
            title = data.get("title") or data.get("position") or data.get("role")
            company = data.get("company") or data.get("organization")
            header = " | ".join([part for part in [title, company] if part])
            if header:
                sections.append(f"### {header}\n")
            date_range = _format_date_range(entry)
            if date_range:
                sections.append(f"*{date_range}*\n")
            details: list[str] = []
            details.extend(_ensure_list(data.get("description")))
            details.extend(_ensure_list(data.get("responsibilities")))
            details.extend(_ensure_list(data.get("achievements")))
            if details:
                for detail in details:
                    sections.append(f"- {detail}\n")
            sections.append("\n")

    education_items = [entry for entry in profile_data if entry.get("category") == "education"]
    if education_items:
        sections.append("## Education\n")
        for entry in education_items:
            data = entry.get("data") or {}
            degree = data.get("degree") or data.get("title")
            institution = data.get("institution") or data.get("school")
            line = " | ".join([part for part in [degree, institution] if part])
            if line:
                sections.append(f"- {line}\n")
        sections.append("\n")

    cert_items = [entry for entry in profile_data if entry.get("category") == "certification"]
    if cert_items:
        sections.append("## Certifications\n")
        for entry in cert_items:
            data = entry.get("data") or {}
            name = data.get("name") or data.get("title")
            issuer = data.get("issuer")
            line = " | ".join([part for part in [name, issuer] if part])
            if line:
                sections.append(f"- {line}\n")
        sections.append("\n")

    project_items = [entry for entry in profile_data if entry.get("category") == "project"]
    if project_items:
        sections.append("## Projects\n")
        for entry in project_items:
            data = entry.get("data") or {}
            title = data.get("title") or data.get("name")
            if title:
                sections.append(f"### {title}\n")
            details = _ensure_list(data.get("description"))
            for detail in details:
                sections.append(f"- {detail}\n")
            sections.append("\n")

    return "".join(sections).rstrip() + "\n"


class EmbeddingProvider(Protocol):
    async def create_embedding(self, text: str, model: EmbeddingModel) -> list[float]:
        ...


_embedding_models_cache: list[EmbeddingModel] | None = None
_embedding_provider: EmbeddingProvider | None = None


def _get_embedding_models_from_supabase() -> list[EmbeddingModel]:
    global _embedding_models_cache
    if _embedding_models_cache is not None:
        return _embedding_models_cache

    if not settings.supabase_url or not settings.supabase_key:
        _embedding_models_cache = []
        return _embedding_models_cache

    db = VectorDatabase(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key,
        postgres_url=settings.supabase_postgres_url or "",
    )
    _embedding_models_cache = db.list_embedding_models()
    return _embedding_models_cache


def get_embedding_models(provider: str | None = None) -> list[EmbeddingModel]:
    models = _get_embedding_models_from_supabase()
    if provider is None:
        return list(models)
    normalized = provider.strip().lower()
    return [model for model in models if model.provider == normalized]


def get_embedding_model_by_name(name: str) -> EmbeddingModel | None:
    for model in _get_embedding_models_from_supabase():
        if model.name == name:
            return model
    return None


def get_default_embedding_model(provider: str) -> EmbeddingModel | None:
    models = get_embedding_models(provider)
    return models[0] if models else None


class SupabaseEmbeddingProvider:
    def __init__(
        self,
        models: list[EmbeddingModel],
        openai_api_key: str | None,
        google_api_key: str | None,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self._models_by_name = {model.name: model for model in models}
        self._openai_client = None
        self._google_client = None
        self._ollama_url = ollama_url

        if any(model.provider == "openai" for model in models) and openai_api_key:
            self._openai_client = openai.OpenAI(api_key=openai_api_key)
        if any(model.provider == "google" for model in models) and google_api_key:
            self._google_client = genai.Client(api_key=google_api_key)

    async def create_embedding(self, text: str, model: EmbeddingModel) -> list[float]:
        if model.provider == "openai":
            if not self._openai_client:
                raise ValueError("OpenAI client not initialized")
            dimensions = model.dimensions if model.dimensions <= 2000 else None
            request_params: dict[str, Any] = {
                "model": model.model_identifier,
                "input": text,
            }
            if dimensions is not None:
                request_params["dimensions"] = dimensions
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
            result = self._google_client.embed_content(
                model=model.model_identifier,
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        raise ValueError(f"Provider {model.provider} not supported")


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is not None:
        return _embedding_provider
    models = _get_embedding_models_from_supabase()
    _embedding_provider = SupabaseEmbeddingProvider(
        models=models,
        openai_api_key=settings.openai_api_key,
        google_api_key=settings.google_api_key,
    )
    return _embedding_provider


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


async def build_embeddings_by_model(
    chunks: list[str],
    model_names: list[str],
) -> dict[str, list[list[float]]]:
    embeddings_by_model: dict[str, list[list[float]]] = {}
    for model_name in model_names:
        model = get_embedding_model_by_name(model_name)
        if not model:
            raise ValueError(f"Embedding model not found: {model_name}")
        embeddings: list[list[float]] = []
        for chunk in chunks:
            embeddings.append(await get_embedding_provider().create_embedding(chunk, model))
        embeddings_by_model[model_name] = embeddings
    return embeddings_by_model


async def create_embedding(
    text: str,
    model_name: str | None = None,
    provider: str | None = None,
) -> list[float]:
    if model_name:
        model = get_embedding_model_by_name(model_name)
    else:
        resolved_provider = provider or settings.default_llm_provider
        model = get_default_embedding_model(resolved_provider)

    if not model:
        raise ValueError("Embedding model not found for the requested provider/name")

    return await get_embedding_provider().create_embedding(text, model)


class BaseLLMService(ABC):
    """
    Abstract base class for LLM services by following the dependency inversion principle,
    depending on abstractions rather than concrete implementations."""

    async def _yield_prompt(self, prompt: str, stream: bool) -> AsyncIterator[str]:
        if stream:
            async for chunk in self._stream_generate(prompt):
                yield chunk
        else:
            yield await self._generate(prompt)

    async def _analyze_prompt(self, prompt: str) -> None:
        _ = prompt

    def _use_rendered_source(self) -> bool:
        return False

    def _handle_analysis_error(self, exc: Exception) -> None:
        raise LLMServiceException("text analysis", str(exc))

    async def _handle_resume_error(self, exc: Exception) -> AsyncIterator[str]:
        raise LLMServiceException("resume generation", str(exc))
        if False:
            yield ""

    async def _handle_resume_source_error(
        self,
        exc: Exception,
        resume_source: dict[str, Any],
    ) -> AsyncIterator[str]:
        raise LLMServiceException("resume generation", str(exc))
        if False:
            yield ""

    @abstractmethod
    async def _stream_generate(self, prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def _generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def generate_resume(
        self,
        job_description: str,
        matched_resumes: List[ResumeMatch],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate resume based on job description and matches"""
        prompt = build_resume_prompt(job_description, matched_resumes)
        try:
            async for chunk in self._yield_prompt(prompt, stream):
                yield chunk
        except Exception as exc:
            async for chunk in self._handle_resume_error(exc):
                yield chunk

    async def analyze_text(self, text: str) -> dict:
        """Analyze text and extract structured information"""
        prompt = build_analysis_prompt(text)
        try:
            await self._analyze_prompt(prompt)
        except Exception as exc:
            self._handle_analysis_error(exc)
        return _default_analysis_result()

    async def generate_resume_from_source(
        self,
        job_description: str,
        resume_source: dict[str, Any],
        match_summary: dict[str, Any],
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Generate resume using structured source data"""
        if self._use_rendered_source():
            yield _render_resume_from_source(resume_source)
            return

        prompt = build_resume_from_source_prompt(
            job_description,
            resume_source,
            match_summary,
        )
        try:
            async for chunk in self._yield_prompt(prompt, stream):
                yield chunk
        except Exception as exc:
            async for chunk in self._handle_resume_source_error(exc, resume_source):
                yield chunk


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

    def __init__(self):
        self.api_key = settings.google_api_key
        self.embedding_model = get_default_embedding_model("google")
        self.model = (
            self.embedding_model.model_identifier
            if self.embedding_model
            else "text-embedding-004"
        )
        self.embedding_dimensions = (
            self.embedding_model.dimensions if self.embedding_model else None
        )
        self.max_tokens = 8192  # Default for Gemini models
        self.temperature = 0.7  # Default temperature

        if not self.api_key:
            logger.warning("Google API key not set, using simulated responses")

    def _use_rendered_source(self) -> bool:
        return not self.api_key

    async def _stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Stream LLM response (simulated for now)"""
        _ = prompt
        async for chunk in _stream_simulated_resume():
            yield chunk

    async def _generate(self, prompt: str) -> str:
        """Generate complete response"""
        result = []
        async for chunk in self._stream_generate(prompt):
            result.append(chunk)
        return "".join(result)


class OpenAILLMService(BaseLLMService):
    """OpenAI GPT service implementation"""

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.embedding_model = get_default_embedding_model("openai")
        self.model = (
            self.embedding_model.model_identifier
            if self.embedding_model
            else "text-embedding-3-small"
        )
        self.embedding_dimensions = (
            self.embedding_model.dimensions if self.embedding_model else None
        )
        self.max_tokens = 4000  # Default for OpenAI models
        self.temperature = 0.7  # Default temperature

        if not self.api_key:
            logger.warning("OpenAI API key not set, using simulated responses")

    def _use_rendered_source(self) -> bool:
        return not self.api_key

    async def _stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Stream LLM response (simulated for now)"""
        _ = prompt
        async for chunk in _stream_simulated_resume():
            yield chunk

    async def _generate(self, prompt: str) -> str:
        """Generate complete response"""
        result = []
        async for chunk in self._stream_generate(prompt):
            result.append(chunk)
        return "".join(result)


class OllamaLLMService(BaseLLMService):
    """Ollama LLM service implementation"""

    def __init__(self):
        self.base_url = settings.ollama_url
        embedding_model = get_default_embedding_model("ollama")
        self.model = (
            embedding_model.model_identifier
            if embedding_model and embedding_model.model_identifier
            else settings.ollama_model
        )
        if not embedding_model:
            logger.warning("No Ollama model in Supabase; using settings fallback.")
        self.max_tokens = 4096
        self.temperature = 0.7

    async def _analyze_prompt(self, prompt: str) -> None:
        result = await self._generate(prompt)
        _ = result

    def _handle_analysis_error(self, exc: Exception) -> None:
        logger.warning("Ollama unavailable, using simulated responses: %s", exc)

    async def _handle_resume_error(self, exc: Exception) -> AsyncIterator[str]:
        logger.warning("Ollama unavailable, using simulated responses: %s", exc)
        async for chunk in _stream_simulated_resume():
            yield chunk

    async def _handle_resume_source_error(
        self,
        exc: Exception,
        resume_source: dict[str, Any],
    ) -> AsyncIterator[str]:
        logger.warning("Ollama unavailable, using simulated responses: %s", exc)
        yield _render_resume_from_source(resume_source)

    async def _stream_generate(self, prompt: str) -> AsyncIterator[str]:
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

    async def _generate(self, prompt: str) -> str:
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


_llm_service: BaseLLMService | None = None


def get_llm_service() -> BaseLLMService:
    """Get LLM service instance (singleton)"""
    global _llm_service

    if _llm_service is None:
        _llm_service = LLMServiceFactory.create()

    return _llm_service
