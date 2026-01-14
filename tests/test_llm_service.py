import pytest

from src.libs.vector_database import EmbeddingModel
from src.services import llm_service
from src.services.llm_service import BaseLLMService, OllamaLLMService


class DummyLLMService(BaseLLMService):
    def __init__(
        self,
        raise_on_stream: bool = False,
        raise_on_generate: bool = False,
    ):
        self.raise_on_stream = raise_on_stream
        self.raise_on_generate = raise_on_generate
        self.stream_calls = 0
        self.generate_calls = 0
        self.handle_analysis_error_called = False
        self.handle_resume_error_called = False

    async def _stream_generate(self, prompt: str):
        self.stream_calls += 1
        if self.raise_on_stream:
            raise RuntimeError("stream failed")
        yield "chunk1"
        yield "chunk2"

    async def _generate(self, prompt: str) -> str:
        self.generate_calls += 1
        if self.raise_on_generate:
            raise RuntimeError("generate failed")
        return "full"

    def _handle_analysis_error(self, exc: Exception) -> None:
        self.handle_analysis_error_called = True

    async def _handle_resume_error(self, exc: Exception):
        self.handle_resume_error_called = True
        yield "fallback"


class RenderedSourceLLMService(DummyLLMService):
    def _use_rendered_source(self) -> bool:
        return True

    async def _stream_generate(self, prompt: str):
        raise AssertionError("_stream_generate should not be called")


@pytest.mark.asyncio
async def test_generate_resume_stream_uses_stream_generate():
    service = DummyLLMService()
    chunks = [chunk async for chunk in service.generate_resume("jd", [], stream=True)]
    assert chunks == ["chunk1", "chunk2"]
    assert service.stream_calls == 1
    assert service.generate_calls == 0


@pytest.mark.asyncio
async def test_generate_resume_non_stream_uses_generate():
    service = DummyLLMService()
    chunks = [chunk async for chunk in service.generate_resume("jd", [], stream=False)]
    assert chunks == ["full"]
    assert service.stream_calls == 0
    assert service.generate_calls == 1


@pytest.mark.asyncio
async def test_generate_resume_handles_stream_errors():
    service = DummyLLMService(raise_on_stream=True)
    chunks = [chunk async for chunk in service.generate_resume("jd", [], stream=True)]
    assert chunks == ["fallback"]
    assert service.handle_resume_error_called is True


@pytest.mark.asyncio
async def test_generate_resume_from_source_uses_rendered_resume():
    service = RenderedSourceLLMService()
    resume_source = {"profile_data": [], "personal_attributes": []}
    chunks = [
        chunk
        async for chunk in service.generate_resume_from_source(
            "jd",
            resume_source,
            {},
            stream=True,
        )
    ]
    assert len(chunks) == 1
    assert chunks[0].startswith("# Resume")


@pytest.mark.asyncio
async def test_analyze_text_returns_default_result():
    service = DummyLLMService(raise_on_generate=True)
    result = await service.analyze_text("jd")
    assert service.handle_analysis_error_called is True
    assert result == llm_service._default_analysis_result()


def test_ollama_model_prefers_supabase_embedding(monkeypatch):
    model = EmbeddingModel(
        id="1",
        name="ollama-model",
        provider="ollama",
        model_identifier="supabase-model",
        dimensions=0,
        is_local=True,
        cost_per_token=None,
    )
    monkeypatch.setattr(llm_service, "get_default_generation_model", lambda provider: model)
    service = OllamaLLMService()
    assert service.model == "supabase-model"
