import pytest

from libs.llm import BaseLLMService


class DummyLLMService(BaseLLMService):
    def __init__(
        self,
        raise_on_stream: bool = False,
        raise_on_generate: bool = False,
    ) -> None:
        self.raise_on_stream = raise_on_stream
        self.raise_on_generate = raise_on_generate
        self.stream_calls = 0
        self.generate_calls = 0

    async def _stream_generate_text(self, prompt: str):
        self.stream_calls += 1
        if self.raise_on_stream:
            raise RuntimeError("stream failed")
        yield "chunk1"
        yield "chunk2"

    async def _generate_text(self, prompt: str) -> str:
        self.generate_calls += 1
        if self.raise_on_generate:
            raise RuntimeError("generate failed")
        return "full"


@pytest.mark.asyncio
async def test_generate_stream_text_uses_stream_generate():
    service = DummyLLMService()
    chunks = [chunk async for chunk in service.generate_stream_text("jd", stream=True)]
    assert chunks == ["chunk1", "chunk2"]
    assert service.stream_calls == 1
    assert service.generate_calls == 0


@pytest.mark.asyncio
async def test_generate_stream_text_non_stream_uses_generate():
    service = DummyLLMService()
    chunks = [chunk async for chunk in service.generate_stream_text("jd", stream=False)]
    assert chunks == ["full"]
    assert service.stream_calls == 0
    assert service.generate_calls == 1


@pytest.mark.asyncio
async def test_generate_stream_text_handles_stream_errors():
    service = DummyLLMService(raise_on_stream=True)
    with pytest.raises(RuntimeError):
        async for _ in service.generate_stream_text("jd", stream=True):
            pass


@pytest.mark.asyncio
async def test_generate_text_response_uses_generate_text():
    service = DummyLLMService()
    result = await service.generate_text_response("jd")
    assert result == "full"
    assert service.generate_calls == 1
