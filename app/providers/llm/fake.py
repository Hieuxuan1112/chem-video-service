"""Scripted LLM for tests: each call consumes the next script item."""

from collections.abc import Callable, Sequence

from app.providers.errors import ProviderError
from app.providers.llm.protocol import LLMRequest, LLMResponse, Usage

_DEFAULT_USAGE = Usage(1000, 500)


class FakeLLM:
    name = "fake"

    def __init__(self, script: Sequence[object], *, usage: Usage | None = _DEFAULT_USAGE) -> None:
        """Items are a response text, a ProviderError to raise, or a callable building the text."""
        self._script = list(script)
        self._usage = usage
        self.requests: list[LLMRequest] = []

    def generate_json(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        if not self._script:
            raise AssertionError("FakeLLM script exhausted")
        item = self._script.pop(0)
        if isinstance(item, ProviderError):
            raise item
        if callable(item):
            builder: Callable[[LLMRequest], str] = item
            item = builder(req)
        assert isinstance(item, str)
        return LLMResponse(text=item, usage=self._usage, model=req.model)
