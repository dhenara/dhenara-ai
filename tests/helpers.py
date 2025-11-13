"""Shared testing helpers and fakes for dhenara_ai tests."""

from __future__ import annotations

from dhenara.ai.providers.base import AIModelProviderClientBase
from dhenara.ai.types.genai import (
    ChatMessageContentPart,
    ChatResponse,
    ChatResponseChoice,
    ChatResponseTextContentItem,
)
from dhenara.ai.types.genai.dhenara.response import AIModelCallResponseMetaData


class DummyFormatter:
    """Lightweight formatter used by fake providers."""

    @classmethod
    def format_prompt(cls, prompt, model_endpoint=None, **kwargs):
        if prompt is None:
            return None
        if isinstance(prompt, dict):
            return prompt
        return {"role": "user", "text": str(prompt)}

    @classmethod
    def format_context(cls, context, model_endpoint=None, **kwargs):
        if not context:
            return []
        return [{"role": "user", "text": str(item)} for item in context]

    @classmethod
    def format_instructions(cls, instructions, model_endpoint=None, **kwargs):
        if not instructions:
            return None
        if isinstance(instructions, dict):
            return instructions
        if isinstance(instructions, str):
            return {"role": "system", "content": instructions}
        joined = " ".join(str(item) for item in instructions)
        return {"role": "system", "content": joined}


class FakeProvider(AIModelProviderClientBase):
    """Deterministic provider implementation used in component/service tests."""

    formatter = DummyFormatter

    def __init__(self, *args, response_text: str = "fake-response", **kwargs):
        super().__init__(*args, **kwargs)
        self.response_text = response_text
        self.recorded_params = None
        self.streaming_chunks: list = []

    # Context lifecycle -------------------------------------------------
    def initialize(self) -> None:  # type: ignore[override]
        return None

    def cleanup(self) -> None:  # type: ignore[override]
        return None

    def _setup_client_sync(self):  # type: ignore[override]
        return self

    async def _setup_client_async(self):  # type: ignore[override]
        return self

    # API invocation ----------------------------------------------------
    def get_api_call_params(self, **kwargs):  # type: ignore[override]
        self.recorded_params = kwargs
        return {"payload": kwargs}

    def do_api_call_sync(self, api_call_params):  # type: ignore[override]
        return {"text": self.response_text, "params": api_call_params}

    async def do_api_call_async(self, api_call_params):  # type: ignore[override]
        return {"text": self.response_text, "params": api_call_params}

    def do_streaming_api_call_sync(self, api_call_params):  # type: ignore[override]
        return iter(self.streaming_chunks)

    async def do_streaming_api_call_async(self, api_call_params):  # type: ignore[override]
        async def _gen():
            for chunk in self.streaming_chunks:
                yield chunk

        return _gen()

    # Response parsing --------------------------------------------------
    def parse_response(self, response):  # type: ignore[override]
        message_part = ChatMessageContentPart(
            type="output_text",
            text=str(response.get("text", self.response_text)),
            annotations=None,
        )
        choice = ChatResponseChoice(
            index=0,
            contents=[
                ChatResponseTextContentItem(
                    index=0,
                    role="assistant",
                    message_contents=[message_part],
                )
            ],
        )
        return ChatResponse(
            model=self.model_endpoint.ai_model.model_name,
            provider=self.model_endpoint.ai_model.provider,
            api_provider=self.model_endpoint.api.provider,
            choices=[choice],
            metadata=AIModelCallResponseMetaData(streaming=False, duration_seconds=0, provider_metadata={}),
            provider_response=response,
        )

    def parse_stream_chunk(self, chunk):  # type: ignore[override]
        return []

    def _get_usage_from_provider_response(self, response):  # type: ignore[override]
        return None


def make_fake_provider(response_text: str = "fake-response"):
    """Factory returning a creator suitable for monkeypatching AIModelClientFactory."""

    def _creator(cls, model_endpoint, config, is_async):  # pragma: no cover - simple passthrough
        return FakeProvider(
            model_endpoint=model_endpoint,
            config=config,
            is_async=is_async,
            response_text=response_text,
        )

    return _creator
