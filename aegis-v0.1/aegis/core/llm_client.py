"""
Thin wrapper around the Anthropic API.

This is intentionally the ONLY file that knows how to talk to a
model provider. Everything else in core/ talks to an LLMClient, not
to the Anthropic SDK directly. That is what makes the future Model
Router (architecture doc section 51 -- routing between local/cloud
models) a change to this one file instead of a rewrite of MAX.
"""

from typing import Protocol


class LLMClient(Protocol):
    """Anything that can turn (system prompt, message history) into a reply."""

    def send(self, system: str, messages: list[dict]) -> str: ...


class AnthropicClient:
    """Real client, backed by the Anthropic API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        # Imported lazily so the package isn't required just to run
        # tests against MockClient below.
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def send(self, system: str, messages: list[dict]) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
        )
        parts = [block.text for block in response.content if block.type == "text"]
        return "".join(parts)


class MockClient:
    """
    Deterministic stand-in for AnthropicClient, used in tests and for
    development without burning API credits. Echoes back a fixed
    reply so the conversation loop, history management, and command
    handling can all be verified without a network call.
    """

    def __init__(self, reply: str = "acknowledged."):
        self._reply = reply
        self.calls: list[dict] = []

    def send(self, system: str, messages: list[dict]) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self._reply
