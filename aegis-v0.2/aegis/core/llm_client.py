"""
Thin wrapper around the Anthropic API.

This is intentionally the ONLY file that knows how to talk to a
model provider. Everything else in core/ talks to an LLMClient, not
to the Anthropic SDK directly. That is what makes the future Model
Router (architecture doc section 51 -- routing between local/cloud
models) a change to this one file instead of a rewrite of MAX.

v0.2 adds tool-use support. Responses are normalized into
LLMResponse/TextBlock/ToolUseBlock -- a provider-agnostic shape --
so the tool loop in core/max.py never touches Anthropic SDK types
directly, and MockClient can simulate tool_use turns in tests with
no network call and no SDK installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    content: list  # list[TextBlock | ToolUseBlock]
    stop_reason: str  # "end_turn", "tool_use", etc.


class LLMClient(Protocol):
    """Anything that can turn (system prompt, history, tools) into a response."""

    def send(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


class AnthropicClient:
    """Real client, backed by the Anthropic API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        # Imported lazily so the package isn't required just to run
        # tests against MockClient below.
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def send(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        content = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))

        return LLMResponse(content=content, stop_reason=response.stop_reason)


class MockClient:
    """
    Deterministic stand-in for AnthropicClient, used in tests and for
    development without burning API credits.

    Two modes:
    - `reply=...` (default): always returns a single end_turn text
      response. Good for plain-conversation tests.
    - `responses=[...]`: returns each canned LLMResponse in order,
      one per call. Good for simulating a tool_use turn followed by
      a final text turn.
    """

    def __init__(self, reply: str = "acknowledged.", responses: list[LLMResponse] | None = None):
        self._reply = reply
        self._responses = list(responses) if responses is not None else None
        self.calls: list[dict] = []

    def send(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})

        if self._responses is not None:
            if not self._responses:
                raise AssertionError("MockClient ran out of canned responses")
            return self._responses.pop(0)

        return LLMResponse(content=[TextBlock(text=self._reply)], stop_reason="end_turn")
