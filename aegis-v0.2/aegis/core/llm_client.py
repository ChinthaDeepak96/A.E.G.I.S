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


def _to_ollama_tool(anthropic_tool: dict) -> dict:
    """Convert one of our Anthropic-shaped tool schemas (core/tools.py)
    into Ollama's function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool["description"],
            "parameters": anthropic_tool["input_schema"],
        },
    }


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    """
    Convert our Anthropic-shaped history (see core/max.py, which
    stores messages in the exact shape the Anthropic API expects) into
    Ollama's simpler format. tool_use blocks become an assistant
    tool_calls entry; tool_result blocks become role="tool" messages.
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        text_parts = []
        tool_calls = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append({"function": {"name": block["name"], "arguments": block["input"]}})
            elif block.get("type") == "tool_result":
                converted.append({"role": "tool", "content": str(block["content"])})

        if text_parts or tool_calls:
            entry = {"role": role, "content": "".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            converted.append(entry)

    return converted


def _from_ollama_response(data: dict) -> LLMResponse:
    message = data.get("message", {})
    content = []

    text = message.get("content") or ""
    if text:
        content.append(TextBlock(text=text))

    tool_calls = message.get("tool_calls") or []
    for i, call in enumerate(tool_calls):
        function = call.get("function", {})
        content.append(
            ToolUseBlock(id=f"ollama_call_{i}", name=function.get("name", ""), input=function.get("arguments", {}) or {})
        )

    stop_reason = "tool_use" if tool_calls else "end_turn"
    return LLMResponse(content=content, stop_reason=stop_reason)


class OllamaClient:
    """
    Free, fully local alternative to AnthropicClient. Talks to a
    locally running Ollama server (https://ollama.com) instead of a
    paid API -- no API key, no account, no per-token cost, ever. This
    is the first real piece of the future Model Router (architecture
    doc section 51): same LLMClient interface, different provider.

    Tool-calling reliability depends on which local model you pick --
    llama3.1 and qwen2.5 handle it reasonably well; smaller or older
    models may ignore tools or hallucinate arguments more often than
    Claude does. That's a real trade-off of running for free on your
    own hardware, not a bug in this client.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434", max_tokens: int = 1024):
        self._model = model
        self._host = host.rstrip("/")
        self._max_tokens = max_tokens

    def send(self, system: str, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        import json
        import urllib.error
        import urllib.request

        ollama_messages = [{"role": "system", "content": system}]
        ollama_messages.extend(_to_ollama_messages(messages))

        payload = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": self._max_tokens},
        }
        if tools:
            payload["tools"] = [_to_ollama_tool(t) for t in tools]

        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host} -- is it running? "
                f"Start it with `ollama serve` and make sure you've pulled "
                f"a model with `ollama pull {self._model}`. ({exc})"
            ) from exc

        return _from_ollama_response(data)


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
