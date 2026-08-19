"""
AEGIS -- the conversational personality and primary human interface
(architecture doc section 6; renamed from "MAX" per project owner's
request -- the assistant now identifies itself as AEGIS directly
rather than under a separate character name).

v0.2 adds tool use: AEGIS can now decide to call a tool, but every
call is reviewed by Guardian Lite (core/guardian.py) first. LOW-risk
tools execute automatically; MEDIUM/HIGH-risk tools are blocked
unless the caller supplies a `confirm` callback that approves them.
No memory beyond the current session, no full Guardian, no voice
yet -- those arrive in later versions.
"""

from __future__ import annotations

from typing import Callable

from core import guardian
from core.llm_client import LLMResponse, TextBlock, ToolUseBlock
from core.tools import TOOLS, anthropic_tool_schemas

SYSTEM_PROMPT = """You are AEGIS, the personal AI interface for the A.E.G.I.S. \
platform (Autonomous Electronic Guardian Intelligence System).

You are running in v0.2: you have access to a small set of tools \
(listing/reading files, checking system info, and running shell \
commands), each gated by Guardian based on its risk level. Low-risk \
tools run automatically; anything riskier requires the user's \
explicit confirmation, which you should wait for rather than \
assuming approval -- if a tool call comes back denied, say so \
plainly rather than trying to route around it. You do not yet have \
persistent memory across sessions, a full agent system, vision, or \
voice -- if asked for something beyond your current tools, say \
plainly that it isn't built yet.

Personality: calm, direct, competent. Brief by default; detailed \
when the question calls for it. No excessive enthusiasm, no filler."""

# Bounded retries (architecture doc section 22): AEGIS should not
# loop on itself indefinitely if the model keeps requesting tools
# without ever reaching a final answer.
TOOL_LOOP_LIMIT = 5

ConfirmCallback = Callable[[str, str, dict], bool]


class CommandResult:
    def __init__(self, handled: bool, output: str | None = None, should_exit: bool = False):
        self.handled = handled
        self.output = output
        self.should_exit = should_exit


def handle_command(text: str) -> CommandResult:
    """
    Basic command handling (v0.1 scope, still in effect). Slash
    commands are intercepted before anything reaches the LLM.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return CommandResult(handled=False)

    command, *_ = stripped[1:].split(maxsplit=1)
    command = command.lower()

    if command in ("quit", "exit"):
        return CommandResult(handled=True, output="Shutting down.", should_exit=True)
    if command == "help":
        return CommandResult(
            handled=True,
            output=(
                "Available commands:\n"
                "  /help   - show this message\n"
                "  /reset  - clear conversation history\n"
                "  /quit   - exit\n"
                "Anything else is sent to AEGIS as conversation."
            ),
        )
    if command == "reset":
        return CommandResult(handled=True, output="__RESET__")

    return CommandResult(handled=True, output=f"Unknown command: /{command}. Try /help.")


def _response_to_api_content(response: LLMResponse) -> list[dict]:
    """Convert a normalized LLMResponse back into Anthropic API message-content
    format, so it can be stored in history and replayed on the next call."""
    blocks = []
    for block in response.content:
        if isinstance(block, TextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    return blocks


def _tool_result(tool_use_id: str, content: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def _extract_text(response: LLMResponse) -> str:
    return "".join(b.text for b in response.content if isinstance(b, TextBlock))


class AEGIS:
    """
    Owns conversation state (working memory only, for v0.1/v0.2) and
    routes input between command handling, tool use, and the LLM.
    """

    def __init__(self, llm_client, history_limit: int = 40):
        self._llm = llm_client
        self._history: list[dict] = []
        self._history_limit = history_limit
        self.should_exit = False

    def reset(self) -> None:
        self._history = []

    def respond(self, user_input: str, confirm: ConfirmCallback | None = None) -> str:
        cmd = handle_command(user_input)
        if cmd.handled:
            if cmd.should_exit:
                self.should_exit = True
            if cmd.output == "__RESET__":
                self.reset()
                return "Conversation history cleared."
            return cmd.output or ""

        self._history.append({"role": "user", "content": user_input})
        self._trim_history()

        tools_schema = anthropic_tool_schemas()

        for _ in range(TOOL_LOOP_LIMIT):
            response = self._llm.send(system=SYSTEM_PROMPT, messages=self._history, tools=tools_schema)
            self._history.append({"role": "assistant", "content": _response_to_api_content(response)})
            self._trim_history()

            if response.stop_reason != "tool_use":
                return _extract_text(response)

            tool_results = []
            for block in response.content:
                if not isinstance(block, ToolUseBlock):
                    continue
                tool_results.append(self._handle_tool_call(block, confirm))

            self._history.append({"role": "user", "content": tool_results})
            self._trim_history()

        return "I wasn't able to finish that after several tool calls -- try rephrasing the request."

    def _handle_tool_call(self, block: ToolUseBlock, confirm: ConfirmCallback | None) -> dict:
        tool = TOOLS.get(block.name)
        if tool is None:
            return _tool_result(block.id, f"Error: unknown tool '{block.name}'.")

        decision = guardian.review(tool)
        if decision.decision == guardian.CONFIRM:
            approved = bool(confirm and confirm(tool.name, decision.risk_category, block.input))
            if not approved:
                return _tool_result(
                    block.id,
                    f"Denied by Guardian: '{tool.name}' is {decision.risk_category} risk "
                    "and was not confirmed by the user.",
                )

        try:
            output = tool.handler(**block.input)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the model, not swallowed
            output = f"Error executing tool '{tool.name}': {exc}"

        return _tool_result(block.id, output)

    def _trim_history(self) -> None:
        # Trims after every append so what's sent to the LLM always stays
        # within history_limit. Known limitation: with a very small
        # history_limit and an in-progress multi-step tool exchange, this
        # can in principle split a tool_use from its matching tool_result,
        # which the Anthropic API requires to stay paired. Not a practical
        # issue at the default limit (40); full turn-aware trimming lands
        # with proper memory management in v0.4.
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]
