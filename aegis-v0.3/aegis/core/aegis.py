"""
A.E.G.I.S. -- conversational personality and primary human interface.

v0.4 adds persistent memory integration on top of the v0.3
conversation + tool architecture.

Memory responsibilities:
- Relevant memories are recalled before normal LLM requests.
- Explicit /memory commands are handled locally.
- Memory is persisted through MemoryManager.
- The LLM is NOT yet allowed to autonomously create arbitrary memories.
  Automatic memory extraction is a later v0.4 milestone.
"""

from __future__ import annotations

from typing import Callable

from core import guardian
from core.llm_client import LLMResponse, TextBlock, ToolUseBlock
from core.tools import TOOLS, anthropic_tool_schemas
from memory import MemoryManager, MemoryType


SYSTEM_PROMPT = """You are AEGIS, the personal AI interface for the A.E.G.I.S.
platform (Autonomous Electronic Guardian Intelligence System).

You are running in v0.4.

Current capabilities:
- Conversation
- Persistent personal memory
- File and system tools
- Running-process inspection
- Open-window inspection
- Application launching and closing
- Keyboard and mouse control

Every tool call is reviewed by Guardian based on its risk level.

LOW-risk tools execute automatically.
MEDIUM/HIGH-risk tools require explicit user confirmation.

Memory:
- A.E.G.I.S. has persistent memory across sessions.
- Relevant memories may be supplied as context for a conversation.
- Do not claim to remember something unless it is present in the
  supplied memory context or the current conversation.
- Do not invent memories.
- Do not expose memory metadata unless the user asks.
- The user can explicitly manage memory through the /memory commands.

Memory is currently a foundation system. Automatic extraction of memories
from ordinary conversation is not enabled yet.

Computer control:
- Keyboard and mouse control acts on whatever window currently has focus.
- You cannot directly see the contents of the screen yet.
- Do not guess what is displayed.
- Real visual screen understanding is not implemented yet.

You do not yet have:
- Voice input/output
- Wake-word detection
- Full Guardian
- Visual screen understanding
- Sensor fusion
- GPS awareness
- Phone integration
- Physical robotics
- Autonomous vehicle control

If asked for something outside your current capabilities, say plainly that
it is not implemented yet.

Personality:
Calm, direct, competent.
Brief by default.
Detailed when the question requires it.
No excessive enthusiasm or filler.
"""


TOOL_LOOP_LIMIT = 5

ConfirmCallback = Callable[[str, str, dict], bool]


class CommandResult:
    def __init__(
        self,
        handled: bool,
        output: str | None = None,
        should_exit: bool = False,
    ):
        self.handled = handled
        self.output = output
        self.should_exit = should_exit


def _memory_help() -> str:
    return (
        "Memory commands:\n"
        "  /memory                    - show recent memories\n"
        "  /memory remember <text>    - explicitly store a memory\n"
        "  /memory recall <query>     - search memories\n"
        "  /memory recent             - show recent memories\n"
        "  /memory forget <id>        - delete one memory\n"
        "  /memory clear              - delete all memories"
    )


def handle_command(
    text: str,
    memory_manager: MemoryManager | None = None,
    confirm: ConfirmCallback | None = None,
) -> CommandResult:
    """
    Handle local slash commands before anything reaches the LLM.

    Memory commands are deliberately handled outside the LLM so that
    memory operations remain deterministic and controllable.
    """

    stripped = text.strip()

    if not stripped.startswith("/"):
        return CommandResult(handled=False)

    parts = stripped[1:].split(maxsplit=2)

    command = parts[0].lower() if parts else ""

    if command in ("quit", "exit"):
        return CommandResult(
            handled=True,
            output="Shutting down.",
            should_exit=True,
        )

    if command == "help":
        return CommandResult(
            handled=True,
            output=(
                "Available commands:\n"
                "  /help   - show this message\n"
                "  /reset  - clear conversation history\n"
                "  /memory - manage persistent memory\n"
                "  /quit   - exit\n"
                "Anything else is sent to AEGIS as conversation.\n\n"
                + _memory_help()
            ),
        )

    if command == "reset":
        return CommandResult(
            handled=True,
            output="__RESET__",
        )

    if command == "memory":
        if memory_manager is None:
            return CommandResult(
                handled=True,
                output="Memory subsystem is unavailable.",
            )

        return _handle_memory_command(
            parts[1:],
            memory_manager,
            confirm,
        )

    return CommandResult(
        handled=True,
        output=f"Unknown command: /{command}. Try /help.",
    )


def _handle_memory_command(
    args: list[str],
    memory_manager: MemoryManager,
    confirm: ConfirmCallback | None,
) -> CommandResult:

    if not args:
        return CommandResult(
            handled=True,
            output=_format_memories(
                memory_manager.recent(limit=10)
            ),
        )

    action = args[0].lower()

    if action == "help":
        return CommandResult(
            handled=True,
            output=_memory_help(),
        )

    if action == "remember":
        if len(args) < 2 or not args[1].strip():
            return CommandResult(
                handled=True,
                output="Usage: /memory remember <text>",
            )

        content = args[1].strip()

        memory = memory_manager.remember(
            content,
            memory_type=MemoryType.PREFERENCE,
            importance=0.8,
            source="explicit_command",
            explicit=True,
        )

        if memory is None:
            return CommandResult(
                handled=True,
                output="I couldn't store that memory.",
            )

        return CommandResult(
            handled=True,
            output=f"Memory saved. ID: {memory.id}",
        )

    if action == "recall":
        if len(args) < 2 or not args[1].strip():
            return CommandResult(
                handled=True,
                output="Usage: /memory recall <query>",
            )

        query = args[1].strip()
        memories = memory_manager.recall(query, limit=10)

        return CommandResult(
            handled=True,
            output=_format_memories(memories),
        )

    if action == "recent":
        return CommandResult(
            handled=True,
            output=_format_memories(
                memory_manager.recent(limit=10)
            ),
        )

    if action == "forget":
        if len(args) < 2 or not args[1].strip():
            return CommandResult(
                handled=True,
                output="Usage: /memory forget <memory-id>",
            )

        memory_id = args[1].strip()

        memory = memory_manager.store.get(memory_id)

        if memory is None:
            return CommandResult(
                handled=True,
                output=f"No memory found with ID: {memory_id}",
            )

        approved = bool(
            confirm
            and confirm(
                "memory_forget",
                guardian.RISK_HIGH,
                {"memory_id": memory_id},
            )
        )

        if not approved:
            return CommandResult(
                handled=True,
                output="Memory deletion denied by Guardian.",
            )

        deleted = memory_manager.forget(memory_id)

        return CommandResult(
            handled=True,
            output=(
                "Memory deleted."
                if deleted
                else "Memory could not be deleted."
            ),
        )

    if action == "clear":
        approved = bool(
            confirm
            and confirm(
                "memory_clear",
                guardian.RISK_HIGH,
                {},
            )
        )

        if not approved:
            return CommandResult(
                handled=True,
                output="Memory clearing denied by Guardian.",
            )

        count = memory_manager.clear()

        return CommandResult(
            handled=True,
            output=f"Deleted {count} memories.",
        )

    return CommandResult(
        handled=True,
        output=f"Unknown memory command: {action}\n\n{_memory_help()}",
    )


def _format_memories(memories) -> str:
    if not memories:
        return "No memories found."

    lines = []

    for memory in memories:
        lines.append(
            f"[{memory.id}] "
            f"{memory.memory_type.value}: "
            f"{memory.content}"
        )

    return "\n".join(lines)


def _response_to_api_content(
    response: LLMResponse,
) -> list[dict]:
    """Convert normalized response into history-compatible content."""

    blocks = []

    for block in response.content:
        if isinstance(block, TextBlock):
            blocks.append(
                {
                    "type": "text",
                    "text": block.text,
                }
            )

        elif isinstance(block, ToolUseBlock):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )

    return blocks


def _tool_result(
    tool_use_id: str,
    content: str,
) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


def _extract_text(
    response: LLMResponse,
) -> str:
    return "".join(
        block.text
        for block in response.content
        if isinstance(block, TextBlock)
    )


class AEGIS:
    """
    Main A.E.G.I.S. runtime.

    v0.4 introduces persistent memory while preserving the v0.3
    tool/Guardian architecture.
    """

    def __init__(
        self,
        llm_client,
        history_limit: int = 40,
        memory_manager: MemoryManager | None = None,
    ):
        self._llm = llm_client
        self._history: list[dict] = []
        self._history_limit = history_limit
        self._memory = memory_manager
        self.should_exit = False

    @property
    def memory(self) -> MemoryManager | None:
        return self._memory

    def reset(self) -> None:
        self._history = []

    def respond(
        self,
        user_input: str,
        confirm: ConfirmCallback | None = None,
    ) -> str:

        cmd = handle_command(
            user_input,
            memory_manager=self._memory,
            confirm=confirm,
        )

        if cmd.handled:
            if cmd.should_exit:
                self.should_exit = True

            if cmd.output == "__RESET__":
                self.reset()
                return "Conversation history cleared."

            return cmd.output or ""

        self._history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        self._trim_history()

        tools_schema = anthropic_tool_schemas()

        memory_context = self._build_memory_context(
            user_input
        )

        system_prompt = SYSTEM_PROMPT

        if memory_context:
            system_prompt += (
                "\n\n"
                "Relevant persistent memory for this request:\n"
                f"{memory_context}"
            )

        for _ in range(TOOL_LOOP_LIMIT):

            response = self._llm.send(
                system=system_prompt,
                messages=self._history,
                tools=tools_schema,
            )

            self._history.append(
                {
                    "role": "assistant",
                    "content": _response_to_api_content(response),
                }
            )

            self._trim_history()

            if response.stop_reason != "tool_use":
                return _extract_text(response)

            tool_results = []

            for block in response.content:
                if not isinstance(block, ToolUseBlock):
                    continue

                tool_results.append(
                    self._handle_tool_call(
                        block,
                        confirm,
                    )
                )

            self._history.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

            self._trim_history()

        return (
            "I wasn't able to finish that after several tool calls "
            "-- try rephrasing the request."
        )

    def _build_memory_context(
        self,
        user_input: str,
    ) -> str:
        if self._memory is None:
            return ""

        return self._memory.format_context(
            user_input,
            limit=8,
        )

    def _handle_tool_call(
        self,
        block: ToolUseBlock,
        confirm: ConfirmCallback | None,
    ) -> dict:

        tool = TOOLS.get(block.name)

        if tool is None:
            return _tool_result(
                block.id,
                f"Error: unknown tool '{block.name}'.",
            )

        decision = guardian.review(tool)

        if decision.decision == guardian.CONFIRM:

            approved = bool(
                confirm
                and confirm(
                    tool.name,
                    decision.risk_category,
                    block.input,
                )
            )

            if not approved:
                return _tool_result(
                    block.id,
                    f"Denied by Guardian: '{tool.name}' "
                    f"is {decision.risk_category} risk "
                    "and was not confirmed by the user.",
                )

        try:
            output = tool.handler(**block.input)

        except Exception as exc:  # noqa: BLE001
            output = (
                f"Error executing tool '{tool.name}': {exc}"
            )

        return _tool_result(
            block.id,
            output,
        )

    def _trim_history(self) -> None:
        """
        Keep the current working context bounded.

        Full turn-aware memory management will be addressed in a
        later v0.4 milestone.
        """

        if len(self._history) > self._history_limit:
            self._history = self._history[
                -self._history_limit:
            ]