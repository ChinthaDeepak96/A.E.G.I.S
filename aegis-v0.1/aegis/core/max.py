"""
MAX -- the conversational personality and primary human interface
(architecture doc section 6). This is v0.1's "Core Brain":
conversation + personality + basic command handling. No tools, no
memory beyond the current session, no Guardian yet -- those arrive
in v0.2 onward.

Per architecture doc section 3.2: MAX != LLM. This class owns
personality, conversation state, and command routing; it delegates
only actual text generation to an LLMClient.
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are MAX, the personal AI interface for the A.E.G.I.S. \
platform (Autonomous Electronic Guardian Intelligence System).

You are running in v0.1: a text conversation engine with no tools, \
no persistent memory beyond this session, and no ability to act on \
any system yet. If asked to do something beyond conversation, say \
plainly that the capability isn't built yet rather than pretending \
to do it.

Personality: calm, direct, competent. Brief by default; detailed \
when the question calls for it. No excessive enthusiasm, no filler."""


class CommandResult:
    def __init__(self, handled: bool, output: str | None = None, should_exit: bool = False):
        self.handled = handled
        self.output = output
        self.should_exit = should_exit


def handle_command(text: str) -> CommandResult:
    """
    Basic command handling (v0.1 scope, per architecture doc section
    35). Slash commands are intercepted before anything reaches the
    LLM -- this is the seam where v0.2's Tool Registry will
    eventually plug in real actions instead of these placeholders.
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
                "Anything else is sent to MAX as conversation."
            ),
        )
    if command == "reset":
        return CommandResult(handled=True, output="__RESET__")

    return CommandResult(handled=True, output=f"Unknown command: /{command}. Try /help.")


class MAX:
    """
    Owns conversation state (working memory only, for v0.1) and
    routes input between command handling and the LLM.
    """

    def __init__(self, llm_client, history_limit: int = 40):
        self._llm = llm_client
        self._history: list[dict] = []
        self._history_limit = history_limit
        self.should_exit = False

    def reset(self) -> None:
        self._history = []

    def respond(self, user_input: str) -> str:
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

        reply = self._llm.send(system=SYSTEM_PROMPT, messages=self._history)

        self._history.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply

    def _trim_history(self) -> None:
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]
