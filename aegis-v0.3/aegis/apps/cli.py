"""
A.E.G.I.S. v0.4 CLI entrypoint.

Persistent memory is initialized here and passed into the AEGIS runtime.
The CLI remains the human confirmation boundary for Guardian-protected
operations.
"""

import sys

from core.config import load_settings
from core.llm_client import AnthropicClient, OllamaClient
from core.aegis import AEGIS
from memory import MemoryManager


def confirm_action(
    tool_name: str,
    risk: str,
    params: dict,
) -> bool:
    print(
        f"\n[Guardian] AEGIS wants to run "
        f"'{tool_name}' (risk: {risk}) "
        f"with input: {params}"
    )

    answer = input("Allow? [y/N] ").strip().lower()

    return answer == "y"


def main() -> None:

    try:
        settings = load_settings()

    except RuntimeError as exc:
        print(
            f"Startup error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if settings.provider == "anthropic":
        client = AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.model,
            max_tokens=settings.max_tokens,
        )
    else:
        client = OllamaClient(
            model=settings.ollama_model,
            host=settings.ollama_host,
            max_tokens=settings.max_tokens,
        )

    # ---------------------------------------------------------
    # v0.4 Persistent Memory
    # ---------------------------------------------------------

    memory_manager = MemoryManager(
        "data/aegis_memory.db"
    )

    aegis = AEGIS(
        client,
        history_limit=settings.history_limit,
        memory_manager=memory_manager,
    )

    print(
        f"A.E.G.I.S. v0.4 -- AEGIS is online "
        f"(provider: {settings.provider}). "
        f"Persistent memory: enabled. "
        f"Type /help for commands, /quit to exit.\n"
    )

    while True:

        try:
            user_input = input("you> ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if not user_input:
            continue

        try:
            reply = aegis.respond(
                user_input,
                confirm=confirm_action,
            )

        except RuntimeError as exc:
            print(
                f"AEGIS> [connection error] {exc}\n"
            )
            continue

        print(
            f"AEGIS> {reply}\n"
        )

        if aegis.should_exit:
            break


if __name__ == "__main__":
    main()