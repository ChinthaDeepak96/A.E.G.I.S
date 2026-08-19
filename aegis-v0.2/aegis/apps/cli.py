"""
v0.1/v0.2 entrypoint. Run with:  python -m apps.cli

Text-only on purpose (voice is v0.5, after the core loop is stable).
As of v0.2, this is also where the human sits behind Guardian's
CONFIRM decisions -- confirm_action() is the only place in the
whole codebase that can approve a MEDIUM/HIGH risk tool call.
"""

import sys

from core.config import load_settings
from core.llm_client import AnthropicClient, OllamaClient
from core.max import MAX


def confirm_action(tool_name: str, risk: str, params: dict) -> bool:
    print(f"\n[Guardian] MAX wants to run '{tool_name}' (risk: {risk}) with input: {params}")
    answer = input("Allow? [y/N] ").strip().lower()
    return answer == "y"


def main() -> None:
    try:
        settings = load_settings()
    except RuntimeError as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        sys.exit(1)

    client = AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.model,
        max_tokens=settings.max_tokens,
    ) if settings.provider == "anthropic" else OllamaClient(
        model=settings.ollama_model,
        host=settings.ollama_host,
        max_tokens=settings.max_tokens,
    )
    max_instance = MAX(client, history_limit=settings.history_limit)

    print(f"A.E.G.I.S. v0.2 -- MAX is online (provider: {settings.provider}). "
          f"Type /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if not user_input:
            continue

        try:
            reply = max_instance.respond(user_input, confirm=confirm_action)
        except RuntimeError as exc:
            print(f"MAX> [connection error] {exc}\n")
            continue
        print(f"MAX> {reply}\n")

        if max_instance.should_exit:
            break


if __name__ == "__main__":
    main()
