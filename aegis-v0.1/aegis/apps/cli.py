"""
v0.1 entrypoint. Run with:  python -m apps.cli

This is a text-only loop on purpose (see roadmap: voice moves to
v0.5, after the core reasoning loop is stable). Talking to MAX
through print/input is slower to type but much faster to debug than
a wake-word / speech-to-text / text-to-speech round-trip.
"""

import sys

from core.config import load_settings
from core.llm_client import AnthropicClient
from core.max import MAX


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
    )
    max_instance = MAX(client, history_limit=settings.history_limit)

    print("A.E.G.I.S. v0.1 -- MAX is online. Type /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

        if not user_input:
            continue

        reply = max_instance.respond(user_input)
        print(f"MAX> {reply}\n")

        if max_instance.should_exit:
            break


if __name__ == "__main__":
    main()
