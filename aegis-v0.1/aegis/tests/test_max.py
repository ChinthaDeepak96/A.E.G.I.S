"""
Verifies v0.1's Done-when criterion: MAX can hold a multi-turn
conversation and respond in-character with no crashes across a
20-turn session. Uses MockClient so this runs with no API key and
no network call.
"""

from core.llm_client import MockClient
from core.max import MAX, handle_command


def test_twenty_turn_conversation_survives():
    client = MockClient(reply="acknowledged.")
    max_instance = MAX(client, history_limit=40)

    for i in range(20):
        reply = max_instance.respond(f"message number {i}")
        assert reply == "acknowledged."

    assert len(client.calls) == 20
    # system prompt (personality) sent unchanged on every call
    assert all(call["system"] for call in client.calls)


def test_history_trims_to_limit():
    client = MockClient()
    max_instance = MAX(client, history_limit=4)

    for i in range(10):
        max_instance.respond(f"message {i}")

    last_call_messages = client.calls[-1]["messages"]
    assert len(last_call_messages) <= 4


def test_reset_clears_history():
    client = MockClient()
    max_instance = MAX(client, history_limit=40)
    max_instance.respond("hello")
    assert len(client.calls[-1]["messages"]) == 1

    reply = max_instance.respond("/reset")
    assert reply == "Conversation history cleared."

    max_instance.respond("hello again")
    assert len(client.calls[-1]["messages"]) == 1  # history was actually cleared


def test_help_command_does_not_hit_llm():
    client = MockClient()
    max_instance = MAX(client, history_limit=40)
    reply = max_instance.respond("/help")
    assert "Available commands" in reply
    assert len(client.calls) == 0


def test_quit_command_sets_exit_flag():
    client = MockClient()
    max_instance = MAX(client, history_limit=40)
    assert max_instance.should_exit is False
    max_instance.respond("/quit")
    assert max_instance.should_exit is True


def test_unknown_command_reports_cleanly():
    result = handle_command("/nonsense")
    assert result.handled is True
    assert "Unknown command" in result.output
