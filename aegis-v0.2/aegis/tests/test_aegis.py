"""
v0.1 tests verify plain conversation still works unchanged. v0.2
tests verify the actual done-when criterion for this version: AEGIS
picks the right tool for a stated goal, and Guardian correctly
blocks a MEDIUM/HIGH-risk tool call without any code change --
just the absence of a confirm callback / a confirm callback that
says no.
"""

from core.llm_client import LLMResponse, MockClient, TextBlock, ToolUseBlock
from core.aegis import AEGIS, handle_command

# ---------------------------------------------------------------------------
# v0.1: plain conversation, unchanged behavior
# ---------------------------------------------------------------------------


def test_twenty_turn_conversation_survives():
    client = MockClient(reply="acknowledged.")
    aegis = AEGIS(client, history_limit=40)

    for i in range(20):
        reply = aegis.respond(f"message number {i}")
        assert reply == "acknowledged."

    assert len(client.calls) == 20
    assert all(call["system"] for call in client.calls)


def test_history_trims_to_limit():
    client = MockClient()
    aegis = AEGIS(client, history_limit=4)

    for i in range(10):
        aegis.respond(f"message {i}")

    last_call_messages = client.calls[-1]["messages"]
    assert len(last_call_messages) <= 4


def test_reset_clears_history():
    client = MockClient()
    aegis = AEGIS(client, history_limit=40)
    aegis.respond("hello")
    assert len(client.calls[-1]["messages"]) == 1

    reply = aegis.respond("/reset")
    assert reply == "Conversation history cleared."

    aegis.respond("hello again")
    assert len(client.calls[-1]["messages"]) == 1


def test_help_command_does_not_hit_llm():
    client = MockClient()
    aegis = AEGIS(client, history_limit=40)
    reply = aegis.respond("/help")
    assert "Available commands" in reply
    assert len(client.calls) == 0


def test_quit_command_sets_exit_flag():
    client = MockClient()
    aegis = AEGIS(client, history_limit=40)
    assert aegis.should_exit is False
    aegis.respond("/quit")
    assert aegis.should_exit is True


def test_unknown_command_reports_cleanly():
    result = handle_command("/nonsense")
    assert result.handled is True
    assert "Unknown command" in result.output


# ---------------------------------------------------------------------------
# v0.2: tool use gated by Guardian Lite
# ---------------------------------------------------------------------------


def test_low_risk_tool_executes_without_confirmation():
    responses = [
        LLMResponse(
            content=[ToolUseBlock(id="t1", name="system_info", input={})],
            stop_reason="tool_use",
        ),
        LLMResponse(content=[TextBlock(text="Here is your system info.")], stop_reason="end_turn"),
    ]
    client = MockClient(responses=responses)
    aegis = AEGIS(client, history_limit=40)

    # No confirm callback -- a LOW-risk tool must not need one.
    reply = aegis.respond("what's my OS?")
    assert reply == "Here is your system info."
    assert len(client.calls) == 2


def test_high_risk_tool_blocked_without_confirmation():
    responses = [
        LLMResponse(
            content=[ToolUseBlock(id="t1", name="run_command", input={"command": "ls"})],
            stop_reason="tool_use",
        ),
        LLMResponse(content=[TextBlock(text="I wasn't able to run that.")], stop_reason="end_turn"),
    ]
    client = MockClient(responses=responses)
    aegis = AEGIS(client, history_limit=40)

    # No confirm callback at all -- Guardian must deny by default.
    reply = aegis.respond("list files via shell")
    assert reply == "I wasn't able to run that."

    second_call_messages = client.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    assert "Denied by Guardian" in tool_result_msg["content"][0]["content"]


def test_high_risk_tool_denied_when_user_says_no():
    responses = [
        LLMResponse(
            content=[ToolUseBlock(id="t1", name="run_command", input={"command": "rm -rf /"})],
            stop_reason="tool_use",
        ),
        LLMResponse(content=[TextBlock(text="Understood, not running that.")], stop_reason="end_turn"),
    ]
    client = MockClient(responses=responses)
    aegis = AEGIS(client, history_limit=40)

    reply = aegis.respond("run rm -rf /", confirm=lambda *a: False)
    assert reply == "Understood, not running that."


def test_high_risk_tool_runs_when_user_confirms():
    responses = [
        LLMResponse(
            content=[ToolUseBlock(id="t1", name="run_command", input={"command": "echo hi"})],
            stop_reason="tool_use",
        ),
        LLMResponse(content=[TextBlock(text="Done.")], stop_reason="end_turn"),
    ]
    client = MockClient(responses=responses)
    aegis = AEGIS(client, history_limit=40)

    reply = aegis.respond("run echo hi", confirm=lambda *a: True)
    assert reply == "Done."

    # The actual command output should have reached the model as a tool_result.
    second_call_messages = client.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]["content"][0]["content"]
    assert "hi" in tool_result_msg


def test_confirm_callback_receives_tool_name_risk_and_input():
    responses = [
        LLMResponse(
            content=[ToolUseBlock(id="t1", name="run_command", input={"command": "echo test"})],
            stop_reason="tool_use",
        ),
        LLMResponse(content=[TextBlock(text="Done.")], stop_reason="end_turn"),
    ]
    client = MockClient(responses=responses)
    aegis = AEGIS(client, history_limit=40)

    seen = {}

    def confirm(tool_name, risk, params):
        seen["tool_name"] = tool_name
        seen["risk"] = risk
        seen["params"] = params
        return True

    aegis.respond("run echo test", confirm=confirm)
    assert seen["tool_name"] == "run_command"
    assert seen["risk"] == "HIGH"
    assert seen["params"] == {"command": "echo test"}


def test_unknown_tool_name_reports_error_without_crashing():
    responses = [
        LLMResponse(
            content=[ToolUseBlock(id="t1", name="not_a_real_tool", input={})],
            stop_reason="tool_use",
        ),
        LLMResponse(content=[TextBlock(text="That tool doesn't exist.")], stop_reason="end_turn"),
    ]
    client = MockClient(responses=responses)
    aegis = AEGIS(client, history_limit=40)

    reply = aegis.respond("use a fake tool")
    assert reply == "That tool doesn't exist."


def test_tool_loop_gives_up_after_max_iterations():
    # Model keeps requesting the same LOW-risk tool forever and never
    # produces a final end_turn -- the bounded-retry cap must kick in
    # rather than looping indefinitely.
    endless_tool_use = LLMResponse(
        content=[ToolUseBlock(id="t1", name="system_info", input={})],
        stop_reason="tool_use",
    )
    client = MockClient(responses=[endless_tool_use] * 10)
    aegis = AEGIS(client, history_limit=100)

    reply = aegis.respond("keep checking my system info forever")
    assert "wasn't able to finish" in reply
    assert len(client.calls) == 5  # TOOL_LOOP_LIMIT
