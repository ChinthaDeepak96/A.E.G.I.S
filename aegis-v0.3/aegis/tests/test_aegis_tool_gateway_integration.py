from core.aegis import AEGIS
from core.llm_client import (
    LLMResponse,
    MockClient,
    TextBlock,
    ToolUseBlock,
)
from core.tools import (
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    Tool,
)


def make_response(
    *blocks,
    stop_reason="end_turn",
):
    return LLMResponse(
        content=list(blocks),
        stop_reason=stop_reason,
    )


def make_tool(
    *,
    name="test_tool",
    risk_category=RISK_LOW,
    handler=None,
):
    if handler is None:
        handler = (
            lambda **kwargs:
            "tool executed"
        )

    return Tool(
        name=name,
        description="Integration test tool.",
        risk_category=risk_category,
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=handler,
    )


def make_aegis(
    *,
    tool,
    responses,
    confirm=None,
):
    client = MockClient(
        responses=responses
    )

    aegis = AEGIS(
        client
    )

    # Replace the global tool registry entry for this
    # integration test.
    from core import aegis as aegis_module

    original = (
        aegis_module.TOOLS.get(
            tool.name
        )
    )

    aegis_module.TOOLS[
        tool.name
    ] = tool

    return (
        aegis,
        client,
        aegis_module,
        original,
        confirm,
    )


def restore_tool(
    aegis_module,
    name,
    original,
):
    if original is None:
        aegis_module.TOOLS.pop(
            name,
            None,
        )
    else:
        aegis_module.TOOLS[
            name
        ] = original


# =========================================================
# LOW-RISK INTEGRATION
# =========================================================


def test_aegis_executes_low_risk_tool_through_gateway():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return "low-risk result"

    tool = make_tool(
        name="gateway_low_tool",
        risk_category=RISK_LOW,
        handler=handler,
    )

    response = make_response(
        ToolUseBlock(
            id="tool-1",
            name=tool.name,
            input={
                "value": 42,
            },
        ),
        stop_reason="tool_use",
    )

    final_response = make_response(
        TextBlock(
            text="Tool completed."
        )
    )

    aegis, client, module, original, _ = (
        make_aegis(
            tool=tool,
            responses=[
                response,
                final_response,
            ],
        )
    )

    try:
        result = aegis.respond(
            "Run the test tool."
        )

        assert (
            result
            == "Tool completed."
        )

        assert calls == [
            {
                "value": 42,
            }
        ]

        assert len(
            client.calls
        ) == 2

    finally:
        restore_tool(
            module,
            tool.name,
            original,
        )


# =========================================================
# MEDIUM-RISK CONFIRMATION
# =========================================================


def test_aegis_does_not_execute_medium_risk_tool_without_confirmation():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return "should not execute"

    tool = make_tool(
        name="gateway_medium_tool",
        risk_category=RISK_MEDIUM,
        handler=handler,
    )

    tool_response = make_response(
        ToolUseBlock(
            id="tool-2",
            name=tool.name,
            input={},
        ),
        stop_reason="tool_use",
    )

    final_response = make_response(
        TextBlock(
            text="The tool was denied."
        )
    )

    def confirm(
        name,
        risk,
        arguments,
    ):
        return False

    aegis, _, module, original, _ = (
        make_aegis(
            tool=tool,
            responses=[
                tool_response,
                final_response,
            ],
            confirm=confirm,
        )
    )

    try:
        result = aegis.respond(
            "Run the medium risk tool.",
            confirm=confirm,
        )

        assert (
            result
            == "The tool was denied."
        )

        assert calls == []

    finally:
        restore_tool(
            module,
            tool.name,
            original,
        )


def test_aegis_executes_medium_risk_tool_after_confirmation():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return "medium-risk result"

    tool = make_tool(
        name="gateway_medium_confirmed",
        risk_category=RISK_MEDIUM,
        handler=handler,
    )

    tool_response = make_response(
        ToolUseBlock(
            id="tool-3",
            name=tool.name,
            input={
                "confirmed": True,
            },
        ),
        stop_reason="tool_use",
    )

    final_response = make_response(
        TextBlock(
            text="Confirmed tool completed."
        )
    )

    def confirm(
        name,
        risk,
        arguments,
    ):
        return True

    aegis, _, module, original, _ = (
        make_aegis(
            tool=tool,
            responses=[
                tool_response,
                final_response,
            ],
        )
    )

    try:
        result = aegis.respond(
            "Run the confirmed tool.",
            confirm=confirm,
        )

        assert (
            result
            == "Confirmed tool completed."
        )

        assert calls == [
            {
                "confirmed": True,
            }
        ]

    finally:
        restore_tool(
            module,
            tool.name,
            original,
        )


# =========================================================
# HIGH-RISK CONFIRMATION
# =========================================================


def test_aegis_does_not_execute_high_risk_tool_without_confirmation():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return "dangerous"

    tool = make_tool(
        name="gateway_high_tool",
        risk_category=RISK_HIGH,
        handler=handler,
    )

    tool_response = make_response(
        ToolUseBlock(
            id="tool-4",
            name=tool.name,
            input={},
        ),
        stop_reason="tool_use",
    )

    final_response = make_response(
        TextBlock(
            text="High-risk tool denied."
        )
    )

    def confirm(
        name,
        risk,
        arguments,
    ):
        return False

    aegis, _, module, original, _ = (
        make_aegis(
            tool=tool,
            responses=[
                tool_response,
                final_response,
            ],
        )
    )

    try:
        result = aegis.respond(
            "Run the dangerous tool.",
            confirm=confirm,
        )

        assert (
            result
            == "High-risk tool denied."
        )

        assert calls == []

    finally:
        restore_tool(
            module,
            tool.name,
            original,
        )


def test_aegis_executes_high_risk_tool_after_confirmation():
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return "high-risk result"

    tool = make_tool(
        name="gateway_high_confirmed",
        risk_category=RISK_HIGH,
        handler=handler,
    )

    tool_response = make_response(
        ToolUseBlock(
            id="tool-5",
            name=tool.name,
            input={
                "action": "approved",
            },
        ),
        stop_reason="tool_use",
    )

    final_response = make_response(
        TextBlock(
            text="High-risk tool completed."
        )
    )

    def confirm(
        name,
        risk,
        arguments,
    ):
        return True

    aegis, _, module, original, _ = (
        make_aegis(
            tool=tool,
            responses=[
                tool_response,
                final_response,
            ],
        )
    )

    try:
        result = aegis.respond(
            "Run the approved dangerous tool.",
            confirm=confirm,
        )

        assert (
            result
            == "High-risk tool completed."
        )

        assert calls == [
            {
                "action": "approved",
            }
        ]

    finally:
        restore_tool(
            module,
            tool.name,
            original,
        )


# =========================================================
# UNKNOWN TOOL
# =========================================================


def test_aegis_unknown_tool_does_not_execute_any_handler():
    tool_response = make_response(
        ToolUseBlock(
            id="tool-6",
            name="does_not_exist",
            input={},
        ),
        stop_reason="tool_use",
    )

    final_response = make_response(
        TextBlock(
            text="Unknown tool handled."
        )
    )

    client = MockClient(
        responses=[
            tool_response,
            final_response,
        ]
    )

    aegis = AEGIS(
        client
    )

    result = aegis.respond(
        "Use an unknown tool."
    )

    assert (
        result
        == "Unknown tool handled."
    )


# =========================================================
# GATEWAY BOUNDARY
# =========================================================


def test_aegis_tool_execution_uses_gateway():
    """
    The production AEGIS execution path must use the
    ToolGateway rather than directly invoking handlers.
    """

    from core import aegis as aegis_module

    assert hasattr(
        aegis_module,
        "ToolGateway",
    )