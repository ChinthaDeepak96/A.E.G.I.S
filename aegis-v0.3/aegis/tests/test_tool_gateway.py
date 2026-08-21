import pytest

from core.guardian import (
    ALLOW,
    CONFIRM,
)
from core.tool_gateway import (
    ToolExecutionResult,
    ToolGateway,
)
from core.tools import (
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    Tool,
)


# =========================================================
# TEST HELPERS
# =========================================================


def make_tool(
    *,
    name="test_tool",
    risk_category=RISK_LOW,
    handler=None,
):
    if handler is None:
        handler = (
            lambda **kwargs:
            "executed"
        )

    return Tool(
        name=name,
        description="Test tool.",
        risk_category=risk_category,
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=handler,
    )


def make_gateway():
    return ToolGateway()


# =========================================================
# BASIC RESULT CONTRACT
# =========================================================


def test_low_risk_tool_executes():
    gateway = make_gateway()

    tool = make_tool(
        risk_category=RISK_LOW
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert isinstance(
        result,
        ToolExecutionResult,
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.result
        == "executed"
    )

    assert (
        result.decision
        == ALLOW
    )

    assert (
        result.requires_confirmation
        is False
    )


def test_medium_risk_tool_requires_confirmation():
    gateway = make_gateway()

    tool = make_tool(
        risk_category=RISK_MEDIUM
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.requires_confirmation
        is True
    )

    assert (
        result.decision
        == CONFIRM
    )

    assert (
        result.result
        is None
    )


def test_high_risk_tool_requires_confirmation():
    gateway = make_gateway()

    tool = make_tool(
        risk_category=RISK_HIGH
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.requires_confirmation
        is True
    )

    assert (
        result.decision
        == CONFIRM
    )


# =========================================================
# CONFIRMED EXECUTION
# =========================================================


def test_medium_risk_tool_executes_after_confirmation():
    gateway = make_gateway()

    tool = make_tool(
        risk_category=RISK_MEDIUM
    )

    result = gateway.execute(
        tool,
        {},
        confirmed=True,
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.result
        == "executed"
    )

    assert (
        result.requires_confirmation
        is False
    )

    assert (
        result.decision
        == CONFIRM
    )


def test_high_risk_tool_executes_after_confirmation():
    gateway = make_gateway()

    tool = make_tool(
        risk_category=RISK_HIGH
    )

    result = gateway.execute(
        tool,
        {},
        confirmed=True,
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.result
        == "executed"
    )


# =========================================================
# CONFIRMATION SAFETY
# =========================================================


def test_medium_risk_handler_is_not_called_without_confirmation():
    gateway = make_gateway()

    state = {
        "called": False
    }

    def handler(**kwargs):
        state["called"] = True
        return "executed"

    tool = make_tool(
        risk_category=RISK_MEDIUM,
        handler=handler,
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.executed
        is False
    )

    assert (
        state["called"]
        is False
    )


def test_high_risk_handler_is_not_called_without_confirmation():
    gateway = make_gateway()

    state = {
        "called": False
    }

    def handler(**kwargs):
        state["called"] = True
        return "executed"

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.executed
        is False
    )

    assert (
        state["called"]
        is False
    )


def test_confirmation_allows_handler_execution():
    gateway = make_gateway()

    state = {
        "called": False
    }

    def handler(**kwargs):
        state["called"] = True
        return "confirmed execution"

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    result = gateway.execute(
        tool,
        {
            "example": "value"
        },
        confirmed=True,
    )

    assert (
        result.executed
        is True
    )

    assert (
        state["called"]
        is True
    )

    assert (
        result.result
        == "confirmed execution"
    )


# =========================================================
# ARGUMENT PASSING
# =========================================================


def test_tool_arguments_are_passed_to_handler():
    gateway = make_gateway()

    received = {}

    def handler(**kwargs):
        received.update(
            kwargs
        )
        return "ok"

    tool = make_tool(
        handler=handler
    )

    arguments = {
        "path": "test.txt",
        "limit": 10,
    }

    result = gateway.execute(
        tool,
        arguments,
    )

    assert (
        result.executed
        is True
    )

    assert (
        received
        == arguments
    )


def test_empty_arguments_are_supported():
    gateway = make_gateway()

    tool = make_tool(
        handler=lambda **kwargs:
            "no arguments"
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.result
        == "no arguments"
    )


# =========================================================
# UNKNOWN RISK
# =========================================================


def test_unknown_risk_requires_confirmation():
    gateway = make_gateway()

    tool = make_tool(
        risk_category="unknown-risk"
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.requires_confirmation
        is True
    )

    assert (
        result.decision
        == CONFIRM
    )


def test_unknown_risk_can_execute_after_confirmation():
    gateway = make_gateway()

    tool = make_tool(
        risk_category="unknown-risk"
    )

    result = gateway.execute(
        tool,
        {},
        confirmed=True,
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.result
        == "executed"
    )


# =========================================================
# HANDLER ERRORS
# =========================================================


def test_handler_exception_does_not_escape_gateway():
    gateway = make_gateway()

    def handler(**kwargs):
        raise RuntimeError(
            "simulated failure"
        )

    tool = make_tool(
        handler=handler
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.error
        is not None
    )

    assert (
        "simulated failure"
        in result.error
    )


def test_confirmed_handler_exception_is_reported():
    gateway = make_gateway()

    def handler(**kwargs):
        raise ValueError(
            "confirmed failure"
        )

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    result = gateway.execute(
        tool,
        {},
        confirmed=True,
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.error
        is not None
    )

    assert (
        "confirmed failure"
        in result.error
    )


# =========================================================
# INVALID INPUT
# =========================================================


def test_none_tool_is_rejected():
    gateway = make_gateway()

    with pytest.raises(
        (TypeError, AttributeError)
    ):
        gateway.execute(
            None,
            {},
        )


def test_non_tool_object_is_rejected():
    gateway = make_gateway()

    with pytest.raises(
        (TypeError, AttributeError)
    ):
        gateway.execute(
            "not-a-tool",
            {},
        )


def test_none_arguments_are_rejected():
    gateway = make_gateway()

    tool = make_tool()

    with pytest.raises(
        (TypeError, ValueError)
    ):
        gateway.execute(
            tool,
            None,
        )


# =========================================================
# RESULT METADATA
# =========================================================


def test_result_contains_tool_name():
    gateway = make_gateway()

    tool = make_tool(
        name="system_info"
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.tool_name
        == "system_info"
    )


def test_confirmation_result_contains_tool_name():
    gateway = make_gateway()

    tool = make_tool(
        name="run_command",
        risk_category=RISK_HIGH,
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.tool_name
        == "run_command"
    )


def test_result_contains_risk_category():
    gateway = make_gateway()

    tool = make_tool(
        risk_category=RISK_MEDIUM
    )

    result = gateway.execute(
        tool,
        {},
    )

    assert (
        result.risk_category
        == RISK_MEDIUM
    )


# =========================================================
# NO SIDE EFFECTS DURING REVIEW
# =========================================================


def test_rejected_execution_does_not_call_handler():
    gateway = make_gateway()

    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return "executed"

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    first = gateway.execute(
        tool,
        {"value": 1},
    )

    second = gateway.execute(
        tool,
        {"value": 2},
    )

    assert (
        first.executed
        is False
    )

    assert (
        second.executed
        is False
    )

    assert calls == []


def test_confirmed_execution_calls_handler_once():
    gateway = make_gateway()

    calls = []

    def handler(**kwargs):
        calls.append(kwargs)
        return "executed"

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    result = gateway.execute(
        tool,
        {"value": 42},
        confirmed=True,
    )

    assert (
        result.executed
        is True
    )

    assert (
        len(calls)
        == 1
    )

    assert (
        calls[0]
        == {"value": 42}
    )