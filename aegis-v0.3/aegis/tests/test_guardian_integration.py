import pytest

from core.guardian import (
    ALLOW,
    CONFIRM,
    GuardianDecision,
    review,
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


# =========================================================
# GUARDIAN DECISION CONTRACT
# =========================================================


def test_low_risk_tool_is_allowed():
    tool = make_tool(
        risk_category=RISK_LOW
    )

    decision = review(
        tool
    )

    assert isinstance(
        decision,
        GuardianDecision,
    )

    assert (
        decision.decision
        == ALLOW
    )

    assert (
        decision.risk_category
        == RISK_LOW
    )

    assert (
        decision.tool_name
        == tool.name
    )


def test_medium_risk_tool_requires_confirmation():
    tool = make_tool(
        risk_category=RISK_MEDIUM
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )

    assert (
        decision.risk_category
        == RISK_MEDIUM
    )


def test_high_risk_tool_requires_confirmation():
    tool = make_tool(
        risk_category=RISK_HIGH
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )

    assert (
        decision.risk_category
        == RISK_HIGH
    )


# =========================================================
# FAIL-CLOSED BEHAVIOR
# =========================================================


def test_unknown_risk_category_requires_confirmation():
    tool = make_tool(
        risk_category="unknown-risk"
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )

    assert (
        decision.risk_category
        == "unknown-risk"
    )


def test_empty_risk_category_requires_confirmation():
    tool = make_tool(
        risk_category=""
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )


def test_none_risk_category_requires_confirmation():
    tool = make_tool(
        risk_category=None
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )


# =========================================================
# TOOL HANDLER SAFETY
# =========================================================


def test_guardian_review_does_not_execute_tool():
    execution = {
        "called": False
    }

    def handler(**kwargs):
        execution["called"] = True
        return "executed"

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )

    assert (
        execution["called"]
        is False
    )


def test_guardian_review_only_evaluates_tool():
    execution = {
        "called": False
    }

    def handler(**kwargs):
        execution["called"] = True
        return "executed"

    tool = make_tool(
        name="dangerous_test_tool",
        risk_category=RISK_MEDIUM,
        handler=handler,
    )

    review(
        tool
    )

    assert (
        execution["called"]
        is False
    )


# =========================================================
# DECISION IDENTITY
# =========================================================


def test_guardian_decision_contains_tool_identity():
    tool = make_tool(
        name="system_info",
        risk_category=RISK_LOW,
    )

    decision = review(
        tool
    )

    assert (
        decision.tool_name
        == "system_info"
    )


def test_guardian_decision_contains_risk_identity():
    tool = make_tool(
        risk_category=RISK_HIGH
    )

    decision = review(
        tool
    )

    assert (
        decision.risk_category
        == RISK_HIGH
    )


def test_guardian_decision_is_independent_per_review():
    low_tool = make_tool(
        name="safe_tool",
        risk_category=RISK_LOW,
    )

    high_tool = make_tool(
        name="dangerous_tool",
        risk_category=RISK_HIGH,
    )

    low_decision = review(
        low_tool
    )

    high_decision = review(
        high_tool
    )

    assert (
        low_decision.decision
        == ALLOW
    )

    assert (
        high_decision.decision
        == CONFIRM
    )

    assert (
        low_decision.tool_name
        == "safe_tool"
    )

    assert (
        high_decision.tool_name
        == "dangerous_tool"
    )


# =========================================================
# HANDLER EXECUTION CONTRACT
# =========================================================


def test_allowed_tool_can_be_executed_after_review():
    execution = {
        "called": False
    }

    def handler(**kwargs):
        execution["called"] = True
        return "executed"

    tool = make_tool(
        risk_category=RISK_LOW,
        handler=handler,
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == ALLOW
    )

    result = tool.handler()

    assert (
        result
        == "executed"
    )

    assert (
        execution["called"]
        is True
    )


def test_confirmation_required_tool_is_not_executed_by_review():
    execution = {
        "called": False
    }

    def handler(**kwargs):
        execution["called"] = True
        return "executed"

    tool = make_tool(
        risk_category=RISK_MEDIUM,
        handler=handler,
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )

    assert (
        execution["called"]
        is False
    )


def test_high_risk_tool_is_not_executed_by_review():
    execution = {
        "called": False
    }

    def handler(**kwargs):
        execution["called"] = True
        return "executed"

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    decision = review(
        tool
    )

    assert (
        decision.decision
        == CONFIRM
    )

    assert (
        execution["called"]
        is False
    )


# =========================================================
# REVIEW DOES NOT MODIFY TOOL
# =========================================================


def test_guardian_review_does_not_modify_tool():
    tool = make_tool(
        name="test_tool",
        risk_category=RISK_MEDIUM,
    )

    original_name = tool.name
    original_description = (
        tool.description
    )
    original_risk = (
        tool.risk_category
    )
    original_parameters = (
        tool.parameters
    )
    original_handler = (
        tool.handler
    )

    review(
        tool
    )

    assert (
        tool.name
        == original_name
    )

    assert (
        tool.description
        == original_description
    )

    assert (
        tool.risk_category
        == original_risk
    )

    assert (
        tool.parameters
        == original_parameters
    )

    assert (
        tool.handler
        is original_handler
    )


# =========================================================
# INVALID INPUT
# =========================================================


def test_review_rejects_none_tool():
    with pytest.raises(
        (TypeError, AttributeError)
    ):
        review(None)


def test_review_rejects_non_tool_object():
    with pytest.raises(
        (TypeError, AttributeError)
    ):
        review(
            "not-a-tool"
        )


# =========================================================
# NO SIDE EFFECTS
# =========================================================


def test_repeated_review_has_no_side_effects():
    execution = {
        "called": False
    }

    def handler(**kwargs):
        execution["called"] = True
        return "executed"

    tool = make_tool(
        risk_category=RISK_HIGH,
        handler=handler,
    )

    first = review(
        tool
    )

    second = review(
        tool
    )

    assert (
        first.decision
        == CONFIRM
    )

    assert (
        second.decision
        == CONFIRM
    )

    assert (
        first.tool_name
        == second.tool_name
    )

    assert (
        first.risk_category
        == second.risk_category
    )

    assert (
        execution["called"]
        is False
    )