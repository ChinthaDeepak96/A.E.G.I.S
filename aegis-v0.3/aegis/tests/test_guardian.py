from core import guardian
from core.tools import RISK_HIGH, RISK_LOW, RISK_MEDIUM, Tool


def _tool(risk: str) -> Tool:
    return Tool(name="fake", description="", risk_category=risk, parameters={}, handler=lambda: "")


def test_low_risk_auto_allowed():
    decision = guardian.review(_tool(RISK_LOW))
    assert decision.decision == guardian.ALLOW


def test_medium_risk_requires_confirmation():
    decision = guardian.review(_tool(RISK_MEDIUM))
    assert decision.decision == guardian.CONFIRM


def test_high_risk_requires_confirmation():
    decision = guardian.review(_tool(RISK_HIGH))
    assert decision.decision == guardian.CONFIRM


def test_unknown_risk_category_fails_closed():
    decision = guardian.review(_tool("SOMETHING_UNRECOGNIZED"))
    assert decision.decision == guardian.CONFIRM
