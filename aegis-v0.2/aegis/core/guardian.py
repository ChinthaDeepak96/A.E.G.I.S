"""
Guardian Lite (architecture doc section 5.3, v0.2 scope).

Guardian Full (v0.7) will have a real risk classifier, audit logs,
secrets management, and anomaly detection. Guardian Lite is
deliberately just a lookup table: every tool already declares its
own risk_category (core/tools.py), so this module's entire job is
to look that category up and decide ALLOW vs CONFIRM.

No tool call ever reaches its handler without going through
`review()` first -- that is the whole point of building this before
the Computer Agent, not after.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.tools import RISK_HIGH, RISK_LOW, RISK_MEDIUM, Tool

ALLOW = "ALLOW"
CONFIRM = "CONFIRM"

_DECISION_BY_RISK = {
    RISK_LOW: ALLOW,
    RISK_MEDIUM: CONFIRM,
    RISK_HIGH: CONFIRM,
}


@dataclass
class GuardianDecision:
    decision: str  # ALLOW or CONFIRM
    risk_category: str
    tool_name: str


def review(tool: Tool) -> GuardianDecision:
    # Fail closed: a risk category Guardian doesn't recognize is
    # treated as needing confirmation, never as safe to auto-run.
    decision = _DECISION_BY_RISK.get(tool.risk_category, CONFIRM)
    return GuardianDecision(decision=decision, risk_category=tool.risk_category, tool_name=tool.name)
