"""
A.E.G.I.S. Tool Security Response Policy.

This module translates security findings into response levels.

It does not:
    - execute tools
    - modify audit records
    - replace Guardian
    - directly block execution

It provides a deterministic policy layer that can later be
consumed by Guardian Full or the Tool Gateway.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.tool_audit_security import (
    SecurityFinding,
)


INFO = "INFO"
REVIEW = "REVIEW"
BLOCK = "BLOCK"


@dataclass(frozen=True)
class SecurityPolicyDecision:
    """
    Policy decision produced from a security finding.
    """

    action: str
    finding_type: str
    severity: str
    tool_name: str | None
    reason: str


class ToolSecurityPolicy:
    """
    Deterministic mapping from security findings to
    security response actions.
    """

    def __init__(
        self,
        *,
        block_high_risk: bool = False,
        block_repeated_failures: bool = False,
        block_repeated_denials: bool = False,
    ):
        self.block_high_risk = bool(
            block_high_risk
        )

        self.block_repeated_failures = bool(
            block_repeated_failures
        )

        self.block_repeated_denials = bool(
            block_repeated_denials
        )

    # =========================================================
    # SINGLE FINDING
    # =========================================================

    def evaluate(
        self,
        finding: SecurityFinding,
    ) -> SecurityPolicyDecision:
        """
        Convert one SecurityFinding into a policy decision.
        """

        if not isinstance(
            finding,
            SecurityFinding,
        ):
            raise TypeError(
                "finding must be a SecurityFinding."
            )

        action = self._action_for(
            finding
        )

        return SecurityPolicyDecision(
            action=action,
            finding_type=(
                finding.finding_type
            ),
            severity=finding.severity,
            tool_name=finding.tool_name,
            reason=finding.message,
        )

    # =========================================================
    # MULTIPLE FINDINGS
    # =========================================================

    def evaluate_all(
        self,
        findings: list[SecurityFinding],
    ) -> list[SecurityPolicyDecision]:
        """
        Evaluate a collection of findings.

        The input list is not modified.
        """

        if not isinstance(
            findings,
            list,
        ):
            raise TypeError(
                "findings must be a list."
            )

        return [
            self.evaluate(
                finding
            )
            for finding in findings
        ]

    # =========================================================
    # ACTION RESOLUTION
    # =========================================================

    def _action_for(
        self,
        finding: SecurityFinding,
    ) -> str:
        """
        Determine the response action for a finding.
        """

        finding_type = (
            finding.finding_type
        )

        if (
            finding_type
            == "HIGH_RISK_ACTIVITY"
        ):
            if self.block_high_risk:
                return BLOCK

            return REVIEW

        if (
            finding_type
            == "REPEATED_FAILURES"
        ):
            if self.block_repeated_failures:
                return BLOCK

            return REVIEW

        if (
            finding_type
            == "REPEATED_DENIALS"
        ):
            if self.block_repeated_denials:
                return BLOCK

            return REVIEW

        if (
            finding_type
            == "REPEATED_CONFIRMATIONS"
        ):
            return INFO

        if (
            finding_type
            == "HIGH_TOOL_FREQUENCY"
        ):
            return INFO

        # Fail closed for unknown security findings.
        return REVIEW