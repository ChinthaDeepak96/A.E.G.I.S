"""
A.E.G.I.S. Tool Security Review.

Combines persisted audit evidence, security analysis, and
security response policy into one read-only review operation.

This module does not execute tools and does not modify audit
records or Guardian decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.tool_audit_security import (
    SecurityFinding,
)
from core.tool_audit_store import (
    SQLiteToolAuditStore,
)
from core.tool_security_policy import (
    SecurityPolicyDecision,
    ToolSecurityPolicy,
)


@dataclass(frozen=True)
class ToolSecurityReview:
    """
    Result of one security review pass.
    """

    findings: list[SecurityFinding]
    decisions: list[SecurityPolicyDecision]


class ToolSecurityReviewer:
    """
    Read-only orchestration layer for security review.
    """

    def __init__(
        self,
        store: SQLiteToolAuditStore,
        policy: ToolSecurityPolicy | None = None,
    ):
        if not isinstance(
            store,
            SQLiteToolAuditStore,
        ):
            raise TypeError(
                "store must be a SQLiteToolAuditStore."
            )

        self._store = store

        self._policy = (
            policy
            if policy is not None
            else ToolSecurityPolicy()
        )

    # =========================================================
    # REVIEW
    # =========================================================

    def review(
        self,
    ) -> ToolSecurityReview:
        """
        Analyze the current audit snapshot and evaluate
        all resulting security findings through policy.
        """

        analyzer = (
            self._store.security_analysis()
        )

        findings = analyzer.findings()

        decisions = (
            self._policy.evaluate_all(
                findings
            )
        )

        return ToolSecurityReview(
            findings=list(
                findings
            ),
            decisions=list(
                decisions
            ),
        )

    # =========================================================
    # CONVENIENCE ACCESSORS
    # =========================================================

    def findings(
        self,
    ) -> list[SecurityFinding]:
        """
        Return current security findings.
        """

        return self.review().findings

    def decisions(
        self,
    ) -> list[SecurityPolicyDecision]:
        """
        Return current security policy decisions.
        """

        return self.review().decisions

    def blocked(
        self,
    ) -> list[SecurityPolicyDecision]:
        """
        Return decisions whose policy action is BLOCK.
        """

        return [
            decision
            for decision in self.decisions()
            if decision.action == "BLOCK"
        ]

    def review_required(
        self,
    ) -> list[SecurityPolicyDecision]:
        """
        Return decisions whose policy action is REVIEW.
        """

        return [
            decision
            for decision in self.decisions()
            if decision.action == "REVIEW"
        ]