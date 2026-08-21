"""
A.E.G.I.S. Tool Audit Security Analysis.

This module detects suspicious patterns in persisted tool-audit
records.

It is intentionally read-only.

It does NOT:
    - execute tools
    - modify audit records
    - override Guardian decisions
    - block tool execution
    - delete audit history

Its purpose is to identify patterns that may deserve review.

Guardian remains the execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from core.tool_audit import ToolAuditRecord
from core.tool_audit_analyzer import ToolAuditAnalyzer


@dataclass(frozen=True)
class SecurityFinding:
    """
    One security-related observation derived from audit evidence.
    """

    finding_type: str
    severity: str
    tool_name: str | None
    count: int
    message: str


class ToolAuditSecurityAnalyzer:
    """
    Read-only security analysis over ToolAuditRecord objects.
    """

    def __init__(
        self,
        analyzer: ToolAuditAnalyzer,
    ):
        if not isinstance(
            analyzer,
            ToolAuditAnalyzer,
        ):
            raise TypeError(
                "analyzer must be a ToolAuditAnalyzer."
            )

        self._analyzer = analyzer

    # =========================================================
    # BASIC SECURITY SIGNALS
    # =========================================================

    def repeated_failures(
        self,
        *,
        threshold: int = 3,
    ) -> list[SecurityFinding]:
        """
        Detect tools that have failed repeatedly.

        A finding is produced when a tool has at least
        `threshold` failed audit records.
        """

        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than zero."
            )

        failures = (
            self._analyzer.failures()
        )

        counts: dict[str, int] = {}

        for record in failures:
            counts[record.tool_name] = (
                counts.get(
                    record.tool_name,
                    0,
                )
                + 1
            )

        findings: list[
            SecurityFinding
        ] = []

        for tool_name, count in counts.items():
            if count < threshold:
                continue

            findings.append(
                SecurityFinding(
                    finding_type=(
                        "REPEATED_FAILURES"
                    ),
                    severity="MEDIUM",
                    tool_name=tool_name,
                    count=count,
                    message=(
                        f"Tool '{tool_name}' "
                        f"failed {count} times."
                    ),
                )
            )

        return findings

    def repeated_denials(
        self,
        *,
        threshold: int = 3,
    ) -> list[SecurityFinding]:
        """
        Detect tools that repeatedly require confirmation
        but are not executed.
        """

        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than zero."
            )

        records = (
            self._analyzer.confirmation_required()
        )

        counts: dict[str, int] = {}

        for record in records:
            counts[record.tool_name] = (
                counts.get(
                    record.tool_name,
                    0,
                )
                + 1
            )

        findings: list[
            SecurityFinding
        ] = []

        for tool_name, count in counts.items():
            if count < threshold:
                continue

            findings.append(
                SecurityFinding(
                    finding_type=(
                        "REPEATED_DENIALS"
                    ),
                    severity="MEDIUM",
                    tool_name=tool_name,
                    count=count,
                    message=(
                        f"Tool '{tool_name}' "
                        f"was denied confirmation "
                        f"{count} times."
                    ),
                )
            )

        return findings

    # =========================================================
    # HIGH-RISK ACTIVITY
    # =========================================================

    def high_risk_activity(
        self,
        *,
        threshold: int = 1,
    ) -> list[SecurityFinding]:
        """
        Detect repeated HIGH-risk activity.

        HIGH-risk records are observations, not automatic
        violations. Guardian remains responsible for the
        authorization decision.
        """

        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than zero."
            )

        records = (
            self._analyzer.high_risk()
        )

        counts: dict[str, int] = {}

        for record in records:
            counts[record.tool_name] = (
                counts.get(
                    record.tool_name,
                    0,
                )
                + 1
            )

        findings: list[
            SecurityFinding
        ] = []

        for tool_name, count in counts.items():
            if count < threshold:
                continue

            findings.append(
                SecurityFinding(
                    finding_type=(
                        "HIGH_RISK_ACTIVITY"
                    ),
                    severity="HIGH",
                    tool_name=tool_name,
                    count=count,
                    message=(
                        f"Tool '{tool_name}' "
                        f"generated {count} "
                        f"HIGH-risk audit records."
                    ),
                )
            )

        return findings

    # =========================================================
    # CONFIRMATION PATTERNS
    # =========================================================

    def repeated_confirmations(
        self,
        *,
        threshold: int = 3,
    ) -> list[SecurityFinding]:
        """
        Detect tools that repeatedly require explicit
        confirmation and are subsequently executed.

        This is a review signal, not a policy violation.
        """

        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than zero."
            )

        records = (
            self._analyzer.confirmed()
        )

        counts: dict[str, int] = {}

        for record in records:
            counts[record.tool_name] = (
                counts.get(
                    record.tool_name,
                    0,
                )
                + 1
            )

        findings: list[
            SecurityFinding
        ] = []

        for tool_name, count in counts.items():
            if count < threshold:
                continue

            findings.append(
                SecurityFinding(
                    finding_type=(
                        "REPEATED_CONFIRMATIONS"
                    ),
                    severity="LOW",
                    tool_name=tool_name,
                    count=count,
                    message=(
                        f"Tool '{tool_name}' "
                        f"was explicitly confirmed "
                        f"{count} times."
                    ),
                )
            )

        return findings

    # =========================================================
    # FREQUENCY ANALYSIS
    # =========================================================

    def high_frequency_tools(
        self,
        *,
        threshold: int = 10,
    ) -> list[SecurityFinding]:
        """
        Detect tools with unusually high audit frequency.

        This is intentionally a simple count-based signal.
        More advanced time-window analysis can be added later.
        """

        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than zero."
            )

        counts = (
            self._analyzer.tool_usage_counts()
        )

        findings: list[
            SecurityFinding
        ] = []

        for tool_name, count in counts.items():
            if count < threshold:
                continue

            findings.append(
                SecurityFinding(
                    finding_type=(
                        "HIGH_TOOL_FREQUENCY"
                    ),
                    severity="LOW",
                    tool_name=tool_name,
                    count=count,
                    message=(
                        f"Tool '{tool_name}' "
                        f"appeared {count} times "
                        f"in the audit history."
                    ),
                )
            )

        return findings

    # =========================================================
    # COMBINED ANALYSIS
    # =========================================================

    def findings(
        self,
        *,
        failure_threshold: int = 3,
        denial_threshold: int = 3,
        confirmation_threshold: int = 3,
        frequency_threshold: int = 10,
        high_risk_threshold: int = 1,
    ) -> list[SecurityFinding]:
        """
        Return all currently supported security findings.

        The results are observations only.
        """

        findings: list[
            SecurityFinding
        ] = []

        findings.extend(
            self.repeated_failures(
                threshold=failure_threshold
            )
        )

        findings.extend(
            self.repeated_denials(
                threshold=denial_threshold
            )
        )

        findings.extend(
            self.high_risk_activity(
                threshold=high_risk_threshold
            )
        )

        findings.extend(
            self.repeated_confirmations(
                threshold=confirmation_threshold
            )
        )

        findings.extend(
            self.high_frequency_tools(
                threshold=frequency_threshold
            )
        )

        return findings