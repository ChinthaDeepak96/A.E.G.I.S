"""
A.E.G.I.S. Tool Audit Analysis.

This module provides read-only analysis over persisted
ToolAuditRecord objects.

The analyzer does not:
    - execute tools
    - modify audit records
    - make Guardian decisions
    - delete audit history

Its purpose is to turn raw audit evidence into useful
operational information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.guardian import (
    CONFIRM,
)
from core.tool_audit import (
    ToolAuditRecord,
)

@dataclass(frozen=True)
class ToolAuditSummary:
    """
    Aggregate statistics for a collection of audit records.
    """

    total: int
    executed: int
    successful: int
    failed: int
    denied: int
    confirmation_required: int
    confirmed: int

    @property
    def success_rate(self) -> float:
        """
        Return successful executions as a fraction.

        Returns 0.0 when there are no records.
        """

        if self.executed == 0:
            return 0.0

        return self.successful / self.executed


class ToolAuditAnalyzer:
    """
    Read-only analysis layer over tool audit records.
    """

    def __init__(
        self,
        records: Iterable[
            ToolAuditRecord
        ],
    ):
        self._records = tuple(
            records
        )

    # =========================================================
    # RECORD ACCESS
    # =========================================================

    def records(
        self,
    ) -> list[ToolAuditRecord]:
        """
        Return a copy of the records.
        """

        return list(
            self._records
        )

    def recent(
        self,
        limit: int = 20,
    ) -> list[ToolAuditRecord]:
        """
        Return the most recent records.

        Records are ordered newest first.
        """

        if limit <= 0:
            return []

        records = sorted(
            self._records,
            key=lambda record: (
                record.timestamp
            ),
            reverse=True,
        )

        return records[:limit]

    # =========================================================
    # FILTERS
    # =========================================================

    def for_tool(
        self,
        tool_name: str,
    ) -> list[ToolAuditRecord]:
        """
        Return records belonging to one tool.
        """

        if not isinstance(
            tool_name,
            str,
        ):
            raise TypeError(
                "tool_name must be a string."
            )

        tool_name = tool_name.strip()

        if not tool_name:
            return []

        return [
            record
            for record in self._records
            if record.tool_name
            == tool_name
        ]

    def for_risk(
        self,
        risk_category: str,
    ) -> list[ToolAuditRecord]:
        """
        Return records belonging to one risk category.
        """

        if not isinstance(
            risk_category,
            str,
        ):
            raise TypeError(
                "risk_category must be a string."
            )

        risk_category = (
            risk_category.strip()
        )

        if not risk_category:
            return []

        return [
            record
            for record in self._records
            if record.risk_category
            == risk_category
        ]

    def failures(
        self,
    ) -> list[ToolAuditRecord]:
        """
        Return execution attempts that failed.
        """

        return [
            record
            for record in self._records
            if not record.success
            and (
                record.executed
                or record.error is not None
            )
        ]

    def confirmed(
        self,
    ) -> list[ToolAuditRecord]:
        """
        Return records where explicit confirmation was supplied.
        """

        return [
            record
            for record in self._records
            if record.confirmed
        ]

    def confirmation_required(
        self,
    ) -> list[ToolAuditRecord]:
        """
        Return records where Guardian required confirmation
        but confirmation was not supplied.
        """

        return [
            record
            for record in self._records
            if record.decision == CONFIRM
            and not record.confirmed
            and not record.executed
        ]

    def denied(
        self,
    ) -> list[ToolAuditRecord]:
        """
        Return records that did not execute.
        """

        return [
            record
            for record in self._records
            if not record.executed
        ]

    # =========================================================
    # HIGH-RISK ACTIVITY
    # =========================================================

    def high_risk(
        self,
    ) -> list[ToolAuditRecord]:
        """
        Return HIGH-risk audit records.

        The analyzer intentionally uses the risk category stored
        in the audit record rather than re-evaluating the tool.
        """

        return [
            record
            for record in self._records
            if record.risk_category
            .strip()
            .upper()
            == "HIGH"
        ]

    # =========================================================
    # SUMMARY
    # =========================================================

    def summarize(
        self,
    ) -> ToolAuditSummary:
        """
        Produce aggregate execution statistics.
        """

        total = len(
            self._records
        )

        executed = sum(
            1
            for record in self._records
            if record.executed
        )

        successful = sum(
            1
            for record in self._records
            if record.success
        )

        failed = sum(
            1
            for record in self._records
            if not record.success
        )

        denied = sum(
            1
            for record in self._records
            if not record.executed
        )

        confirmation_required = sum(
            1
            for record in self._records
            if record.decision == CONFIRM
            and not record.confirmed
            and not record.executed
        )

        confirmed = sum(
            1
            for record in self._records
            if record.confirmed
        )

        return ToolAuditSummary(
            total=total,
            executed=executed,
            successful=successful,
            failed=failed,
            denied=denied,
            confirmation_required=(
                confirmation_required
            ),
            confirmed=confirmed,
        )

    # =========================================================
    # TOOL COUNTS
    # =========================================================

    def tool_usage_counts(
        self,
    ) -> dict[str, int]:
        """
        Count audit records by tool name.
        """

        counts: dict[str, int] = {}

        for record in self._records:
            counts[record.tool_name] = (
                counts.get(
                    record.tool_name,
                    0,
                )
                + 1
            )

        return counts

    def successful_tool_counts(
        self,
    ) -> dict[str, int]:
        """
        Count successful executions by tool name.
        """

        counts: dict[str, int] = {}

        for record in self._records:
            if not record.success:
                continue

            counts[record.tool_name] = (
                counts.get(
                    record.tool_name,
                    0,
                )
                + 1
            )

        return counts