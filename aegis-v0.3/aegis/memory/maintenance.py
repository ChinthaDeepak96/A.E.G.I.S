"""
A.E.G.I.S. Memory Maintenance.

Converts MemoryHealthReport results into deterministic
maintenance proposals.

Important:

    This module does NOT modify memories.

It only answers:

    "Given the current health of this memory,
     what maintenance action should A.E.G.I.S. propose?"

Actual lifecycle mutation remains the responsibility of
MemoryManager and, eventually, the Guardian layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .health import (
    MemoryHealth,
    MemoryHealthAnalyzer,
    MemoryHealthReport,
)
from .models import Memory


class MaintenanceAction(str, Enum):
    """
    Proposed maintenance action.
    """

    NO_ACTION = "no_action"

    REVIEW = "review"

    MARK_STALE = "mark_stale"

    ARCHIVE = "archive"


@dataclass(frozen=True)
class MaintenanceProposal:
    """
    Immutable maintenance proposal.

    No memory is modified.
    """

    memory_id: str

    action: MaintenanceAction

    health: MemoryHealth

    health_score: float

    reason: str

    requires_confirmation: bool = False


class MemoryMaintenancePlanner:
    """
    Deterministic first-generation memory maintenance planner.

    Mapping:

        HEALTHY
            -> NO_ACTION

        AGING
            -> REVIEW

        STALE_CANDIDATE
            -> MARK_STALE

        ARCHIVAL_CANDIDATE
            -> ARCHIVE

    The planner only proposes actions.

    It never calls MemoryManager.
    It never modifies Memory objects.
    """

    def __init__(
        self,
        analyzer: MemoryHealthAnalyzer | None = None,
    ):
        self.analyzer = (
            analyzer
            if analyzer is not None
            else MemoryHealthAnalyzer()
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def propose(
        self,
        memory: Memory,
        *,
        now=None,
    ) -> MaintenanceProposal:
        """
        Analyze one memory and produce a maintenance proposal.

        No memory is modified.
        """

        report = self.analyzer.analyze(
            memory,
            now=now,
        )

        return self.propose_from_report(
            report
        )

    def propose_many(
        self,
        memories: list[Memory],
        *,
        now=None,
    ) -> list[MaintenanceProposal]:
        """
        Produce maintenance proposals for multiple memories.

        Input order is preserved.
        """

        reports = self.analyzer.analyze_many(
            memories,
            now=now,
        )

        return [
            self.propose_from_report(
                report
            )
            for report in reports
        ]

    def propose_from_report(
        self,
        report: MemoryHealthReport,
    ) -> MaintenanceProposal:
        """
        Convert a health report into a maintenance proposal.

        The report must already contain all required information,
        allowing this method to remain completely read-only.
        """

        if not isinstance(
            report,
            MemoryHealthReport,
        ):
            raise TypeError(
                "report must be a MemoryHealthReport"
            )

        action = self._action_for_health(
            report.health
        )

        requires_confirmation = (
            action
            in {
                MaintenanceAction.MARK_STALE,
                MaintenanceAction.ARCHIVE,
            }
        )

        reason = self._reason(
            report,
            action,
        )

        return MaintenanceProposal(
            memory_id=report.memory_id,
            action=action,
            health=report.health,
            health_score=report.health_score,
            reason=reason,
            requires_confirmation=(
                requires_confirmation
            ),
        )

    # =========================================================
    # ACTION MAPPING
    # =========================================================

    @staticmethod
    def _action_for_health(
        health: MemoryHealth,
    ) -> MaintenanceAction:
        """
        Map health classification to maintenance action.
        """

        mapping = {
            MemoryHealth.HEALTHY:
                MaintenanceAction.NO_ACTION,

            MemoryHealth.AGING:
                MaintenanceAction.REVIEW,

            MemoryHealth.STALE_CANDIDATE:
                MaintenanceAction.MARK_STALE,

            MemoryHealth.ARCHIVAL_CANDIDATE:
                MaintenanceAction.ARCHIVE,
        }

        return mapping[
            health
        ]

    # =========================================================
    # REASONS
    # =========================================================

    @staticmethod
    def _reason(
        report: MemoryHealthReport,
        action: MaintenanceAction,
    ) -> str:
        """
        Produce a deterministic explanation for the proposal.
        """

        if (
            action
            == MaintenanceAction.NO_ACTION
        ):
            return (
                "The memory is healthy and does not "
                "currently require maintenance."
            )

        if (
            action
            == MaintenanceAction.REVIEW
        ):
            return (
                "The memory is aging and should be reviewed "
                "before stronger lifecycle action is proposed."
            )

        if (
            action
            == MaintenanceAction.MARK_STALE
        ):
            return (
                "The memory is a stale candidate. Marking it "
                "stale should require explicit confirmation."
            )

        if (
            action
            == MaintenanceAction.ARCHIVE
        ):
            return (
                "The memory is an archival candidate. "
                "Archiving should require explicit confirmation."
            )

        return (
            "No maintenance action was determined."
        )