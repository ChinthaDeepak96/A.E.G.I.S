"""
A.E.G.I.S. Memory Maintenance.

Converts MemoryHealthReport results into deterministic
maintenance proposals.

Important:

    This module does NOT modify memories.

It only answers:

    "Given the current health and usage of this memory,
     what maintenance action should A.E.G.I.S. propose?"

Actual lifecycle mutation remains the responsibility of
MemoryManager and, eventually, the Guardian layer.

Usage information is observational evidence only.
It does not directly trigger mutations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .health import (
    MemoryHealth,
    MemoryHealthAnalyzer,
    MemoryHealthReport,
)
from .models import Memory
from .usage import MemoryUsage


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

    Usage fields are copied from the health report so that the
    proposal contains the evidence that contributed to the
    health assessment.
    """

    memory_id: str

    action: MaintenanceAction

    health: MemoryHealth

    health_score: float

    reason: str

    requires_confirmation: bool = False

    # =========================================================
    # USAGE EVIDENCE
    # =========================================================

    retrieval_count: int = 0

    access_count: int = 0

    days_since_retrieval: float | None = None

    has_been_retrieved: bool = False


class MemoryMaintenancePlanner:
    """
    Deterministic memory maintenance planner.

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

    Usage is supplied to the health analyzer as additional
    observational evidence.
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
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> MaintenanceProposal:
        """
        Analyze one memory and produce a maintenance proposal.

        Parameters
        ----------
        memory:
            Memory to analyze.

        usage:
            Optional usage statistics associated with the memory.

        now:
            Optional reference time for deterministic testing.

        No memory is modified.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "memory must be a Memory instance"
            )

        if usage is not None and not isinstance(
            usage,
            MemoryUsage,
        ):
            raise TypeError(
                "usage must be a MemoryUsage instance"
            )

        report = self.analyzer.analyze(
            memory,
            usage=usage,
            now=now,
        )

        return self.propose_from_report(
            report
        )

    def propose_many(
        self,
        memories: list[Memory],
        *,
        usages: dict[str, MemoryUsage] | None = None,
        now: datetime | None = None,
    ) -> list[MaintenanceProposal]:
        """
        Produce maintenance proposals for multiple memories.

        `usages` maps memory IDs to MemoryUsage objects.

        Input order is preserved.

        Missing usage entries are treated as zero usage.
        """

        if usages is None:
            usages = {}

        # -----------------------------------------------------
        # Validate the usage mapping before analysis.
        # -----------------------------------------------------

        for memory_id, usage in usages.items():

            if not isinstance(
                memory_id,
                str,
            ):
                raise TypeError(
                    "usage map keys must be strings"
                )

            if not isinstance(
                usage,
                MemoryUsage,
            ):
                raise TypeError(
                    "usage map values must be MemoryUsage instances"
                )

        reports = self.analyzer.analyze_many(
            memories,
            usages=usages,
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

        Usage information is preserved from the health report.
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
            retrieval_count=(
                report.retrieval_count
            ),
            access_count=(
                report.access_count
            ),
            days_since_retrieval=(
                report.days_since_retrieval
            ),
            has_been_retrieved=(
                report.has_been_retrieved
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

        Usage evidence can influence the wording but does not
        change the underlying action mapping.
        """

        if (
            action
            == MaintenanceAction.NO_ACTION
        ):
            if report.has_been_retrieved:
                return (
                    "The memory is healthy and has demonstrated "
                    "retrieval usage, so no maintenance is "
                    "currently required."
                )

            return (
                "The memory is healthy and does not "
                "currently require maintenance."
            )

        if (
            action
            == MaintenanceAction.REVIEW
        ):
            if report.has_been_retrieved:
                return (
                    "The memory is aging but has demonstrated "
                    "retrieval usage. It should be reviewed "
                    "before stronger lifecycle action is proposed."
                )

            return (
                "The memory is aging and should be reviewed "
                "before stronger lifecycle action is proposed."
            )

        if (
            action
            == MaintenanceAction.MARK_STALE
        ):
            if report.has_been_retrieved:
                return (
                    "The memory is a stale candidate but has "
                    "demonstrated retrieval usage. Marking it "
                    "stale should require explicit confirmation."
                )

            return (
                "The memory is a stale candidate. Marking it "
                "stale should require explicit confirmation."
            )

        if (
            action
            == MaintenanceAction.ARCHIVE
        ):
            if report.has_been_retrieved:
                return (
                    "The memory is an archival candidate but "
                    "has recorded retrieval usage. Archiving "
                    "should require explicit confirmation."
                )

            return (
                "The memory is an archival candidate. "
                "Archiving should require explicit confirmation."
            )

        return (
            "No maintenance action was determined."
        )