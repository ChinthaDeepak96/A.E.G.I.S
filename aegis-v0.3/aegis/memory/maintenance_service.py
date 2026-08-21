"""
A.E.G.I.S. Memory Maintenance Service.

High-level orchestration layer for memory maintenance.

The service coordinates:

    MemoryManager
        ↓
    MemoryHealthAnalyzer
        ↓
    MemoryMaintenancePlanner
        ↓
    MemoryPolicy
        ↓
    MemoryMaintenanceExecutor

The service itself does not directly mutate Memory objects.

It provides the application-facing API for:

    evaluate()
        Analyze a memory and produce a governed maintenance
        proposal.

    evaluate_many()
        Analyze multiple memories.

    execute()
        Evaluate and execute a maintenance action when policy
        and confirmation requirements permit it.

    execute_proposal()
        Execute an already-created proposal.

This layer exists so callers such as the future A.E.G.I.S.
orchestration layer do not need to understand the individual
maintenance components.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .health import (
    MemoryHealthAnalyzer,
    MemoryHealthReport,
)
from .maintenance import (
    MaintenanceAction,
    MaintenanceProposal,
    MemoryMaintenancePlanner,
)
from .maintenance_executor import (
    MaintenanceExecutionResult,
    MemoryMaintenanceExecutor,
)
from .manager import MemoryManager
from .models import Memory
from .policy import (
    MemoryPolicy,
    MemoryPolicyDecision,
)
from .usage import MemoryUsage


@dataclass(frozen=True)
class MaintenanceEvaluation:
    """
    Result of evaluating one memory for maintenance.

    This object contains the complete evidence chain:

        health report
            ↓
        maintenance proposal
            ↓
        policy decision

    No lifecycle mutation is performed by evaluation.
    """

    memory: Memory

    health: MemoryHealthReport

    proposal: MaintenanceProposal

    decision: MemoryPolicyDecision


@dataclass(frozen=True)
class MaintenanceBatchEvaluation:
    """
    Results of evaluating multiple memories.

    Input ordering is preserved.
    """

    evaluations: list[MaintenanceEvaluation]


class MemoryMaintenanceService:
    """
    High-level memory maintenance orchestration service.
    """

    def __init__(
        self,
        manager: MemoryManager,
        *,
        analyzer: MemoryHealthAnalyzer | None = None,
        planner: MemoryMaintenancePlanner | None = None,
        policy: MemoryPolicy | None = None,
        executor: MemoryMaintenanceExecutor | None = None,
    ):
        if not isinstance(
            manager,
            MemoryManager,
        ):
            raise TypeError(
                "manager must be a MemoryManager instance"
            )

        self.manager = manager

        self.analyzer = (
            analyzer
            if analyzer is not None
            else MemoryHealthAnalyzer()
        )

        self.planner = (
            planner
            if planner is not None
            else MemoryMaintenancePlanner(
                analyzer=self.analyzer
            )
        )

        self.policy = (
            policy
            if policy is not None
            else manager.policy
        )

        self.executor = (
            executor
            if executor is not None
            else MemoryMaintenanceExecutor(
                manager,
                policy=self.policy,
            )
        )

    # =========================================================
    # EVALUATION
    # =========================================================

    def evaluate(
        self,
        memory_id: str,
        *,
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> MaintenanceEvaluation | None:
        """
        Evaluate one stored memory.

        Returns None when the memory does not exist.

        Evaluation is read-only.

        The returned result contains:

            Memory
            MemoryHealthReport
            MaintenanceProposal
            MemoryPolicyDecision
        """

        memory_id = self._normalize_memory_id(
            memory_id
        )

        if not memory_id:
            return None

        memory = self.manager.get(
            memory_id
        )

        if memory is None:
            return None

        if usage is None:
            usage = (
                self.manager.retriever.get_usage(
                    memory.id
                )
            )

        report = self.analyzer.analyze(
            memory,
            usage=usage,
            now=now,
        )

        proposal = (
            self.planner.propose_from_report(
                report
            )
        )

        decision = (
            self.policy.review_maintenance(
                proposal
            )
        )

        return MaintenanceEvaluation(
            memory=memory,
            health=report,
            proposal=proposal,
            decision=decision,
        )

    def evaluate_many(
        self,
        memory_ids: list[str],
        *,
        usages: dict[str, MemoryUsage] | None = None,
        now: datetime | None = None,
    ) -> MaintenanceBatchEvaluation:
        """
        Evaluate multiple stored memories.

        Missing memory IDs are ignored.

        Input ordering is preserved.

        The method performs no lifecycle mutations.
        """

        if not isinstance(
            memory_ids,
            list,
        ):
            raise TypeError(
                "memory_ids must be a list"
            )

        if usages is None:
            usages = {}

        if not isinstance(
            usages,
            dict,
        ):
            raise TypeError(
                "usages must be a dictionary"
            )

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

        evaluations: list[
            MaintenanceEvaluation
        ] = []

        for memory_id in memory_ids:
            normalized_id = (
                self._normalize_memory_id(
                    memory_id
                )
            )

            if not normalized_id:
                continue

            evaluation = self.evaluate(
                normalized_id,
                usage=usages.get(
                    normalized_id
                ),
                now=now,
            )

            if evaluation is not None:
                evaluations.append(
                    evaluation
                )

        return MaintenanceBatchEvaluation(
            evaluations=evaluations
        )

    # =========================================================
    # EXECUTION
    # =========================================================

    def execute(
        self,
        memory_id: str,
        *,
        confirmed: bool = False,
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> MaintenanceExecutionResult | None:
        """
        Evaluate and attempt to execute maintenance for one memory.

        The service first evaluates the memory and then delegates
        execution to MemoryMaintenanceExecutor.

        Returns None when the memory does not exist.
        """

        evaluation = self.evaluate(
            memory_id,
            usage=usage,
            now=now,
        )

        if evaluation is None:
            return None

        return self.execute_proposal(
            evaluation.proposal,
            confirmed=confirmed,
        )

    def execute_proposal(
        self,
        proposal: MaintenanceProposal,
        *,
        confirmed: bool = False,
    ) -> MaintenanceExecutionResult:
        """
        Execute an already-created maintenance proposal.

        All mutation remains inside MemoryMaintenanceExecutor.
        """

        return self.executor.execute(
            proposal,
            confirmed=confirmed,
        )

    # =========================================================
    # CONVENIENCE HELPERS
    # =========================================================

    def review(
        self,
        memory_id: str,
        *,
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> MaintenanceEvaluation | None:
        """
        Alias for evaluate() intended for application code.
        """

        return self.evaluate(
            memory_id,
            usage=usage,
            now=now,
        )

    def pending_action(
        self,
        memory_id: str,
        *,
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> MaintenanceAction | None:
        """
        Return the proposed maintenance action for a memory.

        Returns None when the memory does not exist.
        """

        evaluation = self.evaluate(
            memory_id,
            usage=usage,
            now=now,
        )

        if evaluation is None:
            return None

        return evaluation.proposal.action

    def requires_confirmation(
        self,
        memory_id: str,
        *,
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> bool:
        """
        Return whether the current proposed maintenance action
        requires explicit confirmation.
        """

        evaluation = self.evaluate(
            memory_id,
            usage=usage,
            now=now,
        )

        if evaluation is None:
            return False

        return (
            evaluation.proposal
            .requires_confirmation
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _normalize_memory_id(
        memory_id: str,
    ) -> str:
        """
        Normalize a memory ID.

        Empty or whitespace-only IDs become an empty string.
        """

        if not isinstance(
            memory_id,
            str,
        ):
            raise TypeError(
                "memory_id must be a string"
            )

        return memory_id.strip()