"""
A.E.G.I.S. Memory Maintenance Execution.

Executes maintenance proposals only after they pass MemoryPolicy
governance.

Responsibilities:

    MaintenanceProposal
            ↓
    MemoryPolicy.review_maintenance()
            ↓
       ALLOW / CONFIRM / DENY
            ↓
    MemoryManager lifecycle mutation

This module never mutates Memory directly.

All lifecycle mutations are delegated to MemoryManager.
"""

from __future__ import annotations

from dataclasses import dataclass

from .maintenance import (
    MaintenanceAction,
    MaintenanceProposal,
)
from .manager import MemoryManager
from .policy import (
    MemoryPolicy,
    MemoryPolicyDecision,
)
from .models import Memory


@dataclass(frozen=True)
class MaintenanceExecutionResult:
    """
    Result of attempting to execute a maintenance proposal.

    Fields:

        memory_id:
            ID of the memory involved.

        action:
            Proposed maintenance action.

        decision:
            Governance decision produced by MemoryPolicy.

        executed:
            True when a lifecycle mutation was successfully
            performed.

        confirmed:
            Whether explicit confirmation was supplied.

        memory:
            Resulting Memory when a mutation occurred.

        reason:
            Human-readable explanation.
    """

    memory_id: str

    action: MaintenanceAction

    decision: MemoryPolicyDecision

    executed: bool

    confirmed: bool

    memory: Memory | None

    reason: str


class MemoryMaintenanceExecutor:
    """
    Execute governed maintenance proposals.

    The executor is intentionally separate from:

        MemoryHealthAnalyzer
            → evaluates health

        MemoryMaintenancePlanner
            → creates proposals

        MemoryPolicy
            → decides whether proposals are allowed

        MemoryManager
            → performs lifecycle mutations

    This class coordinates those components.
    """

    def __init__(
        self,
        manager: MemoryManager,
        policy: MemoryPolicy | None = None,
    ):
        if not isinstance(
            manager,
            MemoryManager,
        ):
            raise TypeError(
                "manager must be a MemoryManager instance"
            )

        self.manager = manager

        self.policy = (
            policy
            if policy is not None
            else manager.policy
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def execute(
        self,
        proposal: MaintenanceProposal,
        *,
        confirmed: bool = False,
    ) -> MaintenanceExecutionResult:
        """
        Attempt to execute a maintenance proposal.

        Rules:

            ALLOW
                Execute automatically when the action is
                executable.

            CONFIRM
                Execute only when confirmed=True.

            DENY
                Never execute.

        Non-mutating actions such as NO_ACTION and REVIEW
        produce successful execution results without changing
        the memory.
        """

        if not isinstance(
            proposal,
            MaintenanceProposal,
        ):
            raise TypeError(
                "proposal must be a MaintenanceProposal"
            )

        decision = (
            self.policy.review_maintenance(
                proposal
            )
        )

        # -----------------------------------------------------
        # DENY
        # -----------------------------------------------------

        if (
            decision
            == MemoryPolicyDecision.DENY
        ):
            return MaintenanceExecutionResult(
                memory_id=proposal.memory_id,
                action=proposal.action,
                decision=decision,
                executed=False,
                confirmed=confirmed,
                memory=None,
                reason=(
                    "Maintenance was denied by memory policy."
                ),
            )

        # -----------------------------------------------------
        # CONFIRMATION REQUIRED
        # -----------------------------------------------------

        if (
            decision
            == MemoryPolicyDecision.CONFIRM
            and not confirmed
        ):
            return MaintenanceExecutionResult(
                memory_id=proposal.memory_id,
                action=proposal.action,
                decision=decision,
                executed=False,
                confirmed=False,
                memory=None,
                reason=(
                    "Maintenance requires explicit confirmation."
                ),
            )

        # -----------------------------------------------------
        # NO ACTION
        # -----------------------------------------------------

        if (
            proposal.action
            == MaintenanceAction.NO_ACTION
        ):
            return MaintenanceExecutionResult(
                memory_id=proposal.memory_id,
                action=proposal.action,
                decision=decision,
                executed=False,
                confirmed=confirmed,
                memory=None,
                reason=(
                    "No maintenance action was required."
                ),
            )

        # -----------------------------------------------------
        # REVIEW
        # -----------------------------------------------------

        if (
            proposal.action
            == MaintenanceAction.REVIEW
        ):
            return MaintenanceExecutionResult(
                memory_id=proposal.memory_id,
                action=proposal.action,
                decision=decision,
                executed=False,
                confirmed=confirmed,
                memory=None,
                reason=(
                    "The memory was flagged for review. "
                    "No lifecycle mutation was performed."
                ),
            )

        # -----------------------------------------------------
        # MARK STALE
        # -----------------------------------------------------

        if (
            proposal.action
            == MaintenanceAction.MARK_STALE
        ):
            return self._execute_stale(
                proposal,
                decision,
                confirmed,
            )

        # -----------------------------------------------------
        # ARCHIVE
        # -----------------------------------------------------

        if (
            proposal.action
            == MaintenanceAction.ARCHIVE
        ):
            return self._execute_archive(
                proposal,
                decision,
                confirmed,
            )

        # -----------------------------------------------------
        # Fail closed.
        # -----------------------------------------------------

        return MaintenanceExecutionResult(
            memory_id=proposal.memory_id,
            action=proposal.action,
            decision=MemoryPolicyDecision.DENY,
            executed=False,
            confirmed=confirmed,
            memory=None,
            reason=(
                "Unsupported maintenance action. "
                "Execution was denied."
            ),
        )

    # =========================================================
    # LIFECYCLE EXECUTION
    # =========================================================

    def _execute_stale(
        self,
        proposal: MaintenanceProposal,
        decision: MemoryPolicyDecision,
        confirmed: bool,
    ) -> MaintenanceExecutionResult:
        """
        Execute MARK_STALE through MemoryManager.
        """

        memory = self.manager.stale_memory(
            proposal.memory_id
        )

        if memory is None:
            return MaintenanceExecutionResult(
                memory_id=proposal.memory_id,
                action=proposal.action,
                decision=decision,
                executed=False,
                confirmed=confirmed,
                memory=None,
                reason=(
                    "The memory could not be marked stale. "
                    "It may not exist or may no longer be active."
                ),
            )

        return MaintenanceExecutionResult(
            memory_id=proposal.memory_id,
            action=proposal.action,
            decision=decision,
            executed=True,
            confirmed=confirmed,
            memory=memory,
            reason=(
                "The memory was successfully marked stale."
            ),
        )

    def _execute_archive(
        self,
        proposal: MaintenanceProposal,
        decision: MemoryPolicyDecision,
        confirmed: bool,
    ) -> MaintenanceExecutionResult:
        """
        Execute ARCHIVE through MemoryManager.
        """

        memory = self.manager.archive_memory(
            proposal.memory_id
        )

        if memory is None:
            return MaintenanceExecutionResult(
                memory_id=proposal.memory_id,
                action=proposal.action,
                decision=decision,
                executed=False,
                confirmed=confirmed,
                memory=None,
                reason=(
                    "The memory could not be archived. "
                    "It may not exist or may already be "
                    "archived or superseded."
                ),
            )

        return MaintenanceExecutionResult(
            memory_id=proposal.memory_id,
            action=proposal.action,
            decision=decision,
            executed=True,
            confirmed=confirmed,
            memory=memory,
            reason=(
                "The memory was successfully archived."
            ),
        )