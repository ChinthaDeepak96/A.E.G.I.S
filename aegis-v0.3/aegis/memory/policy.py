"""
A.E.G.I.S. Memory Governance Policy.

This module governs two separate classes of memory decisions:

1. Persistence policy
   Determines whether a memory may become persistent.

2. Maintenance policy
   Determines whether a proposed maintenance action may be
   performed automatically, requires confirmation, or is denied.

The policy layer is intentionally separate from:

- MemoryHealthAnalyzer
    Evaluates memory health.

- MemoryMaintenancePlanner
    Proposes maintenance actions.

- MemoryManager
    Performs actual memory mutations.

The policy never mutates Memory objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .maintenance import (
    MaintenanceAction,
    MaintenanceProposal,
)
from .models import Memory


class MemoryPolicyDecision(str, Enum):
    """
    Governance decision for a memory operation.
    """

    ALLOW = "allow"

    CONFIRM = "confirm"

    DENY = "deny"


@dataclass
class MemoryPolicy:
    """
    Controls memory persistence and maintenance governance.

    Persistence policy:

        Automatic memories must satisfy:
            confidence >= minimum_confidence
            importance >= automatic_importance_threshold

        Sensitive memories may additionally be restricted.

        Explicit memories bypass automatic persistence
        requirements.

    Maintenance policy:

        NO_ACTION
            -> ALLOW

        REVIEW
            -> ALLOW

        MARK_STALE
            -> CONFIRM

        ARCHIVE
            -> CONFIRM

    High-importance memories can be protected from automatic
    destructive maintenance.

    The policy never modifies Memory or MaintenanceProposal.
    """

    # =========================================================
    # PERSISTENCE POLICY
    # =========================================================

    automatic_importance_threshold: float = 0.65

    allow_sensitive_memory: bool = True

    minimum_confidence: float = 0.50

    # =========================================================
    # MAINTENANCE POLICY
    # =========================================================

    # Memories at or above this importance level cannot be
    # automatically subjected to destructive maintenance.
    protected_importance_threshold: float = 0.90

    # Sensitive memories require confirmation before destructive
    # maintenance even when sensitive persistence is allowed.
    require_confirmation_for_sensitive: bool = True

    def __post_init__(self) -> None:
        """
        Validate policy configuration.
        """

        if not 0.0 <= self.automatic_importance_threshold <= 1.0:
            raise ValueError(
                "automatic_importance_threshold must be between "
                "0.0 and 1.0"
            )

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0.0 and 1.0"
            )

        if not 0.0 <= self.protected_importance_threshold <= 1.0:
            raise ValueError(
                "protected_importance_threshold must be between "
                "0.0 and 1.0"
            )

    # =========================================================
    # PERSISTENCE
    # =========================================================

    def should_store(
        self,
        memory: Memory,
        explicit: bool = False,
    ) -> bool:
        """
        Decide whether a memory should be persisted.

        Explicit memories bypass the automatic importance and
        confidence requirements.

        Non-explicit memories must satisfy the configured
        importance and confidence thresholds.

        This preserves the existing MemoryManager API.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "memory must be a Memory instance"
            )

        # -----------------------------------------------------
        # Explicit user-directed memory.
        # -----------------------------------------------------

        if explicit:
            return True

        # -----------------------------------------------------
        # Confidence gate.
        # -----------------------------------------------------

        if (
            memory.confidence
            < self.minimum_confidence
        ):
            return False

        # -----------------------------------------------------
        # Sensitivity gate.
        # -----------------------------------------------------

        if (
            memory.sensitivity > 0
            and not self.allow_sensitive_memory
        ):
            return False

        # -----------------------------------------------------
        # Importance gate.
        # -----------------------------------------------------

        if (
            memory.importance
            < self.automatic_importance_threshold
        ):
            return False

        return True

    # =========================================================
    # MAINTENANCE GOVERNANCE
    # =========================================================

    def review_maintenance(
        self,
        proposal: MaintenanceProposal,
    ) -> MemoryPolicyDecision:
        """
        Review a maintenance proposal.

        The proposal is advisory. This method decides whether
        the proposed action can be:

            ALLOW
                performed automatically.

            CONFIRM
                performed only after explicit confirmation.

            DENY
                rejected by policy.

        This method does not mutate the proposal or its memory.
        """

        if not isinstance(
            proposal,
            MaintenanceProposal,
        ):
            raise TypeError(
                "proposal must be a MaintenanceProposal"
            )

        action = proposal.action

        # -----------------------------------------------------
        # No-op actions are always safe.
        # -----------------------------------------------------

        if (
            action
            == MaintenanceAction.NO_ACTION
        ):
            return MemoryPolicyDecision.ALLOW

        # -----------------------------------------------------
        # Review is non-destructive.
        # -----------------------------------------------------

        if (
            action
            == MaintenanceAction.REVIEW
        ):
            return MemoryPolicyDecision.ALLOW

        # -----------------------------------------------------
        # Unknown / unsupported actions fail closed.
        # -----------------------------------------------------

        if action not in {
            MaintenanceAction.MARK_STALE,
            MaintenanceAction.ARCHIVE,
        }:
            return MemoryPolicyDecision.DENY

        # -----------------------------------------------------
        # Destructive actions require confirmation by default.
        # -----------------------------------------------------

        decision = (
            MemoryPolicyDecision.CONFIRM
        )

        # -----------------------------------------------------
        # A highly important memory must never be silently
        # subjected to destructive maintenance.
        #
        # The proposal itself does not currently carry the
        # original importance value, so this protection can only
        # be applied when that evidence is available through the
        # proposal in future versions.
        #
        # For now, destructive lifecycle actions remain
        # confirmation-gated.
        # -----------------------------------------------------

        if proposal.requires_confirmation:
            return decision

        return decision

    # =========================================================
    # CONVENIENCE HELPERS
    # =========================================================

    def allows_automatic_maintenance(
        self,
        proposal: MaintenanceProposal,
    ) -> bool:
        """
        Return True only when maintenance can execute without
        user confirmation.
        """

        return (
            self.review_maintenance(
                proposal
            )
            == MemoryPolicyDecision.ALLOW
        )

    def requires_maintenance_confirmation(
        self,
        proposal: MaintenanceProposal,
    ) -> bool:
        """
        Return True when maintenance requires explicit
        confirmation.
        """

        return (
            self.review_maintenance(
                proposal
            )
            == MemoryPolicyDecision.CONFIRM
        )

    def denies_maintenance(
        self,
        proposal: MaintenanceProposal,
    ) -> bool:
        """
        Return True when policy denies the proposal.
        """

        return (
            self.review_maintenance(
                proposal
            )
            == MemoryPolicyDecision.DENY
        )