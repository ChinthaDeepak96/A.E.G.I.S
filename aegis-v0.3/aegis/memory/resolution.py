"""
A.E.G.I.S. Memory Conflict Resolution.

This module converts a MemoryConflictDetector result into a
deterministic resolution decision.

Important:
    This module does NOT modify memories or the database.

It only answers:

    "Given this relationship, what action should A.E.G.I.S.
     take?"

Actual memory mutation remains the responsibility of
MemoryManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .conflicts import (
    ConflictResult,
    MemoryRelationship,
)
from .models import Memory


class ResolutionAction(str, Enum):
    """
    Action proposed for a memory relationship.
    """

    KEEP_EXISTING = "keep_existing"

    KEEP_BOTH = "keep_both"

    REJECT_CANDIDATE = "reject_candidate"

    SUPERSEDE_EXISTING = "supersede_existing"


@dataclass(frozen=True)
class ResolutionDecision:
    """
    Immutable resolution decision.

    The decision contains enough information for MemoryManager
    or a future Guardian layer to understand what should happen.

    No memory is modified here.
    """

    action: ResolutionAction

    existing_memory_id: str | None

    candidate_content: str

    relationship: MemoryRelationship

    confidence: float

    reason: str

    requires_confirmation: bool = False


class MemoryConflictResolver:
    """
    Deterministic first-generation conflict resolver.

    Default behavior is deliberately conservative.

    DUPLICATE:
        Reject candidate.

    CONFLICT:
        Keep existing memory by default.

        If allow_supersession=True, explicitly propose
        SUPERSEDE_EXISTING instead.

    RELATED:
        Keep both.

    UNRELATED:
        Keep both.

    This resolver never mutates memories.

    Future versions can use Guardian or an LLM to make more
    sophisticated decisions.
    """

    def resolve(
        self,
        existing: Memory | None,
        candidate: Memory,
        conflict: ConflictResult,
        *,
        allow_supersession: bool = False,
    ) -> ResolutionDecision:
        """
        Produce a resolution decision.

        This method never changes either memory.

        Args:
            existing:
                Existing memory involved in the relationship.
                May be None.

            candidate:
                New candidate memory being evaluated.

            conflict:
                Result produced by MemoryConflictDetector.

            allow_supersession:
                Explicitly allows the resolver to propose
                SUPERSEDE_EXISTING for a genuine conflict.

                This does NOT perform supersession.

        Returns:
            ResolutionDecision
        """

        relationship = conflict.relationship

        # =====================================================
        # DUPLICATE
        # =====================================================

        if relationship == MemoryRelationship.DUPLICATE:
            return ResolutionDecision(
                action=ResolutionAction.REJECT_CANDIDATE,
                existing_memory_id=(
                    existing.id
                    if existing is not None
                    else None
                ),
                candidate_content=candidate.content,
                relationship=relationship,
                confidence=conflict.score,
                reason=(
                    "The candidate duplicates an existing "
                    "memory, so the candidate should not be "
                    "stored again."
                ),
                requires_confirmation=False,
            )

        # =====================================================
        # CONFLICT
        # =====================================================

        if relationship == MemoryRelationship.CONFLICT:

            # -------------------------------------------------
            # Explicit supersession proposal.
            #
            # This only creates a decision. It does not modify
            # the existing memory or candidate.
            # -------------------------------------------------

            if (
                allow_supersession
                and existing is not None
            ):
                return ResolutionDecision(
                    action=ResolutionAction.SUPERSEDE_EXISTING,
                    existing_memory_id=existing.id,
                    candidate_content=candidate.content,
                    relationship=relationship,
                    confidence=conflict.score,
                    reason=(
                        "The candidate conflicts with the "
                        "existing memory and supersession has "
                        "been explicitly allowed. The existing "
                        "memory must still be replaced only "
                        "through an approved mutation operation."
                    ),
                    requires_confirmation=True,
                )

            # -------------------------------------------------
            # Conservative default.
            # -------------------------------------------------

            return ResolutionDecision(
                action=ResolutionAction.KEEP_EXISTING,
                existing_memory_id=(
                    existing.id
                    if existing is not None
                    else None
                ),
                candidate_content=candidate.content,
                relationship=relationship,
                confidence=conflict.score,
                reason=(
                    "The candidate conflicts with an existing "
                    "memory. The existing memory is preserved "
                    "until an explicit resolution decision is "
                    "made."
                ),
                requires_confirmation=True,
            )

        # =====================================================
        # RELATED
        # =====================================================

        if relationship == MemoryRelationship.RELATED:
            return ResolutionDecision(
                action=ResolutionAction.KEEP_BOTH,
                existing_memory_id=(
                    existing.id
                    if existing is not None
                    else None
                ),
                candidate_content=candidate.content,
                relationship=relationship,
                confidence=conflict.score,
                reason=(
                    "The memories are related but do not "
                    "contain a strong contradiction, so both "
                    "may be retained."
                ),
                requires_confirmation=False,
            )

        # =====================================================
        # UNRELATED
        # =====================================================

        return ResolutionDecision(
            action=ResolutionAction.KEEP_BOTH,
            existing_memory_id=(
                existing.id
                if existing is not None
                else None
            ),
            candidate_content=candidate.content,
            relationship=relationship,
            confidence=conflict.score,
            reason=(
                "The memories do not describe the same "
                "information, so both may be retained."
            ),
            requires_confirmation=False,
        )