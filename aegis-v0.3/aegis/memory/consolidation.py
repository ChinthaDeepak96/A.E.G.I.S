"""
A.E.G.I.S. Memory Consolidation.

Memory consolidation coordinates existing memory detection and
resolution components.

It does NOT directly modify memories.

Responsibilities:

    1. Compare a candidate against active memories.
    2. Identify the strongest meaningful relationship.
    3. Produce a consolidation proposal.
    4. Leave actual mutation to MemoryManager.

This keeps detection, decision-making, and mutation separate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .conflicts import (
    ConflictResult,
    MemoryConflictDetector,
    MemoryRelationship,
)
from .models import Memory, MemoryStatus
from .resolution import (
    MemoryConflictResolver,
    ResolutionAction,
    ResolutionDecision,
)


@dataclass(frozen=True)
class ConsolidationProposal:
    """
    Immutable proposal describing how a candidate should be
    consolidated with existing memory.

    No memory is modified by this object.
    """

    decision: ResolutionDecision

    conflict: ConflictResult | None

    existing_memory: Memory | None

    candidate: Memory

    candidates_checked: int

    reason: str


class MemoryConsolidator:
    """
    Coordinates memory comparison and resolution.

    The consolidator is deliberately conservative.

    Only ACTIVE memories participate in normal consolidation.

    Actual memory mutations remain the responsibility of
    MemoryManager.
    """

    def __init__(
        self,
        detector: MemoryConflictDetector | None = None,
        resolver: MemoryConflictResolver | None = None,
    ):
        self.detector = (
            detector
            if detector is not None
            else MemoryConflictDetector()
        )

        self.resolver = (
            resolver
            if resolver is not None
            else MemoryConflictResolver()
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def consolidate(
        self,
        candidate: Memory,
        existing_memories: list[Memory],
        *,
        allow_supersession: bool = False,
    ) -> ConsolidationProposal:
        """
        Compare a candidate against existing memories and return
        the strongest meaningful consolidation proposal.

        No memory is modified.

        Only ACTIVE memories participate in normal consolidation.
        """

        active_memories = [
            memory
            for memory in existing_memories
            if memory.status
            == MemoryStatus.ACTIVE
        ]

        if not active_memories:
            return self._no_existing_memory(
                candidate
            )

        best_existing: Memory | None = None
        best_conflict: ConflictResult | None = None

        for existing in active_memories:
            conflict = self.detector.compare(
                existing,
                candidate,
            )

            if self._is_better_match(
                conflict,
                best_conflict,
            ):
                best_existing = existing
                best_conflict = conflict

        # -----------------------------------------------------
        # No meaningful relationship
        #
        # UNRELATED means there is no existing memory that
        # should participate in consolidation.
        # -----------------------------------------------------

        if (
            best_conflict is None
            or best_conflict.relationship
            == MemoryRelationship.UNRELATED
        ):
            return self._no_existing_memory(
                candidate,
                candidates_checked=len(
                    active_memories
                ),
            )

        assert best_existing is not None

        decision = self.resolver.resolve(
            best_existing,
            candidate,
            best_conflict,
            allow_supersession=allow_supersession,
        )

        return ConsolidationProposal(
            decision=decision,
            conflict=best_conflict,
            existing_memory=best_existing,
            candidate=candidate,
            candidates_checked=len(
                active_memories
            ),
            reason=(
                "The candidate was compared against "
                f"{len(active_memories)} active memories. "
                "The strongest meaningful relationship was "
                "selected for resolution."
            ),
        )

    def find_relationships(
        self,
        candidate: Memory,
        existing_memories: list[Memory],
    ) -> list[
        tuple[Memory, ConflictResult]
    ]:
        """
        Return all meaningful relationships between a candidate
        and active existing memories.

        UNRELATED memories are excluded.

        Results are sorted strongest first.

        No memory is modified.
        """

        relationships: list[
            tuple[Memory, ConflictResult]
        ] = []

        for existing in existing_memories:
            if (
                existing.status
                != MemoryStatus.ACTIVE
            ):
                continue

            result = self.detector.compare(
                existing,
                candidate,
            )

            if (
                result.relationship
                != MemoryRelationship.UNRELATED
            ):
                relationships.append(
                    (
                        existing,
                        result,
                    )
                )

        relationships.sort(
            key=lambda item: (
                self._relationship_priority(
                    item[1].relationship
                ),
                item[1].score,
            ),
            reverse=True,
        )

        return relationships

    # =========================================================
    # HELPERS
    # =========================================================

    def _no_existing_memory(
        self,
        candidate: Memory,
        *,
        candidates_checked: int = 0,
    ) -> ConsolidationProposal:
        """
        Create a proposal for a candidate with no meaningful
        active relationship.

        The candidate may be stored as a new memory.
        """

        decision = ResolutionDecision(
            action=ResolutionAction.KEEP_BOTH,
            existing_memory_id=None,
            candidate_content=candidate.content,
            relationship=MemoryRelationship.UNRELATED,
            confidence=1.0,
            reason=(
                "No active existing memory was found that "
                "requires consolidation. The candidate may "
                "be stored as a new memory."
            ),
            requires_confirmation=False,
        )

        return ConsolidationProposal(
            decision=decision,
            conflict=None,
            existing_memory=None,
            candidate=candidate,
            candidates_checked=candidates_checked,
            reason=(
                "No meaningful active relationship was "
                "found for consolidation."
            ),
        )

    @staticmethod
    def _is_better_match(
        candidate_result: ConflictResult,
        current_result: ConflictResult | None,
    ) -> bool:
        """
        Determine whether a conflict result is stronger than
        the currently selected result.

        Priority:

            DUPLICATE
            CONFLICT
            RELATED
            UNRELATED

        Score breaks ties within the same relationship.
        """

        if current_result is None:
            return True

        candidate_priority = (
            MemoryConsolidator._relationship_priority(
                candidate_result.relationship
            )
        )

        current_priority = (
            MemoryConsolidator._relationship_priority(
                current_result.relationship
            )
        )

        if (
            candidate_priority
            != current_priority
        ):
            return (
                candidate_priority
                > current_priority
            )

        return (
            candidate_result.score
            > current_result.score
        )

    @staticmethod
    def _relationship_priority(
        relationship: MemoryRelationship,
    ) -> int:
        """
        Assign deterministic consolidation priority.
        """

        priorities = {
            MemoryRelationship.DUPLICATE: 4,
            MemoryRelationship.CONFLICT: 3,
            MemoryRelationship.RELATED: 2,
            MemoryRelationship.UNRELATED: 1,
        }

        return priorities[
            relationship
        ]