"""
Tests for A.E.G.I.S. memory conflict resolution.

The resolver is intentionally tested independently from
MemoryManager.

These tests verify that:

    - DUPLICATE -> REJECT_CANDIDATE
    - CONFLICT -> KEEP_EXISTING
    - RELATED -> KEEP_BOTH
    - UNRELATED -> KEEP_BOTH

The resolver must never modify the memories passed to it.
"""

from memory.conflicts import (
    ConflictResult,
    MemoryRelationship,
)
from memory.models import Memory
from memory.resolution import (
    MemoryConflictResolver,
    ResolutionAction,
)


def make_conflict(
    relationship: MemoryRelationship,
    score: float = 0.8,
) -> ConflictResult:
    """
    Create a deterministic ConflictResult for testing.
    """

    return ConflictResult(
        relationship=relationship,
        score=score,
        matched_terms=("aegis",),
        reason="Test relationship.",
    )


def test_duplicate_rejects_candidate():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "AEGIS uses Qwen 7B."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.DUPLICATE,
            score=1.0,
        ),
    )

    assert (
        result.action
        == ResolutionAction.REJECT_CANDIDATE
    )

    assert (
        result.existing_memory_id
        == existing.id
    )

    assert (
        result.relationship
        == MemoryRelationship.DUPLICATE
    )

    assert (
        result.requires_confirmation
        is False
    )


def test_conflict_keeps_existing_memory():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "AEGIS now uses Gemma 4."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.CONFLICT,
            score=0.75,
        ),
    )

    assert (
        result.action
        == ResolutionAction.KEEP_EXISTING
    )

    assert (
        result.existing_memory_id
        == existing.id
    )

    assert (
        result.relationship
        == MemoryRelationship.CONFLICT
    )

    assert (
        result.requires_confirmation
        is True
    )


def test_related_keeps_both():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen for local inference."
    )

    candidate = Memory(
        "AEGIS uses Qwen for coding tasks."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.RELATED,
            score=0.45,
        ),
    )

    assert (
        result.action
        == ResolutionAction.KEEP_BOTH
    )

    assert (
        result.existing_memory_id
        == existing.id
    )

    assert (
        result.relationship
        == MemoryRelationship.RELATED
    )

    assert (
        result.requires_confirmation
        is False
    )


def test_unrelated_keeps_both():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "My favorite programming language is Python."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.UNRELATED,
            score=0.0,
        ),
    )

    assert (
        result.action
        == ResolutionAction.KEEP_BOTH
    )

    assert (
        result.existing_memory_id
        == existing.id
    )

    assert (
        result.relationship
        == MemoryRelationship.UNRELATED
    )

    assert (
        result.requires_confirmation
        is False
    )


def test_conflict_resolution_does_not_modify_existing_memory():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    original_content = existing.content
    original_status = existing.status
    original_updated_at = existing.updated_at

    candidate = Memory(
        "AEGIS now uses Gemma 4."
    )

    resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.CONFLICT,
        ),
    )

    assert (
        existing.content
        == original_content
    )

    assert (
        existing.status
        == original_status
    )

    assert (
        existing.updated_at
        == original_updated_at
    )


def test_duplicate_resolution_does_not_modify_candidate():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "AEGIS uses Qwen 7B."
    )

    original_content = candidate.content
    original_status = candidate.status

    resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.DUPLICATE,
            score=1.0,
        ),
    )

    assert (
        candidate.content
        == original_content
    )

    assert (
        candidate.status
        == original_status
    )


def test_resolution_can_handle_missing_existing_memory():
    resolver = MemoryConflictResolver()

    candidate = Memory(
        "AEGIS uses Qwen 7B."
    )

    result = resolver.resolve(
        None,
        candidate,
        make_conflict(
            MemoryRelationship.UNRELATED,
            score=0.0,
        ),
    )

    assert (
        result.action
        == ResolutionAction.KEEP_BOTH
    )

    assert (
        result.existing_memory_id
        is None
    )

    assert (
        result.candidate_content
        == candidate.content
    )


def test_conflict_requires_confirmation():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "I prefer VS Code."
    )

    candidate = Memory(
        "I now prefer PyCharm."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.CONFLICT,
            score=0.9,
        ),
    )

    assert (
        result.action
        == ResolutionAction.KEEP_EXISTING
    )

    assert (
        result.requires_confirmation
        is True
    )


def test_duplicate_does_not_require_confirmation():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "AEGIS uses Qwen 7B."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.DUPLICATE,
            score=1.0,
        ),
    )

    assert (
        result.action
        == ResolutionAction.REJECT_CANDIDATE
    )

    assert (
        result.requires_confirmation
        is False
    )

def test_conflict_can_propose_supersession_when_explicitly_allowed():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "AEGIS now uses Gemma 4."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.CONFLICT,
            score=0.9,
        ),
        allow_supersession=True,
    )

    assert (
        result.action
        == ResolutionAction.SUPERSEDE_EXISTING
    )

    assert (
        result.existing_memory_id
        == existing.id
    )

    assert (
        result.requires_confirmation
        is True
    )


def test_conflict_does_not_propose_supersession_by_default():
    resolver = MemoryConflictResolver()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "AEGIS now uses Gemma 4."
    )

    result = resolver.resolve(
        existing,
        candidate,
        make_conflict(
            MemoryRelationship.CONFLICT,
            score=0.9,
        ),
    )

    assert (
        result.action
        == ResolutionAction.KEEP_EXISTING
    )