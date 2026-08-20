"""
Integration tests for MemoryManager + ConflictDetector +
MemoryConflictResolver.
"""

from unittest import result

from memory import (
    MemoryManager,
    MemoryRelationship,
    MemoryStatus,
    MemoryType,
    ResolutionAction,
)


def test_unrelated_candidate_is_stored():
    manager = MemoryManager(":memory:")

    decision, memory = manager.resolve_candidate(
        "My favorite programming language is Python.",
        memory_type=MemoryType.PREFERENCE,
        importance=0.9,
        confidence=1.0,
        sensitivity=0.0,
    )

    assert (
        decision.action
        == ResolutionAction.KEEP_BOTH
    )

    assert memory is not None

    assert (
        memory.content
        == "My favorite programming language is Python."
    )

    assert (
        manager.get(memory.id)
        is not None
    )


def test_duplicate_candidate_is_rejected():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert existing is not None

    decision, result = manager.resolve_candidate(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert (
        decision.action
        == ResolutionAction.REJECT_CANDIDATE
    )

    assert (
        decision.relationship
        == MemoryRelationship.DUPLICATE
    )

    assert (
        decision.existing_memory_id
        == existing.id
    )

    assert result is not None
    assert result.id == existing.id

    memories = manager.recent()

    assert len(memories) == 1


def test_conflicting_candidate_preserves_existing():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert existing is not None

    decision, result = manager.resolve_candidate(
        "AEGIS now uses Gemma 4.",
        importance=0.9,
    )

    assert (
        decision.action
        == ResolutionAction.KEEP_EXISTING
    )

    assert (
        decision.relationship
        == MemoryRelationship.CONFLICT
    )

    assert (
        decision.requires_confirmation
        is True
    )

    assert result is not None
    assert result.id == existing.id

    current = manager.get(
        existing.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.ACTIVE
    )

    assert (
        current.content
        == "AEGIS uses Qwen 7B."
    )


def test_related_candidate_is_kept():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "AEGIS uses Qwen for local inference.",
        importance=0.8,
    )

    assert existing is not None

    decision, memory = manager.resolve_candidate(
        "AEGIS uses Qwen for coding tasks.",
        importance=0.8,
    )

    assert (
        decision.action
        == ResolutionAction.KEEP_BOTH
    )

    assert (
        decision.relationship
        == MemoryRelationship.RELATED
    )

    assert memory is not None

    memories = manager.recent()

    assert len(memories) == 2


def test_conflict_does_not_modify_existing_memory():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert existing is not None

    original_id = existing.id
    original_content = existing.content

    decision, _ = manager.resolve_candidate(
        "AEGIS now uses Gemma 4.",
        importance=0.9,
    )

    assert (
        decision.action
        == ResolutionAction.KEEP_EXISTING
    )

    current = manager.get(
        original_id
    )

    assert current is not None

    assert (
        current.content
        == original_content
    )

    assert (
        current.status
        == MemoryStatus.ACTIVE
    )


def test_no_existing_memory_stores_candidate():
    manager = MemoryManager(":memory:")

    decision, memory = manager.resolve_candidate(
        "AEGIS is a personal AI system.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
        confidence=0.95,
        sensitivity=0.0,
    )

    assert (
        decision.relationship
        == MemoryRelationship.UNRELATED
    )

    assert (
        decision.action
        == ResolutionAction.KEEP_BOTH
    )

    assert memory is not None

    assert (
        len(manager.recent())
        == 1
    )


def test_sensitive_candidate_is_rejected():
    manager = MemoryManager(":memory:")

    decision, memory = manager.resolve_candidate(
        "Sensitive candidate memory.",
        importance=0.9,
        sensitivity=0.5,
    )

    assert (
        decision.action
        == ResolutionAction.REJECT_CANDIDATE
    )

    assert memory is None

    assert (
        len(manager.recent())
        == 0
    )


def test_policy_rejection_is_reported():
    manager = MemoryManager(":memory:")

    decision, memory = manager.resolve_candidate(
        "Low importance candidate.",
        importance=0.01,
        confidence=0.5,
        sensitivity=0.0,
    )

    assert memory is None

    assert (
        decision.action
        == ResolutionAction.REJECT_CANDIDATE
    )