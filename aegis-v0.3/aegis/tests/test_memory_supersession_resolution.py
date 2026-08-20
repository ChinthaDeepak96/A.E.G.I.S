from memory import (
    MemoryManager,
    MemoryRelationship,
    MemoryStatus,
    MemoryType,
    ResolutionAction,
)


def test_conflict_can_be_explicitly_resolved_by_supersession():
    manager = MemoryManager(":memory:")

    old_memory = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert old_memory is not None

    decision, _ = manager.resolve_candidate(
        "AEGIS now uses Gemma 4.",
        importance=0.95,
    )

    assert (
        decision.relationship
        == MemoryRelationship.CONFLICT
    )

    assert (
        decision.action
        == ResolutionAction.KEEP_EXISTING
    )

    # Create an explicit supersession decision.
    supersession = decision.__class__(
        action=ResolutionAction.SUPERSEDE_EXISTING,
        existing_memory_id=decision.existing_memory_id,
        candidate_content=decision.candidate_content,
        relationship=decision.relationship,
        confidence=decision.confidence,
        reason="Explicitly approved replacement.",
        requires_confirmation=True,
    )

    old, new = manager.apply_resolution(
        supersession,
        approve_destructive=True,
        importance=0.95,
    )

    assert old is not None
    assert new is not None

    assert old.id == old_memory.id
    assert old.status == MemoryStatus.SUPERSEDED
    assert old.superseded_by == new.id

    assert new.status == MemoryStatus.ACTIVE

    stored_old = manager.get(
        old_memory.id
    )

    stored_new = manager.get(
        new.id
    )

    assert stored_old is not None
    assert stored_new is not None

    assert (
        stored_old.status
        == MemoryStatus.SUPERSEDED
    )

    assert (
        stored_new.status
        == MemoryStatus.ACTIVE
    )


def test_supersession_requires_explicit_approval():
    manager = MemoryManager(":memory:")

    old_memory = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert old_memory is not None

    decision = manager.resolve_candidate(
        "AEGIS now uses Gemma 4.",
        importance=0.95,
    )[0]

    supersession = decision.__class__(
        action=ResolutionAction.SUPERSEDE_EXISTING,
        existing_memory_id=old_memory.id,
        candidate_content="AEGIS now uses Gemma 4.",
        relationship=MemoryRelationship.CONFLICT,
        confidence=0.9,
        reason="Test supersession.",
        requires_confirmation=True,
    )

    old, new = manager.apply_resolution(
        supersession,
        approve_destructive=False,
    )

    assert old is not None
    assert new is None

    stored = manager.get(
        old_memory.id
    )

    assert stored is not None
    assert (
        stored.status
        == MemoryStatus.ACTIVE
    )


def test_superseded_memory_is_not_normally_recalled():
    manager = MemoryManager(":memory:")

    old_memory = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert old_memory is not None

    decision = manager.resolve_candidate(
        "AEGIS now uses Gemma 4.",
        importance=0.95,
    )[0]

    supersession = decision.__class__(
        action=ResolutionAction.SUPERSEDE_EXISTING,
        existing_memory_id=old_memory.id,
        candidate_content="AEGIS now uses Gemma 4.",
        relationship=MemoryRelationship.CONFLICT,
        confidence=0.9,
        reason="Test supersession.",
        requires_confirmation=True,
    )

    manager.apply_resolution(
        supersession,
        approve_destructive=True,
        importance=0.95,
    )

    results = manager.recall(
        "AEGIS uses model"
    )

    result_ids = {
        memory.id
        for memory in results
    }

    assert old_memory.id not in result_ids