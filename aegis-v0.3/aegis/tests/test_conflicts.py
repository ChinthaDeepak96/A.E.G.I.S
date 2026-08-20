from memory import (
    MemoryManager,
    MemoryRelationship,
    MemoryStatus,
    MemoryType,
)
from memory.models import Memory


def test_duplicate_is_detected():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert existing is not None

    status, result = manager.store_candidate(
        "AEGIS uses Qwen 7B.",
        memory_type=existing.memory_type,
        importance=0.9,
        confidence=1.0,
        sensitivity=0.0,
    )

    assert status == "duplicate"
    assert result is not None
    assert result.id == existing.id


def test_conflict_is_detected():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert existing is not None

    status, result = manager.store_candidate(
        "AEGIS now uses Gemma 4.",
        memory_type=existing.memory_type,
        importance=0.95,
        confidence=1.0,
        sensitivity=0.0,
    )

    assert status == "conflict"
    assert result is not None
    assert result.id == existing.id


def test_conflict_does_not_modify_existing_memory():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert existing is not None

    original_status = existing.status
    original_content = existing.content

    status, _ = manager.store_candidate(
        "AEGIS now uses Gemma 4.",
        memory_type=existing.memory_type,
        importance=0.95,
        confidence=1.0,
        sensitivity=0.0,
    )

    assert status == "conflict"

    stored = manager.get(
        existing.id
    )

    assert stored is not None
    assert stored.status == original_status
    assert stored.content == original_content


def test_unrelated_candidate_is_stored():
    manager = MemoryManager(":memory:")

    manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    status, memory = manager.store_candidate(
        "My favorite programming language is Python.",
        memory_type=MemoryType.PREFERENCE,
        importance=0.9,
        confidence=1.0,
        sensitivity=0.0,
    )

    assert status == "stored"
    assert memory is not None


def test_conflict_only_checks_active_memories():
    manager = MemoryManager(":memory:")

    old = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert old is not None

    old.archive()
    manager.store.save(old)

    status, memory = manager.store_candidate(
        "AEGIS now uses Gemma 4.",
        memory_type=old.memory_type,
        importance=0.9,
        confidence=1.0,
        sensitivity=0.0,
    )

    assert status == "stored"
    assert memory is not None


def test_detect_conflict_returns_conflicting_memory():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "I prefer VS Code for development.",
        importance=0.9,
    )

    assert existing is not None

    results = manager.detect_conflict(
        "I now prefer PyCharm for development."
    )

    assert len(results) == 1

    found_memory, result = results[0]

    assert found_memory.id == existing.id
    assert (
        result.relationship
        == MemoryRelationship.CONFLICT
    )


def test_detect_conflict_does_not_modify_memory():
    manager = MemoryManager(":memory:")

    existing = manager.remember(
        "I prefer VS Code for development.",
        importance=0.9,
    )

    assert existing is not None

    manager.detect_conflict(
        "I now prefer PyCharm for development."
    )

    stored = manager.get(
        existing.id
    )

    assert stored is not None
    assert stored.status == MemoryStatus.ACTIVE
    assert (
        stored.content
        == "I prefer VS Code for development."
    )