from memory import (
    MemoryManager,
    MemoryStatus,
    MemoryType,
)


def test_memory_database_is_created(tmp_path):
    db = tmp_path / "memory.db"
    manager = MemoryManager(str(db))
    assert db.exists()
    assert manager.recent() == []


def test_explicit_memory_is_persisted(tmp_path):
    manager = MemoryManager(str(tmp_path / "memory.db"))
    memory = manager.remember(
        "A.E.G.I.S. is my personal AI assistant.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.4,
        explicit=True,
    )
    assert memory is not None
    assert manager.recent()[0].content == "A.E.G.I.S. is my personal AI assistant."


def test_high_importance_memory_is_automatically_stored(tmp_path):
    manager = MemoryManager(str(tmp_path / "memory.db"))
    memory = manager.remember(
        "A.E.G.I.S. v0.4 uses persistent memory.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
    )
    assert memory is not None


def test_low_importance_memory_is_not_automatically_stored(tmp_path):
    manager = MemoryManager(str(tmp_path / "memory.db"))
    memory = manager.remember("Temporary thought.", importance=0.2)
    assert memory is None
    assert manager.recent() == []


def test_recall_finds_relevant_memory(tmp_path):
    manager = MemoryManager(str(tmp_path / "memory.db"))
    manager.remember(
        "A.E.G.I.S. currently uses Gemma 4 through Ollama.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
    )
    manager.remember(
        "The project has a Guardian risk system.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
    )
    results = manager.recall("Gemma Ollama")
    assert results
    assert "Gemma" in results[0].content


def test_forget_removes_memory(tmp_path):
    manager = MemoryManager(str(tmp_path / "memory.db"))
    memory = manager.remember("This should be forgotten.", importance=0.9)
    assert memory is not None
    assert manager.forget(memory.id) is True
    assert manager.store.get(memory.id) is None


def test_clear_removes_all_memory(tmp_path):
    manager = MemoryManager(str(tmp_path / "memory.db"))
    manager.remember("Memory one.", importance=0.9)
    manager.remember("Memory two.", importance=0.9)
    assert manager.clear() == 2
    assert manager.recent() == []
def test_get_returns_memory_by_id(tmp_path):
    manager = MemoryManager(
        str(tmp_path / "memory.db")
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
    )

    assert memory is not None

    retrieved = manager.get(memory.id)

    assert retrieved is not None
    assert retrieved.id == memory.id
    assert retrieved.content == "A.E.G.I.S. uses Gemma 4."


def test_get_returns_none_for_unknown_id(tmp_path):
    manager = MemoryManager(
        str(tmp_path / "memory.db")
    )

    assert manager.get(
        "does-not-exist"
    ) is None


def test_format_memory_contains_metadata(tmp_path):
    manager = MemoryManager(
        str(tmp_path / "memory.db")
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
        confidence=0.95,
        sensitivity=0.1,
        source="test",
        tags=["aegis", "llm"],
    )

    assert memory is not None

    formatted = manager.format_memory(memory)

    assert memory.id in formatted
    assert "semantic" in formatted
    assert "Gemma 4" in formatted
    assert "0.90" in formatted
    assert "0.95" in formatted
    assert "test" in formatted
    assert "aegis" in formatted

def test_find_similar_memory(tmp_path):
    manager = MemoryManager(
        str(tmp_path / "memory.db")
    )

    original = manager.remember(
        "A.E.G.I.S. uses Gemma 4 through Ollama.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
    )

    assert original is not None

    matches = manager.find_similar(
        "A.E.G.I.S. uses Gemma 4 with Ollama."
    )

    assert matches
    assert matches[0][0].id == original.id
    assert matches[0][1] >= 0.60


def test_store_candidate_rejects_low_importance(tmp_path):
    manager = MemoryManager(
        str(tmp_path / "memory.db")
    )

    status, memory = manager.store_candidate(
        "Temporary insignificant thought.",
        MemoryType.EPISODIC,
        importance=0.2,
        confidence=0.9,
        sensitivity=0.0,
    )

    assert status == "rejected"
    assert memory is None


def test_store_candidate_detects_duplicate(tmp_path):
    manager = MemoryManager(
        str(tmp_path / "memory.db")
    )

    original = manager.remember(
        "A.E.G.I.S. uses Gemma 4 through Ollama.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
    )

    assert original is not None

    status, memory = manager.store_candidate(
        "A.E.G.I.S. uses Gemma 4 through Ollama.",
        MemoryType.SEMANTIC,
        importance=0.9,
        confidence=0.95,
        sensitivity=0.0,
    )

    assert status == "duplicate"
    assert memory is not None
    assert memory.id == original.id

    assert len(manager.recent()) == 1


def test_store_candidate_stores_new_memory(tmp_path):
    manager = MemoryManager(
        str(tmp_path / "memory.db")
    )

    status, memory = manager.store_candidate(
        "A.E.G.I.S. is a long-term personal AI platform.",
        MemoryType.SEMANTIC,
        importance=0.9,
        confidence=0.95,
        sensitivity=0.0,
    )

    assert status == "stored"
    assert memory is not None
    assert len(manager.recent()) == 1

def test_new_memory_starts_active():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS starts with an active memory.",
        importance=0.9,
    )

    assert memory is not None
    assert memory.status == MemoryStatus.ACTIVE


def test_memory_can_be_marked_stale():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Temporary AEGIS information.",
        importance=0.9,
    )

    assert memory is not None

    memory.mark_stale()
    manager.store.save(memory)

    stored = manager.get(memory.id)

    assert stored is not None
    assert stored.status == MemoryStatus.STALE
    assert stored.stale_at is not None


def test_memory_can_be_reactivated():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Information that may become relevant again.",
        importance=0.9,
    )

    assert memory is not None

    memory.mark_stale()
    manager.store.save(memory)

    memory.activate()
    manager.store.save(memory)

    stored = manager.get(memory.id)

    assert stored is not None
    assert stored.status == MemoryStatus.ACTIVE
    assert stored.stale_at is None


def test_memory_can_be_archived():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Old AEGIS information.",
        importance=0.9,
    )

    assert memory is not None

    memory.archive()
    manager.store.save(memory)

    stored = manager.get(memory.id)

    assert stored is not None
    assert stored.status == MemoryStatus.ARCHIVED
    assert stored.archived_at is not None


def test_memory_can_be_superseded():
    manager = MemoryManager(":memory:")

    old_memory = manager.remember(
        "AEGIS uses an older model.",
        importance=0.9,
    )

    new_memory = manager.remember(
        "AEGIS now uses a newer model.",
        importance=0.95,
    )

    assert old_memory is not None
    assert new_memory is not None

    old_memory.supersede(
        new_memory.id
    )

    manager.store.save(
        old_memory
    )

    stored = manager.get(
        old_memory.id
    )

    assert stored is not None
    assert stored.status == MemoryStatus.SUPERSEDED
    assert stored.superseded_by == new_memory.id


def test_recent_can_filter_by_status():
    manager = MemoryManager(":memory:")

    active = manager.remember(
        "Active memory.",
        importance=0.9,
    )

    stale = manager.remember(
        "Stale memory.",
        importance=0.9,
    )

    assert active is not None
    assert stale is not None

    stale.mark_stale()
    manager.store.save(stale)

    active_memories = manager.recent(
        status=MemoryStatus.ACTIVE
    )

    stale_memories = manager.recent(
        status=MemoryStatus.STALE
    )

    assert any(
        m.id == active.id
        for m in active_memories
    )

    assert not any(
        m.id == stale.id
        for m in active_memories
    )

    assert any(
        m.id == stale.id
        for m in stale_memories
    )


def test_stale_memory_is_not_returned_by_default_search():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses lifecycle aware memory.",
        importance=0.9,
    )

    assert memory is not None

    memory.mark_stale()
    manager.store.save(memory)

    results = manager.recall(
        "AEGIS lifecycle"
    )

    assert not any(
        m.id == memory.id
        for m in results
    )


def test_archived_memory_is_not_returned_by_default_search():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Archived AEGIS information.",
        importance=0.9,
    )

    assert memory is not None

    memory.archive()
    manager.store.save(memory)

    results = manager.recall(
        "Archived AEGIS information"
    )

    assert not any(
        m.id == memory.id
        for m in results
    )

def test_update_memory_changes_existing_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert memory is not None

    updated = manager.update_memory(
        memory.id,
        "AEGIS uses Gemma 4.",
    )

    assert updated is not None
    assert updated.id == memory.id
    assert updated.content == "AEGIS uses Gemma 4."
    assert updated.status == MemoryStatus.ACTIVE


def test_update_memory_rejects_archived_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Old AEGIS configuration.",
        importance=0.9,
    )

    assert memory is not None

    memory.archive()
    manager.store.save(memory)

    result = manager.update_memory(
        memory.id,
        "New configuration.",
    )

    assert result is None


def test_supersede_memory_creates_replacement():
    manager = MemoryManager(":memory:")

    old_memory = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert old_memory is not None

    old_id = old_memory.id

    old, new = manager.supersede_memory(
        old_id,
        "AEGIS uses Gemma 4.",
        importance=0.95,
    )

    assert old is not None
    assert new is not None

    assert old.id == old_id
    assert old.status == MemoryStatus.SUPERSEDED
    assert old.superseded_by == new.id

    assert new.status == MemoryStatus.ACTIVE
    assert (
        new.metadata["supersedes_memory_id"]
        == old.id
    )


def test_superseded_memory_is_not_returned_by_normal_recall():
    manager = MemoryManager(":memory:")

    old = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert old is not None

    old_id = old.id

    manager.supersede_memory(
        old_id,
        "AEGIS uses Gemma 4.",
        importance=0.95,
    )

    results = manager.recall(
        "AEGIS uses model"
    )

    result_ids = {
        memory.id
        for memory in results
    }

    assert old_id not in result_ids


def test_restore_memory_reactivates_stale_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS temporary information.",
        importance=0.9,
    )

    assert memory is not None

    memory.mark_stale()
    manager.store.save(memory)

    restored = manager.restore_memory(
        memory.id
    )

    assert restored is not None
    assert restored.status == MemoryStatus.ACTIVE


def test_restore_memory_rejects_archived_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Archived AEGIS information.",
        importance=0.9,
    )

    assert memory is not None

    memory.archive()
    manager.store.save(memory)

    restored = manager.restore_memory(
        memory.id
    )

    assert restored is None


def test_archive_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Information no longer needed.",
        importance=0.9,
    )

    assert memory is not None

    archived = manager.archive_memory(
        memory.id
    )

    assert archived is not None
    assert archived.status == MemoryStatus.ARCHIVED
    assert archived.archived_at is not None


def test_stale_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "Information becoming outdated.",
        importance=0.9,
    )

    assert memory is not None

    stale = manager.stale_memory(
        memory.id
    )

    assert stale is not None
    assert stale.status == MemoryStatus.STALE
    assert stale.stale_at is not None