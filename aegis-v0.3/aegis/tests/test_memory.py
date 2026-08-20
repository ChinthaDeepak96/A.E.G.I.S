from memory import MemoryManager, MemoryType


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