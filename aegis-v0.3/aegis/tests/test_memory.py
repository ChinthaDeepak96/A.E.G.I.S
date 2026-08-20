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
