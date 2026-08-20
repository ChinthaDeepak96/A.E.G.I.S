from memory import MemoryManager, MemoryType


def test_exact_phrase_ranks_first():
    manager = MemoryManager(":memory:")

    exact = manager.remember(
        "AEGIS uses Gemma 4 through Ollama.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
    )

    related = manager.remember(
        "The AEGIS project uses an LLM backend.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.95,
    )

    assert exact is not None
    assert related is not None

    results = manager.recall(
        "AEGIS uses Gemma 4"
    )

    assert results
    assert results[0].id == exact.id


def test_multiple_query_terms_are_ranked():
    manager = MemoryManager(":memory:")

    gemma = manager.remember(
        "AEGIS uses Gemma through Ollama.",
        importance=0.8,
    )

    unrelated = manager.remember(
        "The Guardian monitors system risk.",
        importance=0.95,
    )

    assert gemma is not None
    assert unrelated is not None

    results = manager.recall(
        "Gemma Ollama"
    )

    assert results
    assert results[0].id == gemma.id


def test_tags_contribute_to_retrieval():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "The project uses a local language model.",
        importance=0.8,
        tags=[
            "gemma",
            "ollama",
        ],
    )

    assert memory is not None

    results = manager.recall(
        "Gemma Ollama"
    )

    assert results
    assert results[0].id == memory.id


def test_inactive_memories_are_excluded():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    memory.mark_stale()

    manager.store.save(
        memory
    )

    results = manager.recall(
        "Gemma 4"
    )

    assert not any(
        item.id == memory.id
        for item in results
    )


def test_superseded_memory_is_excluded():
    manager = MemoryManager(":memory:")

    old = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    new = manager.remember(
        "AEGIS uses Gemma 4.",
        importance=0.95,
    )

    assert old is not None
    assert new is not None

    old.supersede(
        new.id
    )

    manager.store.save(
        old
    )

    results = manager.recall(
        "AEGIS model"
    )

    assert old.id not in {
        item.id
        for item in results
    }


def test_empty_query_returns_no_results():
    manager = MemoryManager(":memory:")

    manager.remember(
        "AEGIS uses Gemma.",
        importance=0.9,
    )

    assert manager.recall(
        ""
    ) == []


def test_query_is_case_insensitive():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    results = manager.recall(
        "gemma"
    )

    assert results
    assert results[0].id == memory.id


def test_limit_is_respected():
    manager = MemoryManager(":memory:")

    for index in range(10):
        manager.remember(
            f"AEGIS memory Gemma {index}.",
            importance=0.9,
        )

    results = manager.recall(
        "AEGIS Gemma",
        limit=3,
    )

    assert len(results) == 3