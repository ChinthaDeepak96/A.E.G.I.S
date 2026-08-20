import numpy as np

from memory import MemoryManager
from memory.retrieval import MemoryRetriever


def test_semantic_retrieval_finds_meaningfully_related_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS currently uses Gemma 4 through Ollama.",
        importance=0.9,
    )

    assert memory is not None

    results = manager.recall(
        "What language model does AEGIS use?"
    )

    assert results

    assert any(
        result.id == memory.id
        for result in results
    )


def test_semantic_retrieval_can_find_memory_without_exact_terms():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "The system communicates with the local model "
        "through Ollama.",
        importance=0.9,
    )

    assert memory is not None

    results = manager.recall(
        "How does AEGIS communicate with its AI model?"
    )

    assert results

    assert any(
        result.id == memory.id
        for result in results
    )


def test_unrelated_memory_has_low_semantic_similarity():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4 through Ollama.",
        importance=0.9,
    )

    assert memory is not None

    retriever = manager.retriever

    results = retriever.semantic_candidates(
        "My favorite programming language is Python.",
        limit=10,
    )

    assert not any(
        result.id == memory.id
        for result, _score in results
    )


def test_semantic_candidates_are_sorted():
    manager = MemoryManager(":memory:")

    first = manager.remember(
        "AEGIS uses Gemma 4 through Ollama.",
        importance=0.9,
    )

    second = manager.remember(
        "The AEGIS project uses a language model.",
        importance=0.8,
    )

    assert first is not None
    assert second is not None

    results = manager.retriever.semantic_candidates(
        "What model does AEGIS use?",
        limit=10,
    )

    scores = [
        score
        for _memory, score
        in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_semantic_retrieval_ignores_stale_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    manager.stale_memory(
        memory.id
    )

    results = manager.retriever.semantic_candidates(
        "What model does AEGIS use?",
        limit=10,
    )

    assert not any(
        result.id == memory.id
        for result, _score in results
    )


def test_semantic_retrieval_ignores_archived_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    manager.archive_memory(
        memory.id
    )

    results = manager.retriever.semantic_candidates(
        "What model does AEGIS use?",
        limit=10,
    )

    assert not any(
        result.id == memory.id
        for result, _score in results
    )


def test_hybrid_retrieval_deduplicates_candidates():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4 through Ollama.",
        importance=0.9,
    )

    assert memory is not None

    candidates = (
        manager.retriever._collect_candidates(
            "AEGIS Gemma Ollama",
            limit=20,
        )
    )

    ids = [
        candidate.memory.id
        for candidate in candidates
    ]

    assert len(ids) == len(
        set(ids)
    )


def test_retriever_can_be_constructed_with_custom_weights():
    manager = MemoryManager(":memory:")

    retriever = MemoryRetriever(
        manager.store,
        embedder=manager.embedder,
        lexical_weight=0.7,
        semantic_weight=0.3,
    )

    assert np.isclose(
        retriever.lexical_weight,
        0.7,
    )

    assert np.isclose(
        retriever.semantic_weight,
        0.3,
    )