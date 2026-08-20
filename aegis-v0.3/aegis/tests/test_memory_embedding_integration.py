import numpy as np

from memory import MemoryManager


def test_remember_creates_embedding():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    result = manager.store.get_embedding(
        memory.id
    )

    assert result is not None

    embedding, model_name, dimensions, _ = result

    assert embedding.shape == (
        384,
    )

    assert dimensions == 384

    assert model_name == (
        manager.embedder.model_name
    )


def test_recalled_embedding_matches_memory():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4 through Ollama.",
        importance=0.9,
    )

    assert memory is not None

    result = manager.store.get_embedding(
        memory.id
    )

    assert result is not None

    embedding, _, _, _ = result

    expected = manager.embedder.encode(
        memory.content
    )

    assert np.allclose(
        embedding,
        expected,
        atol=1e-5,
    )


def test_update_memory_regenerates_embedding():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    assert memory is not None

    before = manager.store.get_embedding(
        memory.id
    )

    assert before is not None

    old_embedding = before[0].copy()

    updated = manager.update_memory(
        memory.id,
        "AEGIS uses Gemma 4.",
    )

    assert updated is not None

    after = manager.store.get_embedding(
        memory.id
    )

    assert after is not None

    new_embedding = after[0]

    assert not np.allclose(
        old_embedding,
        new_embedding,
        atol=1e-5,
    )

    expected = manager.embedder.encode(
        "AEGIS uses Gemma 4."
    )

    assert np.allclose(
        new_embedding,
        expected,
        atol=1e-5,
    )


def test_forget_removes_embedding():
    manager = MemoryManager(":memory:")

    memory = manager.remember(
        "AEGIS uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    assert (
        manager.store.get_embedding(
            memory.id
        )
        is not None
    )

    assert manager.forget(
        memory.id
    ) is True

    assert (
        manager.store.get_embedding(
            memory.id
        )
        is None
    )