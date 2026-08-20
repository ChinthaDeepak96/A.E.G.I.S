from datetime import datetime, timezone

import numpy as np

from memory import Memory, MemoryStatus, MemoryType
from memory.embeddings import MemoryEmbedder
from memory.store import SQLiteMemoryStore


def create_memory(
    content: str = "AEGIS uses Gemma 4.",
) -> Memory:
    return Memory(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=0.9,
        confidence=1.0,
        sensitivity=0.0,
    )


def create_embedding() -> np.ndarray:
    """
    Create a deterministic test vector.

    Store-level tests should not need to load the embedding model.
    """
    return np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float32,
    )


def test_embedding_table_is_created():
    store = SQLiteMemoryStore(":memory:")

    with store._connect() as conn:
        row = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'memory_embeddings'
            """
        ).fetchone()

    assert row is not None
    assert row["name"] == "memory_embeddings"


def test_save_and_get_embedding():
    store = SQLiteMemoryStore(":memory:")

    memory = create_memory()
    store.save(memory)

    embedding = create_embedding()

    store.save_embedding(
        memory_id=memory.id,
        embedding=embedding,
        model_name="test-model",
    )

    result = store.get_embedding(
        memory.id
    )

    assert result is not None

    vector, model_name, dimensions, created_at = result

    assert isinstance(
        vector,
        np.ndarray,
    )

    assert vector.dtype == np.float32
    assert vector.shape == (4,)

    assert np.allclose(
        vector,
        embedding,
    )

    assert model_name == "test-model"
    assert dimensions == 4
    assert created_at


def test_embedding_survives_store_reopen(tmp_path):
    database = tmp_path / "memory.db"

    store = SQLiteMemoryStore(
        database
    )

    memory = create_memory()

    store.save(memory)

    embedding = create_embedding()

    store.save_embedding(
        memory.id,
        embedding,
        "test-model",
    )

    # Re-open the database using a new store.
    reopened = SQLiteMemoryStore(
        database
    )

    result = reopened.get_embedding(
        memory.id
    )

    assert result is not None

    vector, model_name, dimensions, _ = result

    assert np.allclose(
        vector,
        embedding,
    )

    assert model_name == "test-model"
    assert dimensions == 4


def test_get_embedding_returns_none_for_unknown_memory():
    store = SQLiteMemoryStore(":memory:")

    result = store.get_embedding(
        "does-not-exist"
    )

    assert result is None


def test_delete_embedding():
    store = SQLiteMemoryStore(":memory:")

    memory = create_memory()

    store.save(memory)

    embedding = create_embedding()

    store.save_embedding(
        memory.id,
        embedding,
        "test-model",
    )

    assert (
        store.get_embedding(
            memory.id
        )
        is not None
    )

    deleted = store.delete_embedding(
        memory.id
    )

    assert deleted is True

    assert (
        store.get_embedding(
            memory.id
        )
        is None
    )


def test_delete_missing_embedding_returns_false():
    store = SQLiteMemoryStore(":memory:")

    deleted = store.delete_embedding(
        "does-not-exist"
    )

    assert deleted is False


def test_saving_embedding_replaces_existing_embedding():
    store = SQLiteMemoryStore(":memory:")

    memory = create_memory()

    store.save(memory)

    first = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    second = np.array(
        [0.0, 1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    store.save_embedding(
        memory.id,
        first,
        "model-a",
    )

    store.save_embedding(
        memory.id,
        second,
        "model-b",
    )

    result = store.get_embedding(
        memory.id
    )

    assert result is not None

    vector, model_name, dimensions, _ = result

    assert np.allclose(
        vector,
        second,
    )

    assert model_name == "model-b"
    assert dimensions == 4


def test_embedding_is_removed_when_memory_is_deleted():
    store = SQLiteMemoryStore(":memory:")

    memory = create_memory()

    store.save(memory)

    store.save_embedding(
        memory.id,
        create_embedding(),
        "test-model",
    )

    assert (
        store.get_embedding(
            memory.id
        )
        is not None
    )

    deleted = store.delete(
        memory.id
    )

    assert deleted is True

    assert (
        store.get(memory.id)
        is None
    )

    assert (
        store.get_embedding(
            memory.id
        )
        is None
    )


def test_embedding_dimensions_are_recorded_correctly():
    store = SQLiteMemoryStore(":memory:")

    memory = create_memory()

    store.save(memory)

    embedding = np.arange(
        8,
        dtype=np.float32,
    )

    store.save_embedding(
        memory.id,
        embedding,
        "test-model",
    )

    result = store.get_embedding(
        memory.id
    )

    assert result is not None

    vector, _, dimensions, _ = result

    assert vector.shape == (8,)
    assert dimensions == 8


def test_embedding_values_are_preserved():
    store = SQLiteMemoryStore(":memory:")

    memory = create_memory()

    store.save(memory)

    embedding = np.array(
        [
            -0.25,
            0.0,
            0.125,
            0.999,
        ],
        dtype=np.float32,
    )

    store.save_embedding(
        memory.id,
        embedding,
        "test-model",
    )

    result = store.get_embedding(
        memory.id
    )

    assert result is not None

    vector, _, _, _ = result

    assert np.array_equal(
        vector,
        embedding,
    )