from memory.manager import MemoryManager
from memory.retrieval import MemoryRetriever
from memory.scoring import MemoryScorer
from memory.usage_store import SQLiteMemoryUsageStore


def make_manager_and_retriever(
    database,
    usage_database,
):
    manager = MemoryManager(
        str(database)
    )

    usage_store = SQLiteMemoryUsageStore(
        str(usage_database)
    )

    retriever = MemoryRetriever(
        manager.store,
        scorer=MemoryScorer(),
        usage_store=usage_store,
    )

    return (
        manager,
        retriever,
        usage_store,
    )


def test_first_retrieval_persists_usage(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    (
        manager,
        retriever,
        usage_store,
    ) = make_manager_and_retriever(
        database,
        usage_database,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    results = retriever.retrieve(
        "Gemma"
    )

    assert results
    assert results[0].id == memory.id

    usage = usage_store.get(
        memory.id
    )

    assert usage is not None
    assert usage.retrieval_count == 1
    assert usage.access_count == 1
    assert usage.last_retrieved_at is not None

    usage_store.close()


def test_second_retrieval_increments_persisted_usage(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    (
        manager,
        retriever,
        usage_store,
    ) = make_manager_and_retriever(
        database,
        usage_database,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    retriever.retrieve("Gemma")
    retriever.retrieve("Gemma")

    usage = usage_store.get(
        memory.id
    )

    assert usage is not None
    assert usage.retrieval_count == 2
    assert usage.access_count == 2

    usage_store.close()


def test_usage_survives_new_retriever(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    manager = MemoryManager(
        str(database)
    )

    usage_store = SQLiteMemoryUsageStore(
        str(usage_database)
    )

    retriever = MemoryRetriever(
        manager.store,
        scorer=MemoryScorer(),
        usage_store=usage_store,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    retriever.retrieve(
        "Gemma"
    )

    usage_store.close()

    # ---------------------------------------------------------
    # Create a completely new usage store and retriever.
    # ---------------------------------------------------------

    reopened_usage_store = (
        SQLiteMemoryUsageStore(
            str(usage_database)
        )
    )

    reopened_retriever = MemoryRetriever(
        manager.store,
        scorer=MemoryScorer(),
        usage_store=reopened_usage_store,
    )

    usage = reopened_retriever.get_usage(
        memory.id
    )

    assert usage.retrieval_count == 1
    assert usage.access_count == 1
    assert usage.last_retrieved_at is not None

    reopened_usage_store.close()


def test_usage_continues_after_retriever_recreation(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    manager = MemoryManager(
        str(database)
    )

    first_usage_store = (
        SQLiteMemoryUsageStore(
            str(usage_database)
        )
    )

    first_retriever = MemoryRetriever(
        manager.store,
        scorer=MemoryScorer(),
        usage_store=first_usage_store,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    first_retriever.retrieve(
        "Gemma"
    )

    first_usage_store.close()

    second_usage_store = (
        SQLiteMemoryUsageStore(
            str(usage_database)
        )
    )

    second_retriever = MemoryRetriever(
        manager.store,
        scorer=MemoryScorer(),
        usage_store=second_usage_store,
    )

    second_retriever.retrieve(
        "Gemma"
    )

    usage = second_usage_store.get(
        memory.id
    )

    assert usage is not None
    assert usage.retrieval_count == 2
    assert usage.access_count == 2

    second_usage_store.close()


def test_last_retrieved_timestamp_persists(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    (
        manager,
        retriever,
        usage_store,
    ) = make_manager_and_retriever(
        database,
        usage_database,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    retriever.retrieve(
        "Gemma"
    )

    usage_before = usage_store.get(
        memory.id
    )

    assert usage_before is not None

    timestamp = (
        usage_before.last_retrieved_at
    )

    assert timestamp is not None

    usage_store.close()

    reopened_store = (
        SQLiteMemoryUsageStore(
            str(usage_database)
        )
    )

    usage_after = reopened_store.get(
        memory.id
    )

    assert usage_after is not None

    assert (
        usage_after.last_retrieved_at
        == timestamp
    )

    reopened_store.close()


def test_empty_query_does_not_create_usage(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    (
        manager,
        retriever,
        usage_store,
    ) = make_manager_and_retriever(
        database,
        usage_database,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    results = retriever.retrieve(
        ""
    )

    assert results == []

    usage = usage_store.get(
        memory.id
    )

    assert usage is None

    usage_store.close()


def test_no_result_query_does_not_create_usage(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    (
        manager,
        retriever,
        usage_store,
    ) = make_manager_and_retriever(
        database,
        usage_database,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    results = retriever.retrieve(
        "zzzzzzzzzzzzzzzz"
    )

    assert results == []

    usage = usage_store.get(
        memory.id
    )

    assert usage is None

    usage_store.close()


def test_only_returned_memory_usage_is_persisted(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    (
        manager,
        retriever,
        usage_store,
    ) = make_manager_and_retriever(
        database,
        usage_database,
    )

    first = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    second = manager.remember(
        "A.E.G.I.S. uses Ollama.",
        importance=0.9,
    )

    assert first is not None
    assert second is not None

    results = retriever.retrieve(
        "Gemma",
        limit=1,
    )

    assert results
    assert len(results) == 1
    assert results[0].id == first.id

    first_usage = usage_store.get(
        first.id
    )

    second_usage = usage_store.get(
        second.id
    )

    assert first_usage is not None
    assert first_usage.retrieval_count == 1
    assert first_usage.access_count == 1

    assert second_usage is None

    usage_store.close()


def test_usage_does_not_modify_memory(
    tmp_path,
):
    database = (
        tmp_path / "memory.db"
    )

    usage_database = (
        tmp_path / "usage.db"
    )

    (
        manager,
        retriever,
        usage_store,
    ) = make_manager_and_retriever(
        database,
        usage_database,
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    original_id = memory.id
    original_content = memory.content
    original_status = memory.status
    original_updated_at = (
        memory.updated_at
    )

    results = retriever.retrieve(
        "Gemma"
    )

    assert results
    assert results[0].id == original_id

    assert memory.id == original_id
    assert memory.content == original_content
    assert memory.status == original_status
    assert (
        memory.updated_at
        == original_updated_at
    )

    usage_store.close()