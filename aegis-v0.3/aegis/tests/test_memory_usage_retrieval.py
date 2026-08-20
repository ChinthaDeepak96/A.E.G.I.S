from memory.manager import MemoryManager
from memory.retrieval import MemoryRetriever
from memory.scoring import MemoryScorer


def make_manager_and_retriever():
    manager = MemoryManager(":memory:")

    retriever = MemoryRetriever(
        manager.store,
        scorer=MemoryScorer(),
    )

    return manager, retriever


def test_retrieved_memory_records_usage():
    manager, retriever = (
        make_manager_and_retriever()
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

    usage = retriever.get_usage(
        memory.id
    )

    assert usage.retrieval_count == 1
    assert usage.access_count == 1
    assert usage.last_retrieved_at is not None


def test_multiple_retrievals_accumulate_usage():
    manager, retriever = (
        make_manager_and_retriever()
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    retriever.retrieve("Gemma")
    retriever.retrieve("Gemma")
    retriever.retrieve("Gemma")

    usage = retriever.get_usage(
        memory.id
    )

    assert usage.retrieval_count == 3
    assert usage.access_count == 3
    assert usage.last_retrieved_at is not None


def test_empty_query_does_not_record_usage():
    manager, retriever = (
        make_manager_and_retriever()
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    results = retriever.retrieve("")

    assert results == []

    usage = retriever.get_usage(
        memory.id
    )

    assert usage.retrieval_count == 0
    assert usage.access_count == 0
    assert usage.last_retrieved_at is None


def test_no_result_query_does_not_record_usage():
    manager, retriever = (
        make_manager_and_retriever()
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    # Force a query that should not produce a candidate under
    # the configured semantic threshold.
    results = retriever.retrieve(
        "zzzzzzzzzzzzzzzz"
    )

    assert results == []

    usage = retriever.get_usage(
        memory.id
    )

    assert usage.retrieval_count == 0
    assert usage.access_count == 0
    assert usage.last_retrieved_at is None

def test_only_returned_memories_record_usage():
    manager, retriever = (
        make_manager_and_retriever()
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

    first_usage = retriever.get_usage(
        first.id
    )

    second_usage = retriever.get_usage(
        second.id
    )

    assert first_usage.retrieval_count == 1
    assert first_usage.access_count == 1

    assert second_usage.retrieval_count == 0
    assert second_usage.access_count == 0


def test_usage_does_not_change_memory():
    manager, retriever = (
        make_manager_and_retriever()
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    original_id = memory.id
    original_content = memory.content
    original_status = memory.status
    original_updated_at = memory.updated_at

    results = retriever.retrieve(
        "Gemma"
    )

    assert results
    assert results[0].id == original_id

    assert memory.id == original_id
    assert memory.content == original_content
    assert memory.status == original_status
    assert memory.updated_at == original_updated_at