from datetime import datetime, timezone

from memory.usage import MemoryUsage
from memory.usage_store import SQLiteMemoryUsageStore


def test_usage_store_creates_empty_database(
    tmp_path,
):
    database = tmp_path / "usage.db"

    store = SQLiteMemoryUsageStore(
        str(database)
    )

    assert store.count() == 0

    store.close()


def test_save_and_get_usage(
    tmp_path,
):
    database = tmp_path / "usage.db"

    store = SQLiteMemoryUsageStore(
        str(database)
    )

    usage = MemoryUsage(
        retrieval_count=5,
        access_count=8,
        last_retrieved_at=(
            "2026-08-21T10:00:00+00:00"
        ),
    )

    store.save(
        "memory-1",
        usage,
    )

    restored = store.get(
        "memory-1"
    )

    assert restored is not None

    assert (
        restored.retrieval_count
        == 5
    )

    assert (
        restored.access_count
        == 8
    )

    assert (
        restored.last_retrieved_at
        == "2026-08-21T10:00:00+00:00"
    )

    store.close()


def test_get_unknown_usage_returns_none(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    assert (
        store.get("unknown")
        is None
    )

    store.close()


def test_save_updates_existing_usage(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    first = MemoryUsage(
        retrieval_count=1,
        access_count=2,
    )

    second = MemoryUsage(
        retrieval_count=7,
        access_count=9,
    )

    store.save(
        "memory-1",
        first,
    )

    store.save(
        "memory-1",
        second,
    )

    restored = store.get(
        "memory-1"
    )

    assert restored is not None

    assert (
        restored.retrieval_count
        == 7
    )

    assert (
        restored.access_count
        == 9
    )

    assert store.count() == 1

    store.close()


def test_get_or_create_creates_usage(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    usage = store.get_or_create(
        "memory-1"
    )

    assert isinstance(
        usage,
        MemoryUsage,
    )

    assert usage.retrieval_count == 0
    assert usage.access_count == 0

    assert store.exists(
        "memory-1"
    )

    store.close()


def test_get_or_create_returns_existing_usage(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    usage = MemoryUsage(
        retrieval_count=4,
        access_count=6,
    )

    store.save(
        "memory-1",
        usage,
    )

    restored = store.get_or_create(
        "memory-1"
    )

    assert (
        restored.retrieval_count
        == 4
    )

    assert (
        restored.access_count
        == 6
    )

    assert store.count() == 1

    store.close()


def test_delete_usage(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    store.save(
        "memory-1",
        MemoryUsage(
            retrieval_count=2
        ),
    )

    assert store.delete(
        "memory-1"
    ) is True

    assert (
        store.get("memory-1")
        is None
    )

    assert store.delete(
        "memory-1"
    ) is False

    store.close()


def test_exists_returns_false_for_unknown_memory(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    assert (
        store.exists(
            "does-not-exist"
        )
        is False
    )

    store.close()


def test_exists_returns_true_after_save(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    store.save(
        "memory-1",
        MemoryUsage(),
    )

    assert (
        store.exists(
            "memory-1"
        )
        is True
    )

    store.close()


def test_empty_memory_id_is_rejected(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    try:
        store.save(
            "",
            MemoryUsage(),
        )
        assert False
    except ValueError:
        pass

    store.close()


def test_usage_survives_store_reopen(
    tmp_path,
):
    database = tmp_path / "usage.db"

    store = SQLiteMemoryUsageStore(
        str(database)
    )

    usage = MemoryUsage(
        retrieval_count=12,
        access_count=15,
        last_retrieved_at=(
            "2026-08-21T12:30:00+00:00"
        ),
    )

    store.save(
        "memory-1",
        usage,
    )

    store.close()

    reopened = SQLiteMemoryUsageStore(
        str(database)
    )

    restored = reopened.get(
        "memory-1"
    )

    assert restored is not None

    assert (
        restored.retrieval_count
        == 12
    )

    assert (
        restored.access_count
        == 15
    )

    assert (
        restored.last_retrieved_at
        == "2026-08-21T12:30:00+00:00"
    )

    reopened.close()


def test_retrieval_timestamp_is_preserved(
    tmp_path,
):
    store = SQLiteMemoryUsageStore(
        str(
            tmp_path / "usage.db"
        )
    )

    timestamp = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    usage = MemoryUsage(
        retrieval_count=1,
        access_count=1,
        last_retrieved_at=timestamp,
    )

    store.save(
        "memory-1",
        usage,
    )

    restored = store.get(
        "memory-1"
    )

    assert restored is not None

    assert (
        restored.last_retrieved_at
        == timestamp
    )

    store.close()