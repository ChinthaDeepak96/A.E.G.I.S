from datetime import datetime, timedelta, timezone

from memory.usage import MemoryUsage


def test_new_usage_has_zero_counts():
    usage = MemoryUsage()

    assert usage.retrieval_count == 0
    assert usage.access_count == 0
    assert usage.last_retrieved_at is None


def test_record_retrieval_increments_counts():
    usage = MemoryUsage()

    timestamp = datetime.now(
        timezone.utc
    )

    usage.record_retrieval(
        timestamp=timestamp
    )

    assert usage.retrieval_count == 1
    assert usage.access_count == 1
    assert usage.last_retrieved_at is not None


def test_multiple_retrievals_increment_count():
    usage = MemoryUsage()

    usage.record_retrieval()
    usage.record_retrieval()
    usage.record_retrieval()

    assert usage.retrieval_count == 3
    assert usage.access_count == 3


def test_record_access_only_increments_access_count():
    usage = MemoryUsage()

    usage.record_access()
    usage.record_access()

    assert usage.access_count == 2
    assert usage.retrieval_count == 0


def test_has_been_retrieved():
    usage = MemoryUsage()

    assert (
        usage.has_been_retrieved()
        is False
    )

    usage.record_retrieval()

    assert (
        usage.has_been_retrieved()
        is True
    )


def test_days_since_retrieval():
    now = datetime.now(
        timezone.utc
    )

    retrieved_at = (
        now
        - timedelta(days=10)
    )

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=retrieved_at
    )

    days = usage.days_since_retrieval(
        now=now
    )

    assert days is not None
    assert 9.9 <= days <= 10.1


def test_days_since_retrieval_returns_none_before_first_retrieval():
    usage = MemoryUsage()

    assert (
        usage.days_since_retrieval()
        is None
    )


def test_usage_serialization():
    usage = MemoryUsage()

    usage.record_retrieval()

    data = usage.to_dict()

    assert data["retrieval_count"] == 1
    assert data["access_count"] == 1
    assert data["last_retrieved_at"] is not None


def test_usage_can_be_restored_from_dict():
    usage = MemoryUsage(
        retrieval_count=5,
        last_retrieved_at=(
            "2026-08-20T10:00:00+00:00"
        ),
        access_count=7,
    )

    restored = MemoryUsage.from_dict(
        usage.to_dict()
    )

    assert (
        restored.retrieval_count
        == 5
    )

    assert (
        restored.access_count
        == 7
    )

    assert (
        restored.last_retrieved_at
        == "2026-08-20T10:00:00+00:00"
    )


def test_empty_dict_creates_default_usage():
    usage = MemoryUsage.from_dict({})

    assert usage.retrieval_count == 0
    assert usage.access_count == 0
    assert usage.last_retrieved_at is None


def test_reset_clears_usage():
    usage = MemoryUsage()

    usage.record_retrieval()
    usage.record_access()

    usage.reset()

    assert usage.retrieval_count == 0
    assert usage.access_count == 0
    assert usage.last_retrieved_at is None


def test_negative_counts_are_normalized_when_restored():
    usage = MemoryUsage.from_dict(
        {
            "retrieval_count": -10,
            "access_count": -5,
        }
    )

    assert usage.retrieval_count == 0
    assert usage.access_count == 0