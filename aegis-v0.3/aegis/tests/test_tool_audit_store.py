from datetime import datetime, timedelta, timezone

import pytest

from core.tool_audit import ToolAuditRecord
from core.tool_audit_store import SQLiteToolAuditStore


def make_record(
    *,
    tool_name="test_tool",
    risk_category="low",
    decision="ALLOW",
    confirmed=False,
    executed=True,
    success=True,
    arguments=None,
    result=None,
    error=None,
    timestamp=None,
):
    return ToolAuditRecord(
        tool_name=tool_name,
        risk_category=risk_category,
        decision=decision,
        confirmed=confirmed,
        executed=executed,
        success=success,
        arguments=(
            {}
            if arguments is None
            else arguments
        ),
        result=result,
        error=error,
        timestamp=timestamp,
    )


@pytest.fixture
def store(tmp_path):
    database_path = (
        tmp_path / "tool_audit.db"
    )

    audit_store = SQLiteToolAuditStore(
        str(database_path)
    )

    yield audit_store

    audit_store.close()


# =========================================================
# INITIALIZATION
# =========================================================


def test_store_initializes_database(store):
    assert store is not None


def test_empty_store_returns_no_records(store):
    assert store.list() == []


def test_empty_store_recent_returns_no_records(store):
    assert store.recent() == []


# =========================================================
# SAVE
# =========================================================


def test_save_returns_record_with_id(store):
    record = make_record()

    saved = store.save(record)

    assert saved is record
    assert getattr(
        saved,
        "id",
        None,
    ) is not None


def test_saved_record_can_be_retrieved(store):
    record = make_record(
        tool_name="read_file",
        risk_category="low",
        decision="ALLOW",
        arguments={
            "path": "hello.txt"
        },
        result="hello",
    )

    saved = store.save(record)

    loaded = store.get(
        saved.id
    )

    assert loaded is not None
    assert (
        loaded.id
        == saved.id
    )
    assert (
        loaded.tool_name
        == "read_file"
    )
    assert (
        loaded.risk_category
        == "low"
    )
    assert (
        loaded.decision
        == "ALLOW"
    )
    assert (
        loaded.arguments
        == {
            "path": "hello.txt"
        }
    )
    assert (
        loaded.result
        == "hello"
    )


def test_save_preserves_confirmation_state(store):
    record = make_record(
        decision="CONFIRM",
        confirmed=True,
        executed=True,
        success=True,
    )

    saved = store.save(record)

    loaded = store.get(
        saved.id
    )

    assert loaded.confirmed is True
    assert loaded.executed is True
    assert loaded.success is True


def test_save_preserves_failure_information(store):
    record = make_record(
        decision="CONFIRM",
        confirmed=True,
        executed=True,
        success=False,
        error="handler failed",
    )

    saved = store.save(record)

    loaded = store.get(
        saved.id
    )

    assert loaded.success is False
    assert (
        loaded.error
        == "handler failed"
    )


# =========================================================
# GET
# =========================================================


def test_get_unknown_id_returns_none(store):
    assert (
        store.get(
            "does-not-exist"
        )
        is None
    )


def test_get_empty_id_returns_none(store):
    assert store.get("") is None


def test_get_whitespace_id_returns_none(store):
    assert (
        store.get(
            "   "
        )
        is None
    )


# =========================================================
# LIST
# =========================================================


def test_list_returns_saved_records(store):
    first = store.save(
        make_record(
            tool_name="first"
        )
    )

    second = store.save(
        make_record(
            tool_name="second"
        )
    )

    records = store.list()

    assert len(records) == 2

    ids = {
        record.id
        for record in records
    }

    assert first.id in ids
    assert second.id in ids


def test_list_preserves_saved_record_data(store):
    store.save(
        make_record(
            tool_name="system_info",
            risk_category="low",
            decision="ALLOW",
            arguments={
                "detail": True
            },
            result="system information",
        )
    )

    records = store.list()

    assert len(records) == 1

    record = records[0]

    assert (
        record.tool_name
        == "system_info"
    )

    assert (
        record.arguments
        == {
            "detail": True
        }
    )

    assert (
        record.result
        == "system information"
    )


# =========================================================
# ORDERING
# =========================================================


def test_list_returns_newest_records_first(store):
    now = datetime.now(
        timezone.utc
    )

    older = store.save(
        make_record(
            tool_name="older",
            timestamp=(
                now
                - timedelta(
                    minutes=2
                )
            ),
        )
    )

    newer = store.save(
        make_record(
            tool_name="newer",
            timestamp=(
                now
                - timedelta(
                    minutes=1
                )
            ),
        )
    )

    records = store.list()

    assert (
        records[0].id
        == newer.id
    )

    assert (
        records[1].id
        == older.id
    )


# =========================================================
# RECENT
# =========================================================


def test_recent_returns_limited_records(store):
    for index in range(5):
        store.save(
            make_record(
                tool_name=f"tool_{index}"
            )
        )

    records = store.recent(
        limit=3
    )

    assert len(records) == 3


def test_recent_zero_returns_empty(store):
    store.save(
        make_record()
    )

    assert (
        store.recent(
            limit=0
        )
        == []
    )


def test_recent_negative_returns_empty(store):
    store.save(
        make_record()
    )

    assert (
        store.recent(
            limit=-1
        )
        == []
    )


def test_recent_preserves_newest_first(store):
    now = datetime.now(
        timezone.utc
    )

    older = store.save(
        make_record(
            tool_name="older",
            timestamp=(
                now
                - timedelta(
                    minutes=2
                )
            ),
        )
    )

    newer = store.save(
        make_record(
            tool_name="newer",
            timestamp=(
                now
                - timedelta(
                    minutes=1
                )
            ),
        )
    )

    records = store.recent(
        limit=2
    )

    assert (
        records[0].id
        == newer.id
    )

    assert (
        records[1].id
        == older.id
    )


# =========================================================
# CLEAR
# =========================================================


def test_clear_removes_all_records(store):
    store.save(
        make_record(
            tool_name="first"
        )
    )

    store.save(
        make_record(
            tool_name="second"
        )
    )

    removed = store.clear()

    assert removed == 2
    assert store.list() == []


def test_clear_empty_store_returns_zero(store):
    assert store.clear() == 0


def test_store_can_be_reused_after_clear(store):
    store.save(
        make_record(
            tool_name="before"
        )
    )

    store.clear()

    record = store.save(
        make_record(
            tool_name="after"
        )
    )

    records = store.list()

    assert len(records) == 1
    assert (
        records[0].id
        == record.id
    )


# =========================================================
# PERSISTENCE
# =========================================================


def test_records_survive_store_reopen(
    tmp_path,
):
    database_path = (
        tmp_path / "persistent.db"
    )

    first_store = SQLiteToolAuditStore(
        str(database_path)
    )

    saved = first_store.save(
        make_record(
            tool_name="persistent_tool",
            arguments={
                "value": 42
            },
            result="persisted",
        )
    )

    first_store.close()

    second_store = SQLiteToolAuditStore(
        str(database_path)
    )

    try:
        loaded = second_store.get(
            saved.id
        )

        assert loaded is not None
        assert (
            loaded.tool_name
            == "persistent_tool"
        )
        assert (
            loaded.arguments
            == {
                "value": 42
            }
        )
        assert (
            loaded.result
            == "persisted"
        )

    finally:
        second_store.close()


# =========================================================
# SERIALIZATION
# =========================================================


def test_complex_arguments_survive_storage(store):
    arguments = {
        "command": "dir",
        "options": {
            "recursive": True,
            "limit": 10,
        },
        "items": [
            "one",
            "two",
        ],
    }

    saved = store.save(
        make_record(
            arguments=arguments
        )
    )

    loaded = store.get(
        saved.id
    )

    assert loaded is not None
    assert (
        loaded.arguments
        == arguments
    )


def test_none_result_and_error_survive_storage(
    store,
):
    saved = store.save(
        make_record(
            result=None,
            error=None,
        )
    )

    loaded = store.get(
        saved.id
    )

    assert loaded is not None
    assert loaded.result is None
    assert loaded.error is None


# =========================================================
# MULTIPLE STORES
# =========================================================


def test_different_databases_are_independent(
    tmp_path,
):
    first_store = SQLiteToolAuditStore(
        str(
            tmp_path / "first.db"
        )
    )

    second_store = SQLiteToolAuditStore(
        str(
            tmp_path / "second.db"
        )
    )

    try:
        first_store.save(
            make_record(
                tool_name="first"
            )
        )

        second_store.save(
            make_record(
                tool_name="second"
            )
        )

        assert len(
            first_store.list()
        ) == 1

        assert len(
            second_store.list()
        ) == 1

        assert (
            first_store.list()[0].tool_name
            == "first"
        )

        assert (
            second_store.list()[0].tool_name
            == "second"
        )

    finally:
        first_store.close()
        second_store.close()


# =========================================================
# CLOSE
# =========================================================


def test_close_can_be_called_multiple_times(
    store,
):
    store.close()
    store.close()