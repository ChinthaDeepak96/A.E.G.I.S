from core.guardian import (
    ALLOW,
    CONFIRM,
)
from core.tool_audit import (
    ToolAuditRecord,
)
from core.tool_audit_analyzer import (
    ToolAuditAnalyzer,
)
from core.tool_audit_store import (
    SQLiteToolAuditStore,
)


def make_record(
    *,
    tool_name="system_info",
    risk_category="LOW",
    decision=ALLOW,
    confirmed=False,
    executed=True,
    success=True,
    error=None,
):
    return ToolAuditRecord(
        tool_name=tool_name,
        risk_category=risk_category,
        decision=decision,
        confirmed=confirmed,
        executed=executed,
        success=success,
        arguments={},
        result=(
            "ok"
            if success
            else None
        ),
        error=error,
    )


def make_store(tmp_path):
    return SQLiteToolAuditStore(
        str(
            tmp_path
            / "audit.db"
        )
    )


def test_store_analyze_returns_analyzer(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        analyzer = store.analyze()

        assert isinstance(
            analyzer,
            ToolAuditAnalyzer,
        )

        assert (
            analyzer.records()
            == []
        )

    finally:
        store.close()


def test_store_analyzer_reads_persisted_records(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        store.save(
            make_record(
                tool_name="system_info"
            )
        )

        store.save(
            make_record(
                tool_name="list_processes"
            )
        )

        analyzer = store.analyze()

        records = (
            analyzer.records()
        )

        assert len(records) == 2

        assert {
            record.tool_name
            for record in records
        } == {
            "system_info",
            "list_processes",
        }

    finally:
        store.close()


def test_store_analyzer_summary_uses_persisted_data(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        store.save(
            make_record(
                executed=True,
                success=True,
            )
        )

        store.save(
            make_record(
                executed=True,
                success=False,
                error="failure",
            )
        )

        store.save(
            make_record(
                decision=CONFIRM,
                confirmed=False,
                executed=False,
                success=False,
            )
        )

        analyzer = store.analyze()

        summary = (
            analyzer.summarize()
        )

        assert summary.total == 3
        assert summary.executed == 2
        assert summary.successful == 1
        assert summary.failed == 2
        assert summary.denied == 1
        assert (
            summary.confirmation_required
            == 1
        )

    finally:
        store.close()


def test_store_analyzer_finds_failures(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        store.save(
            make_record(
                success=True
            )
        )

        store.save(
            make_record(
                tool_name="failing_tool",
                executed=True,
                success=False,
                error="simulated failure",
            )
        )

        analyzer = store.analyze()

        failures = (
            analyzer.failures()
        )

        assert len(failures) == 1

        assert (
            failures[0].tool_name
            == "failing_tool"
        )

        assert (
            failures[0].error
            == "simulated failure"
        )

    finally:
        store.close()


def test_store_analyzer_finds_high_risk_activity(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        store.save(
            make_record(
                tool_name="safe_tool",
                risk_category="LOW",
            )
        )

        store.save(
            make_record(
                tool_name="dangerous_tool",
                risk_category="HIGH",
                decision=CONFIRM,
                confirmed=True,
            )
        )

        analyzer = store.analyze()

        records = (
            analyzer.high_risk()
        )

        assert len(records) == 1

        assert (
            records[0].tool_name
            == "dangerous_tool"
        )

    finally:
        store.close()


def test_store_analyzer_is_snapshot_based(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        store.save(
            make_record(
                tool_name="first_tool"
            )
        )

        analyzer = store.analyze()

        store.save(
            make_record(
                tool_name="second_tool"
            )
        )

        # The analyzer represents the records that existed
        # when it was created.
        assert len(
            analyzer.records()
        ) == 1

        assert (
            analyzer.records()[0].tool_name
            == "first_tool"
        )

        # A newly-created analyzer sees the new record.
        refreshed = (
            store.analyze()
        )

        assert len(
            refreshed.records()
        ) == 2

    finally:
        store.close()
        