from datetime import datetime, timedelta, timezone

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


def make_record(
    *,
    tool_name="system_info",
    risk_category="LOW",
    decision=ALLOW,
    confirmed=False,
    executed=True,
    success=True,
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
        arguments={},
        result=(
            "ok"
            if success
            else None
        ),
        error=error,
        timestamp=timestamp,
    )


def test_empty_analyzer():
    analyzer = ToolAuditAnalyzer([])

    assert analyzer.records() == []

    summary = analyzer.summarize()

    assert summary.total == 0
    assert summary.executed == 0
    assert summary.successful == 0
    assert summary.failed == 0
    assert summary.denied == 0
    assert summary.success_rate == 0.0


def test_records_returns_copy():
    record = make_record()

    analyzer = ToolAuditAnalyzer(
        [record]
    )

    records = analyzer.records()

    assert records == [record]

    records.clear()

    assert analyzer.records() == [record]


def test_recent_returns_newest_first():
    now = datetime.now(
        timezone.utc
    )

    old = make_record(
        tool_name="old",
        timestamp=(
            now
            - timedelta(
                minutes=5
            )
        ),
    )

    new = make_record(
        tool_name="new",
        timestamp=now,
    )

    analyzer = ToolAuditAnalyzer(
        [old, new]
    )

    recent = analyzer.recent()

    assert [
        record.tool_name
        for record in recent
    ] == [
        "new",
        "old",
    ]


def test_recent_respects_limit():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                tool_name="one"
            ),
            make_record(
                tool_name="two"
            ),
            make_record(
                tool_name="three"
            ),
        ]
    )

    assert len(
        analyzer.recent(
            limit=2
        )
    ) == 2

    assert (
        analyzer.recent(
            limit=0
        )
        == []
    )


def test_filter_for_tool():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                tool_name="system_info"
            ),
            make_record(
                tool_name="list_processes"
            ),
            make_record(
                tool_name="system_info"
            ),
        ]
    )

    records = analyzer.for_tool(
        "system_info"
    )

    assert len(records) == 2

    assert all(
        record.tool_name
        == "system_info"
        for record in records
    )


def test_filter_for_risk():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                risk_category="LOW"
            ),
            make_record(
                risk_category="HIGH"
            ),
        ]
    )

    records = analyzer.for_risk(
        "HIGH"
    )

    assert len(records) == 1
    assert (
        records[0].risk_category
        == "HIGH"
    )


def test_failures():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                success=True
            ),
            make_record(
                executed=True,
                success=False,
                error="failure",
            ),
            make_record(
                executed=False,
                success=False,
            ),
        ]
    )

    failures = analyzer.failures()

    assert len(failures) == 1

    assert (
        failures[0].error
        == "failure"
    )


def test_confirmed_records():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                confirmed=False
            ),
            make_record(
                confirmed=True
            ),
        ]
    )

    assert len(
        analyzer.confirmed()
    ) == 1


def test_confirmation_required():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                decision=CONFIRM,
                confirmed=False,
                executed=False,
                success=False,
            ),
            make_record(
                decision=CONFIRM,
                confirmed=True,
                executed=True,
                success=True,
            ),
        ]
    )

    records = (
        analyzer.confirmation_required()
    )

    assert len(records) == 1

    assert (
        records[0].decision
        == CONFIRM
    )


def test_denied_records():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                executed=True
            ),
            make_record(
                executed=False,
                success=False,
            ),
        ]
    )

    assert len(
        analyzer.denied()
    ) == 1


def test_high_risk_records():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                risk_category="LOW"
            ),
            make_record(
                risk_category="HIGH"
            ),
            make_record(
                risk_category="high"
            ),
        ]
    )

    assert len(
        analyzer.high_risk()
    ) == 2


def test_summary():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                executed=True,
                success=True,
            ),
            make_record(
                executed=True,
                success=False,
                error="failed",
            ),
            make_record(
                decision=CONFIRM,
                confirmed=False,
                executed=False,
                success=False,
            ),
            make_record(
                decision=CONFIRM,
                confirmed=True,
                executed=True,
                success=True,
            ),
        ]
    )

    summary = analyzer.summarize()

    assert summary.total == 4
    assert summary.executed == 3
    assert summary.successful == 2
    assert summary.failed == 2
    assert summary.denied == 1
    assert summary.confirmation_required == 1
    assert summary.confirmed == 1

    assert (
        summary.success_rate
        == 2 / 3
    )


def test_tool_usage_counts():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                tool_name="system_info"
            ),
            make_record(
                tool_name="system_info"
            ),
            make_record(
                tool_name="list_processes"
            ),
        ]
    )

    assert (
        analyzer.tool_usage_counts()
        == {
            "system_info": 2,
            "list_processes": 1,
        }
    )


def test_successful_tool_counts():
    analyzer = ToolAuditAnalyzer(
        [
            make_record(
                tool_name="system_info",
                success=True,
            ),
            make_record(
                tool_name="system_info",
                success=False,
                error="failure",
            ),
            make_record(
                tool_name="list_processes",
                success=True,
            ),
        ]
    )

    assert (
        analyzer.successful_tool_counts()
        == {
            "system_info": 1,
            "list_processes": 1,
        }
    )


def test_invalid_tool_name_is_rejected():
    analyzer = ToolAuditAnalyzer([])

    try:
        analyzer.for_tool(None)
        assert False
    except TypeError:
        pass


def test_invalid_risk_category_is_rejected():
    analyzer = ToolAuditAnalyzer([])

    try:
        analyzer.for_risk(None)
        assert False
    except TypeError:
        pass