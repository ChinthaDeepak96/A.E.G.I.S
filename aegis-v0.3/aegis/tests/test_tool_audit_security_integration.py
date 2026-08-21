from core.guardian import (
    ALLOW,
    CONFIRM,
)
from core.tool_audit import (
    ToolAuditRecord,
)
from core.tool_audit_security import (
    ToolAuditSecurityAnalyzer,
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
            / "security_audit.db"
        )
    )


def test_store_security_analysis_returns_security_analyzer(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        analyzer = (
            store.security_analysis()
        )

        assert isinstance(
            analyzer,
            ToolAuditSecurityAnalyzer,
        )

    finally:
        store.close()


def test_security_analysis_reads_persisted_failures(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="failing_tool",
                    success=False,
                    error="failure",
                )
            )

        analyzer = (
            store.security_analysis()
        )

        findings = (
            analyzer.repeated_failures(
                threshold=3
            )
        )

        assert len(findings) == 1

        assert (
            findings[0].tool_name
            == "failing_tool"
        )

        assert (
            findings[0].count
            == 3
        )

    finally:
        store.close()


def test_security_analysis_reads_persisted_denials(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="protected_tool",
                    decision=CONFIRM,
                    confirmed=False,
                    executed=False,
                    success=False,
                )
            )

        analyzer = (
            store.security_analysis()
        )

        findings = (
            analyzer.repeated_denials(
                threshold=3
            )
        )

        assert len(findings) == 1

        assert (
            findings[0].tool_name
            == "protected_tool"
        )

        assert (
            findings[0].count
            == 3
        )

    finally:
        store.close()


def test_security_analysis_reads_persisted_high_risk_activity(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        store.save(
            make_record(
                tool_name="high_risk_tool",
                risk_category="HIGH",
                decision=CONFIRM,
                confirmed=True,
                executed=True,
                success=True,
            )
        )

        analyzer = (
            store.security_analysis()
        )

        findings = (
            analyzer.high_risk_activity()
        )

        assert len(findings) == 1

        assert (
            findings[0].tool_name
            == "high_risk_tool"
        )

        assert (
            findings[0].severity
            == "HIGH"
        )

    finally:
        store.close()


def test_security_analysis_uses_current_snapshot(
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

        first = (
            store.security_analysis()
        )

        store.save(
            make_record(
                tool_name="second_tool"
            )
        )

        # The first analyzer represents the snapshot
        # that existed when it was created.
        assert (
            first._analyzer.records()
            != []
        )

        assert len(
            first._analyzer.records()
        ) == 1

        # A new analyzer sees the newly persisted record.
        second = (
            store.security_analysis()
        )

        assert len(
            second._analyzer.records()
        ) == 2

    finally:
        store.close()


def test_security_analysis_combines_multiple_signals(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="dangerous_tool",
                    risk_category="HIGH",
                    decision=CONFIRM,
                    confirmed=True,
                    executed=True,
                    success=False,
                    error="failure",
                )
            )

        analyzer = (
            store.security_analysis()
        )

        findings = analyzer.findings(
            failure_threshold=3,
            high_risk_threshold=1,
            confirmation_threshold=3,
            frequency_threshold=10,
        )

        finding_types = {
            finding.finding_type
            for finding in findings
        }

        assert (
            "REPEATED_FAILURES"
            in finding_types
        )

        assert (
            "HIGH_RISK_ACTIVITY"
            in finding_types
        )

        assert (
            "REPEATED_CONFIRMATIONS"
            in finding_types
        )

    finally:
        store.close()