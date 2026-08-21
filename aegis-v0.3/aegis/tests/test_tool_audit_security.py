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
from core.tool_audit_security import (
    SecurityFinding,
    ToolAuditSecurityAnalyzer,
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


def make_analyzer(
    records,
):
    return ToolAuditSecurityAnalyzer(
        ToolAuditAnalyzer(
            records
        )
    )


def test_empty_security_analyzer():
    analyzer = make_analyzer([])

    assert analyzer.findings() == []


def test_repeated_failures():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="dangerous_tool",
                success=False,
                error="failure",
            ),
            make_record(
                tool_name="dangerous_tool",
                success=False,
                error="failure",
            ),
            make_record(
                tool_name="dangerous_tool",
                success=False,
                error="failure",
            ),
        ]
    )

    findings = analyzer.repeated_failures(
        threshold=3
    )

    assert len(findings) == 1

    finding = findings[0]

    assert isinstance(
        finding,
        SecurityFinding,
    )

    assert (
        finding.finding_type
        == "REPEATED_FAILURES"
    )

    assert (
        finding.tool_name
        == "dangerous_tool"
    )

    assert finding.count == 3
    assert finding.severity == "MEDIUM"


def test_repeated_failures_ignore_below_threshold():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="system_info",
                success=False,
                error="failure",
            ),
            make_record(
                tool_name="system_info",
                success=False,
                error="failure",
            ),
        ]
    )

    assert (
        analyzer.repeated_failures(
            threshold=3
        )
        == []
    )


def test_repeated_denials():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="high_risk_tool",
                decision=CONFIRM,
                confirmed=False,
                executed=False,
                success=False,
            ),
            make_record(
                tool_name="high_risk_tool",
                decision=CONFIRM,
                confirmed=False,
                executed=False,
                success=False,
            ),
            make_record(
                tool_name="high_risk_tool",
                decision=CONFIRM,
                confirmed=False,
                executed=False,
                success=False,
            ),
        ]
    )

    findings = analyzer.repeated_denials(
        threshold=3
    )

    assert len(findings) == 1

    assert (
        findings[0].finding_type
        == "REPEATED_DENIALS"
    )

    assert (
        findings[0].count
        == 3
    )


def test_high_risk_activity():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="delete_data",
                risk_category="HIGH",
                decision=CONFIRM,
                confirmed=True,
            )
        ]
    )

    findings = (
        analyzer.high_risk_activity()
    )

    assert len(findings) == 1

    assert (
        findings[0].finding_type
        == "HIGH_RISK_ACTIVITY"
    )

    assert (
        findings[0].severity
        == "HIGH"
    )


def test_high_risk_is_case_insensitive():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="dangerous",
                risk_category="high",
                decision=CONFIRM,
                confirmed=True,
            )
        ]
    )

    assert len(
        analyzer.high_risk_activity()
    ) == 1


def test_repeated_confirmations():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="medium_tool",
                decision=CONFIRM,
                confirmed=True,
            ),
            make_record(
                tool_name="medium_tool",
                decision=CONFIRM,
                confirmed=True,
            ),
            make_record(
                tool_name="medium_tool",
                decision=CONFIRM,
                confirmed=True,
            ),
        ]
    )

    findings = (
        analyzer.repeated_confirmations(
            threshold=3
        )
    )

    assert len(findings) == 1

    assert (
        findings[0].finding_type
        == "REPEATED_CONFIRMATIONS"
    )

    assert (
        findings[0].severity
        == "LOW"
    )


def test_high_frequency_tools():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="system_info"
            ),
            make_record(
                tool_name="system_info"
            ),
            make_record(
                tool_name="system_info"
            ),
        ]
    )

    findings = (
        analyzer.high_frequency_tools(
            threshold=3
        )
    )

    assert len(findings) == 1

    assert (
        findings[0].finding_type
        == "HIGH_TOOL_FREQUENCY"
    )

    assert findings[0].count == 3


def test_combined_findings():
    analyzer = make_analyzer(
        [
            make_record(
                tool_name="dangerous",
                risk_category="HIGH",
                decision=CONFIRM,
                confirmed=True,
                success=False,
                error="failure",
            ),
            make_record(
                tool_name="dangerous",
                risk_category="HIGH",
                decision=CONFIRM,
                confirmed=True,
                success=False,
                error="failure",
            ),
            make_record(
                tool_name="dangerous",
                risk_category="HIGH",
                decision=CONFIRM,
                confirmed=True,
                success=False,
                error="failure",
            ),
        ]
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


def test_invalid_failure_threshold():
    analyzer = make_analyzer([])

    try:
        analyzer.repeated_failures(
            threshold=0
        )
        assert False
    except ValueError:
        pass


def test_invalid_denial_threshold():
    analyzer = make_analyzer([])

    try:
        analyzer.repeated_denials(
            threshold=0
        )
        assert False
    except ValueError:
        pass


def test_invalid_high_risk_threshold():
    analyzer = make_analyzer([])

    try:
        analyzer.high_risk_activity(
            threshold=0
        )
        assert False
    except ValueError:
        pass


def test_security_analyzer_rejects_invalid_analyzer():
    try:
        ToolAuditSecurityAnalyzer(
            None
        )
        assert False
    except TypeError:
        pass