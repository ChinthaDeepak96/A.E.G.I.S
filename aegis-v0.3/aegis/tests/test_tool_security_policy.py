from core.tool_audit_security import (
    SecurityFinding,
)
from core.tool_security_policy import (
    BLOCK,
    INFO,
    REVIEW,
    SecurityPolicyDecision,
    ToolSecurityPolicy,
)


def make_finding(
    *,
    finding_type="HIGH_RISK_ACTIVITY",
    severity="HIGH",
    tool_name="run_command",
    count=1,
    message="security finding",
):
    return SecurityFinding(
        finding_type=finding_type,
        severity=severity,
        tool_name=tool_name,
        count=count,
        message=message,
    )


def test_high_risk_defaults_to_review():
    policy = ToolSecurityPolicy()

    decision = policy.evaluate(
        make_finding(
            finding_type="HIGH_RISK_ACTIVITY"
        )
    )

    assert isinstance(
        decision,
        SecurityPolicyDecision,
    )

    assert decision.action == REVIEW


def test_high_risk_can_be_blocked():
    policy = ToolSecurityPolicy(
        block_high_risk=True
    )

    decision = policy.evaluate(
        make_finding(
            finding_type="HIGH_RISK_ACTIVITY"
        )
    )

    assert decision.action == BLOCK


def test_repeated_failures_default_to_review():
    policy = ToolSecurityPolicy()

    decision = policy.evaluate(
        make_finding(
            finding_type="REPEATED_FAILURES",
            severity="MEDIUM",
            count=3,
        )
    )

    assert decision.action == REVIEW


def test_repeated_failures_can_be_blocked():
    policy = ToolSecurityPolicy(
        block_repeated_failures=True
    )

    decision = policy.evaluate(
        make_finding(
            finding_type="REPEATED_FAILURES",
            severity="MEDIUM",
            count=3,
        )
    )

    assert decision.action == BLOCK


def test_repeated_denials_default_to_review():
    policy = ToolSecurityPolicy()

    decision = policy.evaluate(
        make_finding(
            finding_type="REPEATED_DENIALS",
            severity="MEDIUM",
            count=3,
        )
    )

    assert decision.action == REVIEW


def test_repeated_denials_can_be_blocked():
    policy = ToolSecurityPolicy(
        block_repeated_denials=True
    )

    decision = policy.evaluate(
        make_finding(
            finding_type="REPEATED_DENIALS",
            severity="MEDIUM",
            count=3,
        )
    )

    assert decision.action == BLOCK


def test_repeated_confirmations_are_information():
    policy = ToolSecurityPolicy()

    decision = policy.evaluate(
        make_finding(
            finding_type="REPEATED_CONFIRMATIONS",
            severity="LOW",
            count=3,
        )
    )

    assert decision.action == INFO


def test_high_frequency_is_information():
    policy = ToolSecurityPolicy()

    decision = policy.evaluate(
        make_finding(
            finding_type="HIGH_TOOL_FREQUENCY",
            severity="LOW",
            count=10,
        )
    )

    assert decision.action == INFO


def test_unknown_finding_fails_closed_to_review():
    policy = ToolSecurityPolicy()

    decision = policy.evaluate(
        make_finding(
            finding_type="UNKNOWN_SIGNAL"
        )
    )

    assert decision.action == REVIEW


def test_evaluate_all():
    policy = ToolSecurityPolicy()

    findings = [
        make_finding(
            finding_type="HIGH_RISK_ACTIVITY"
        ),
        make_finding(
            finding_type="REPEATED_CONFIRMATIONS",
            severity="LOW",
        ),
    ]

    decisions = policy.evaluate_all(
        findings
    )

    assert len(decisions) == 2

    assert decisions[0].action == REVIEW
    assert decisions[1].action == INFO


def test_evaluate_all_does_not_modify_input():
    policy = ToolSecurityPolicy()

    findings = [
        make_finding()
    ]

    original = list(
        findings
    )

    policy.evaluate_all(
        findings
    )

    assert findings == original


def test_invalid_finding_is_rejected():
    policy = ToolSecurityPolicy()

    try:
        policy.evaluate(None)
        assert False
    except TypeError:
        pass


def test_invalid_findings_collection_is_rejected():
    policy = ToolSecurityPolicy()

    try:
        policy.evaluate_all(None)
        assert False
    except TypeError:
        pass