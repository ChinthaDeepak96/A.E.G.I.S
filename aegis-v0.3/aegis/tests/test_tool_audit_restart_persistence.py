from core.guardian import ALLOW
from core.tool_audit import ToolAuditRecord
from core.tool_audit_store import SQLiteToolAuditStore
from core.tool_security_policy import (
    BLOCK,
    ToolSecurityPolicy,
)
from core.tool_security_review import (
    ToolSecurityReviewer,
)


def make_record(
    *,
    tool_name="persistent_tool",
    decision=ALLOW,
    executed=False,
    success=False,
    error="failure",
):
    return ToolAuditRecord(
        tool_name=tool_name,
        risk_category="LOW",
        decision=decision,
        confirmed=False,
        executed=executed,
        success=success,
        arguments={
            "source": "restart-test"
        },
        result=None,
        error=error,
    )


def test_audit_survives_store_recreation(
    tmp_path,
):
    database = (
        tmp_path
        / "persistent_audit.db"
    )

    store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        original = store.save(
            make_record()
        )

        original_id = original.id

    finally:
        store.close()

    # Simulate a fresh A.E.G.I.S. process.
    new_store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        records = new_store.list()

        assert len(records) == 1

        restored = records[0]

        assert restored.id == original_id

        assert (
            restored.tool_name
            == "persistent_tool"
        )

        assert (
            restored.arguments
            == {
                "source": "restart-test"
            }
        )

        assert (
            restored.error
            == "failure"
        )

    finally:
        new_store.close()


def test_security_analysis_survives_store_recreation(
    tmp_path,
):
    database = (
        tmp_path
        / "persistent_security.db"
    )

    store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="persistent_tool"
                )
            )

    finally:
        store.close()

    # Fresh store instance.
    new_store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        analyzer = (
            new_store.security_analysis()
        )

        findings = analyzer.findings()

        assert len(findings) >= 1

        matching = [
            finding
            for finding in findings
            if finding.tool_name
            == "persistent_tool"
        ]

        assert len(matching) == 1

        assert (
            matching[0].finding_type
            == "REPEATED_FAILURES"
        )

    finally:
        new_store.close()


def test_security_policy_still_blocks_after_restart(
    tmp_path,
):
    database = (
        tmp_path
        / "persistent_policy.db"
    )

    store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="persistent_tool"
                )
            )

    finally:
        store.close()

    # Simulate restart.
    new_store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        policy = ToolSecurityPolicy(
            block_repeated_failures=True
        )

        reviewer = ToolSecurityReviewer(
            new_store,
            policy=policy,
        )

        decisions = (
            reviewer.decisions()
        )

        assert len(decisions) >= 1

        matching = [
            decision
            for decision in decisions
            if decision.tool_name
            == "persistent_tool"
        ]

        assert len(matching) == 1

        assert (
            matching[0].action
            == BLOCK
        )

    finally:
        new_store.close()


def test_recent_survives_store_recreation(
    tmp_path,
):
    database = (
        tmp_path
        / "persistent_recent.db"
    )

    store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        for index in range(5):
            store.save(
                make_record(
                    tool_name=(
                        f"tool_{index}"
                    )
                )
            )

    finally:
        store.close()

    new_store = SQLiteToolAuditStore(
        str(database)
    )

    try:
        recent = new_store.recent(
            limit=3
        )

        assert len(recent) == 3

        assert (
            recent[0].tool_name
            == "tool_4"
        )

        assert (
            recent[1].tool_name
            == "tool_3"
        )

        assert (
            recent[2].tool_name
            == "tool_2"
        )

    finally:
        new_store.close()