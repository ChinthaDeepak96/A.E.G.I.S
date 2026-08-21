from core.guardian import (
    ALLOW,
    CONFIRM,
)
from core.tool_audit import (
    ToolAuditRecord,
)
from core.tool_security_policy import (
    BLOCK,
    REVIEW,
    ToolSecurityPolicy,
)
from core.tool_security_review import (
    ToolSecurityReview,
    ToolSecurityReviewer,
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
        result="ok" if success else None,
        error=error,
    )


def make_store(tmp_path):
    return SQLiteToolAuditStore(
        str(
            tmp_path
            / "security_review.db"
        )
    )


def test_reviewer_rejects_invalid_store():
    try:
        ToolSecurityReviewer(None)
        assert False
    except TypeError:
        pass


def test_reviewer_uses_default_policy(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        reviewer = ToolSecurityReviewer(
            store
        )

        result = reviewer.review()

        assert isinstance(
            result,
            ToolSecurityReview,
        )

        assert result.findings == []
        assert result.decisions == []

    finally:
        store.close()


def test_review_reads_security_findings(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="failing_tool",
                    success=False,
                    error="failure",
                )
            )

        reviewer = ToolSecurityReviewer(
            store
        )

        result = reviewer.review()

        assert len(
            result.findings
        ) == 1

        assert (
            result.findings[0].finding_type
            == "REPEATED_FAILURES"
        )

        assert (
            result.findings[0].tool_name
            == "failing_tool"
        )

    finally:
        store.close()


def test_review_applies_policy(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="failing_tool",
                    success=False,
                    error="failure",
                )
            )

        policy = ToolSecurityPolicy(
            block_repeated_failures=True
        )

        reviewer = ToolSecurityReviewer(
            store,
            policy=policy,
        )

        result = reviewer.review()

        assert len(
            result.decisions
        ) == 1

        assert (
            result.decisions[0].action
            == BLOCK
        )

    finally:
        store.close()


def test_review_defaults_to_review(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="failing_tool",
                    success=False,
                    error="failure",
                )
            )

        reviewer = ToolSecurityReviewer(
            store
        )

        decisions = reviewer.decisions()

        assert len(decisions) == 1

        assert (
            decisions[0].action
            == REVIEW
        )

    finally:
        store.close()


def test_blocked_returns_only_block_decisions(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="blocked_tool",
                    success=False,
                    error="failure",
                )
            )

        policy = ToolSecurityPolicy(
            block_repeated_failures=True
        )

        reviewer = ToolSecurityReviewer(
            store,
            policy=policy,
        )

        blocked = reviewer.blocked()

        assert len(blocked) == 1

        assert (
            blocked[0].action
            == BLOCK
        )

    finally:
        store.close()


def test_review_required_returns_review_decisions(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        for _ in range(3):
            store.save(
                make_record(
                    tool_name="review_tool",
                    success=False,
                    error="failure",
                )
            )

        reviewer = ToolSecurityReviewer(
            store
        )

        required = (
            reviewer.review_required()
        )

        assert len(required) == 1

        assert (
            required[0].action
            == REVIEW
        )

    finally:
        store.close()


def test_reviewer_is_read_only(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        store.save(
            make_record()
        )

        before = store.list()

        reviewer = ToolSecurityReviewer(
            store
        )

        reviewer.review()

        after = store.list()

        assert before == after

    finally:
        store.close()