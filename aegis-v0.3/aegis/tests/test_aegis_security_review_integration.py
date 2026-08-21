from core.aegis import AEGIS
from core.llm_client import (
    LLMResponse,
    MockClient,
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


def make_response():
    return LLMResponse(
        content=[],
        stop_reason="end_turn",
    )


def make_store(tmp_path):
    return SQLiteToolAuditStore(
        str(
            tmp_path
            / "aegis_security_review.db"
        )
    )


def make_record(
    *,
    tool_name="system_info",
    success=True,
    error=None,
):
    return ToolAuditRecord(
        tool_name=tool_name,
        risk_category="LOW",
        decision="ALLOW",
        confirmed=False,
        executed=True,
        success=success,
        arguments={},
        result="ok" if success else None,
        error=error,
    )


def make_aegis(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    aegis = AEGIS(
        MockClient(
            responses=[
                make_response()
            ]
        ),
        tool_audit_store=store,
    )

    return aegis, store


def test_aegis_exposes_security_reviewer(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path
    )

    try:
        assert isinstance(
            aegis.security_review,
            ToolSecurityReviewer,
        )

    finally:
        store.close()


def test_aegis_security_review_returns_review_object(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path
    )

    try:
        result = (
            aegis.security_review.review()
        )

        assert isinstance(
            result,
            ToolSecurityReview,
        )

        assert result.findings == []
        assert result.decisions == []

    finally:
        store.close()


def test_aegis_security_review_reads_audit_history(
    tmp_path,
):
    aegis, store = make_aegis(
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

        result = (
            aegis.security_review.review()
        )

        assert len(
            result.findings
        ) == 1

        assert (
            result.findings[0].tool_name
            == "failing_tool"
        )

        assert (
            result.decisions[0].action
            == REVIEW
        )

    finally:
        store.close()


def test_aegis_security_review_can_use_custom_policy(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path
    )

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


def test_aegis_security_review_is_read_only(
    tmp_path,
):
    aegis, store = make_aegis(
        tmp_path
    )

    try:
        store.save(
            make_record()
        )

        before = store.list()

        aegis.security_review.review()

        after = store.list()

        assert before == after

    finally:
        store.close()