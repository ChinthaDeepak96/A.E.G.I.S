from core.guardian import ALLOW
from core.tool_audit_store import (
    SQLiteToolAuditStore,
)
from core.tool_gateway import (
    ToolGateway,
)
from core.tool_security_policy import (
    ToolSecurityPolicy,
)
from core.tool_security_review import (
    ToolSecurityReviewer,
)
from core.tools import Tool


def make_store(tmp_path):
    return SQLiteToolAuditStore(
        str(
            tmp_path
            / "security_block_audit.db"
        )
    )


def make_tool(
    *,
    name="blocked_tool",
):
    return Tool(
        name=name,
        description="Test tool",
        risk_category="LOW",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=lambda: "SHOULD NOT EXECUTE",
    )


def seed_failures(
    store,
    tool_name,
):
    for _ in range(3):
        from core.tool_audit import (
            ToolAuditRecord,
        )

        store.save(
            ToolAuditRecord(
                tool_name=tool_name,
                risk_category="LOW",
                decision=ALLOW,
                confirmed=False,
                executed=False,
                success=False,
                arguments={},
                result=None,
                error="failure",
            )
        )


def test_security_block_is_audited(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        seed_failures(
            store,
            "blocked_tool",
        )

        reviewer = ToolSecurityReviewer(
            store,
            policy=ToolSecurityPolicy(
                block_repeated_failures=True
            ),
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        result = gateway.execute(
            make_tool(),
            {
                "test": "value"
            },
        )

        assert result.executed is False
        assert (
            result.decision
            == "SECURITY_BLOCK"
        )

        records = store.list()

        assert len(records) == 4

        block_record = records[0]

        assert (
            block_record.tool_name
            == "blocked_tool"
        )

        assert (
            block_record.decision
            == "SECURITY_BLOCK"
        )

        assert (
            block_record.executed
            is False
        )

        assert (
            block_record.success
            is False
        )

        assert (
            block_record.confirmed
            is False
        )

        assert (
            block_record.arguments
            == {
                "test": "value"
            }
        )

    finally:
        store.close()


def test_security_block_has_error_reason(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        seed_failures(
            store,
            "blocked_tool",
        )

        reviewer = ToolSecurityReviewer(
            store,
            policy=ToolSecurityPolicy(
                block_repeated_failures=True
            ),
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        result = gateway.execute(
            make_tool(),
            {},
        )

        records = store.list()

        block_record = records[0]

        assert (
            block_record.decision
            == "SECURITY_BLOCK"
        )

        assert (
            block_record.error
            is not None
        )

        assert (
            "blocked"
            in block_record.error.lower()
        )

    finally:
        store.close()


def test_security_block_does_not_count_as_success(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        seed_failures(
            store,
            "blocked_tool",
        )

        reviewer = ToolSecurityReviewer(
            store,
            policy=ToolSecurityPolicy(
                block_repeated_failures=True
            ),
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        gateway.execute(
            make_tool(),
            {},
        )

        records = store.list()

        block_record = records[0]

        assert (
            block_record.executed
            is False
        )

        assert (
            block_record.success
            is False
        )

    finally:
        store.close()


def test_security_block_preserves_original_audit_history(
    tmp_path,
):
    store = make_store(tmp_path)

    try:
        seed_failures(
            store,
            "blocked_tool",
        )

        before = store.list()

        reviewer = ToolSecurityReviewer(
            store,
            policy=ToolSecurityPolicy(
                block_repeated_failures=True
            ),
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        gateway.execute(
            make_tool(),
            {},
        )

        after = store.list()

        assert len(after) == (
            len(before) + 1
        )

        assert after[1:] == before

    finally:
        store.close()