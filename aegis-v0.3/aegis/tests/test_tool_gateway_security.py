from core.guardian import (
    ALLOW,
)
from core.tool_audit import (
    ToolAuditRecord,
)
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
from core.tools import (
    Tool,
)


def make_store(tmp_path):
    return SQLiteToolAuditStore(
        str(
            tmp_path
            / "gateway_security.db"
        )
    )


def make_tool(
    *,
    name="test_tool",
    risk_category="LOW",
    handler=None,
):
    if handler is None:
        handler = (
            lambda: "executed"
        )

    return Tool(
        name=name,
        description="Test tool",
        risk_category=risk_category,
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=handler,
    )


def make_failure_record(
    *,
    tool_name="test_tool",
):
    return ToolAuditRecord(
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


def test_gateway_without_security_reviewer_behaves_normally():
    called = []

    tool = make_tool(
        handler=lambda: called.append(
            True
        ) or "ok"
    )

    gateway = ToolGateway()

    result = gateway.execute(
        tool,
        {},
    )

    assert result.executed is True
    assert result.success is True
    assert called == [True]


def test_security_review_does_not_block_review_action(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        for _ in range(3):
            store.save(
                make_failure_record(
                    tool_name="test_tool"
                )
            )

        reviewer = ToolSecurityReviewer(
            store
        )

        called = []

        tool = make_tool(
            handler=lambda: called.append(
                True
            ) or "ok"
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        result = gateway.execute(
            tool,
            {},
        )

        assert result.executed is True
        assert result.success is True
        assert called == [True]

    finally:
        store.close()


def test_security_block_prevents_handler_execution(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        for _ in range(3):
            store.save(
                make_failure_record(
                    tool_name="dangerous_tool"
                )
            )

        reviewer = ToolSecurityReviewer(
            store,
            policy=ToolSecurityPolicy(
                block_repeated_failures=True
            ),
        )

        called = []

        tool = make_tool(
            name="dangerous_tool",
            handler=lambda: called.append(
                True
            ) or "SHOULD NOT RUN"
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        result = gateway.execute(
            tool,
            {},
        )

        assert result.executed is False
        assert result.success is False
        assert (
            result.decision
            == "SECURITY_BLOCK"
        )

        assert called == []

    finally:
        store.close()


def test_security_block_is_tool_specific(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        for _ in range(3):
            store.save(
                make_failure_record(
                    tool_name="blocked_tool"
                )
            )

        reviewer = ToolSecurityReviewer(
            store,
            policy=ToolSecurityPolicy(
                block_repeated_failures=True
            ),
        )

        called = []

        allowed_tool = make_tool(
            name="allowed_tool",
            handler=lambda: called.append(
                "allowed"
            ) or "ok",
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        result = gateway.execute(
            allowed_tool,
            {},
        )

        assert result.executed is True
        assert result.success is True
        assert called == ["allowed"]

    finally:
        store.close()


def test_security_block_does_not_invoke_handler(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        for _ in range(3):
            store.save(
                make_failure_record(
                    tool_name="blocked_tool"
                )
            )

        reviewer = ToolSecurityReviewer(
            store,
            policy=ToolSecurityPolicy(
                block_repeated_failures=True
            ),
        )

        invocation_count = 0

        def handler():
            nonlocal invocation_count
            invocation_count += 1
            return "executed"

        tool = make_tool(
            name="blocked_tool",
            handler=handler,
        )

        gateway = ToolGateway(
            audit_store=store,
            security_reviewer=reviewer,
        )

        result = gateway.execute(
            tool,
            {},
        )

        assert result.executed is False
        assert invocation_count == 0

    finally:
        store.close()