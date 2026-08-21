from datetime import datetime, timezone

from core.tool_audit import ToolAuditRecord
from core.tool_audit_store import SQLiteToolAuditStore
from core.tool_gateway import ToolGateway
from core.tools import (
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    Tool,
)


def make_tool(
    *,
    name="test_tool",
    risk_category=RISK_LOW,
    handler=None,
):
    if handler is None:
        handler = (
            lambda **kwargs:
            "tool executed"
        )

    return Tool(
        name=name,
        description="Audit integration test tool.",
        risk_category=risk_category,
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=handler,
    )


def make_store(
    tmp_path,
):
    return SQLiteToolAuditStore(
        str(
            tmp_path
            / "tool_audit.db"
        )
    )


# =========================================================
# STORE / GATEWAY SETUP
# =========================================================


def test_gateway_can_accept_audit_store(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        assert gateway is not None

    finally:
        store.close()


# =========================================================
# LOW-RISK EXECUTION
# =========================================================


def test_low_risk_execution_creates_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="low_tool",
            risk_category=RISK_LOW,
        )

        result = gateway.execute(
            tool,
            {
                "value": 42
            },
        )

        assert result.executed is True
        assert result.success is True

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "low_tool"
        )

        assert (
            record.risk_category
            == RISK_LOW
        )

        assert (
            record.decision
            == "ALLOW"
        )

        assert (
            record.confirmed
            is False
        )

        assert (
            record.executed
            is True
        )

        assert (
            record.success
            is True
        )

        assert (
            record.arguments
            == {
                "value": 42
            }
        )

    finally:
        store.close()


# =========================================================
# CONFIRMATION REQUIRED
# =========================================================


def test_medium_risk_denial_creates_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="medium_tool",
            risk_category=RISK_MEDIUM,
        )

        result = gateway.execute(
            tool,
            {
                "action": "test"
            },
            confirmed=False,
        )

        assert (
            result.requires_confirmation
            is True
        )

        assert (
            result.executed
            is False
        )

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "medium_tool"
        )

        assert (
            record.risk_category
            == RISK_MEDIUM
        )

        assert (
            record.decision
            == "CONFIRM"
        )

        assert (
            record.confirmed
            is False
        )

        assert (
            record.executed
            is False
        )

        assert (
            record.success
            is False
        )

    finally:
        store.close()


def test_medium_risk_confirmed_execution_creates_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="medium_confirmed",
            risk_category=RISK_MEDIUM,
            handler=lambda **kwargs: (
                "confirmed result"
            ),
        )

        result = gateway.execute(
            tool,
            {
                "approved": True
            },
            confirmed=True,
        )

        assert (
            result.executed
            is True
        )

        assert (
            result.success
            is True
        )

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "medium_confirmed"
        )

        assert (
            record.decision
            == "CONFIRM"
        )

        assert (
            record.confirmed
            is True
        )

        assert (
            record.executed
            is True
        )

        assert (
            record.success
            is True
        )

        assert (
            record.result
            == "confirmed result"
        )

    finally:
        store.close()


# =========================================================
# HIGH-RISK EXECUTION
# =========================================================


def test_high_risk_denial_creates_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="high_tool",
            risk_category=RISK_HIGH,
        )

        result = gateway.execute(
            tool,
            {},
            confirmed=False,
        )

        assert (
            result.executed
            is False
        )

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "high_tool"
        )

        assert (
            record.risk_category
            == RISK_HIGH
        )

        assert (
            record.decision
            == "CONFIRM"
        )

        assert (
            record.confirmed
            is False
        )

        assert (
            record.executed
            is False
        )

    finally:
        store.close()


def test_high_risk_confirmed_execution_creates_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="high_confirmed",
            risk_category=RISK_HIGH,
            handler=lambda **kwargs: (
                "high result"
            ),
        )

        result = gateway.execute(
            tool,
            {
                "approved": True
            },
            confirmed=True,
        )

        assert (
            result.executed
            is True
        )

        assert (
            result.success
            is True
        )

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "high_confirmed"
        )

        assert (
            record.risk_category
            == RISK_HIGH
        )

        assert (
            record.decision
            == "CONFIRM"
        )

        assert (
            record.confirmed
            is True
        )

        assert (
            record.executed
            is True
        )

        assert (
            record.success
            is True
        )

    finally:
        store.close()


# =========================================================
# HANDLER FAILURE
# =========================================================


def test_handler_failure_creates_failed_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        def failing_handler(**kwargs):
            raise RuntimeError(
                "simulated failure"
            )

        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="failing_tool",
            risk_category=RISK_LOW,
            handler=failing_handler,
        )

        result = gateway.execute(
            tool,
            {
                "value": 123
            },
        )

        assert (
            result.executed
            is True
        )

        assert (
            result.success
            is False
        )

        records = store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "failing_tool"
        )

        assert (
            record.executed
            is True
        )

        assert (
            record.success
            is False
        )

        assert (
            record.error
            is not None
        )

        assert (
            "simulated failure"
            in record.error
        )

    finally:
        store.close()


# =========================================================
# RESULT EVIDENCE
# =========================================================


def test_successful_result_is_persisted(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="result_tool",
            handler=lambda **kwargs: (
                "important output"
            ),
        )

        gateway.execute(
            tool,
            {
                "input": "test"
            },
        )

        record = store.recent(
            limit=1
        )[0]

        assert (
            record.result
            == "important output"
        )

        assert (
            record.error
            is None
        )

    finally:
        store.close()


# =========================================================
# ARGUMENT EVIDENCE
# =========================================================


def test_tool_arguments_are_persisted(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        arguments = {
            "path": "example.txt",
            "limit": 20,
            "options": {
                "recursive": True,
            },
        }

        tool = make_tool(
            name="arguments_tool"
        )

        gateway.execute(
            tool,
            arguments,
        )

        record = store.recent(
            limit=1
        )[0]

        assert (
            record.arguments
            == arguments
        )

    finally:
        store.close()


# =========================================================
# MULTIPLE EXECUTIONS
# =========================================================


def test_every_execution_creates_an_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="repeated_tool"
        )

        gateway.execute(
            tool,
            {
                "run": 1
            },
        )

        gateway.execute(
            tool,
            {
                "run": 2
            },
        )

        gateway.execute(
            tool,
            {
                "run": 3
            },
        )

        records = store.list()

        assert len(records) == 3

        values = {
            record.arguments["run"]
            for record in records
        }

        assert values == {
            1,
            2,
            3,
        }

    finally:
        store.close()


# =========================================================
# AUDIT FAILURE MUST NOT BREAK EXECUTION
# =========================================================


def test_tool_execution_remains_successful_when_audit_store_fails(
    tmp_path,
):
    class FailingAuditStore:
        def save(self, record):
            raise RuntimeError(
                "audit database unavailable"
            )

    gateway = ToolGateway(
        audit_store=FailingAuditStore()
    )

    tool = make_tool(
        name="resilient_tool",
        handler=lambda **kwargs: (
            "execution succeeded"
        ),
    )

    result = gateway.execute(
        tool,
        {
            "value": 1
        },
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.success
        is True
    )

    assert (
        result.result
        == "execution succeeded"
    )


# =========================================================
# TIMESTAMP EVIDENCE
# =========================================================


def test_audit_record_contains_timestamp(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="timestamp_tool"
        )

        gateway.execute(
            tool,
            {},
        )

        record = store.recent(
            limit=1
        )[0]

        assert (
            record.timestamp
            is not None
        )

        assert (
            record.timestamp.tzinfo
            is not None
        )

    finally:
        store.close()


# =========================================================
# PERSISTENCE ACROSS REOPEN
# =========================================================


def test_audit_created_by_gateway_survives_store_reopen(
    tmp_path,
):
    database_path = (
        tmp_path
        / "persistent_audit.db"
    )

    first_store = (
        SQLiteToolAuditStore(
            str(database_path)
        )
    )

    gateway = ToolGateway(
        audit_store=first_store
    )

    tool = make_tool(
        name="persistent_gateway_tool",
        handler=lambda **kwargs: (
            "persisted result"
        ),
    )

    gateway.execute(
        tool,
        {
            "value": 99
        },
    )

    first_store.close()

    second_store = (
        SQLiteToolAuditStore(
            str(database_path)
        )
    )

    try:
        records = second_store.list()

        assert len(records) == 1

        record = records[0]

        assert (
            record.tool_name
            == "persistent_gateway_tool"
        )

        assert (
            record.arguments
            == {
                "value": 99
            }
        )

        assert (
            record.result
            == "persisted result"
        )

    finally:
        second_store.close()


# =========================================================
# AUDIT RECORD TYPE
# =========================================================


def test_gateway_audit_entry_is_tool_audit_record(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="typed_audit_tool"
        )

        gateway.execute(
            tool,
            {},
        )

        record = store.recent(
            limit=1
        )[0]

        assert isinstance(
            record,
            ToolAuditRecord,
        )

    finally:
        store.close()


# =========================================================
# DENIED EXECUTION MUST NOT RUN HANDLER
# =========================================================


def test_denied_tool_is_audited_without_handler_execution(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    calls = []

    try:
        def handler(**kwargs):
            calls.append(kwargs)
            return "should never happen"

        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="denied_tool",
            risk_category=RISK_HIGH,
            handler=handler,
        )

        gateway.execute(
            tool,
            {
                "danger": True
            },
            confirmed=False,
        )

        assert calls == []

        record = store.recent(
            limit=1
        )[0]

        assert (
            record.executed
            is False
        )

        assert (
            record.success
            is False
        )

        assert (
            record.confirmed
            is False
        )

    finally:
        store.close()


# =========================================================
# CONFIRMED EXECUTION EVIDENCE
# =========================================================


def test_confirmation_is_explicitly_recorded(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="confirmed_tool",
            risk_category=RISK_MEDIUM,
        )

        gateway.execute(
            tool,
            {
                "approved": True
            },
            confirmed=True,
        )

        record = store.recent(
            limit=1
        )[0]

        assert (
            record.decision
            == "CONFIRM"
        )

        assert (
            record.confirmed
            is True
        )

        assert (
            record.executed
            is True
        )

    finally:
        store.close()


# =========================================================
# AUDIT ORDERING
# =========================================================


def test_audit_records_are_stored_in_execution_order(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        first = make_tool(
            name="first_execution"
        )

        second = make_tool(
            name="second_execution"
        )

        gateway.execute(
            first,
            {},
        )

        gateway.execute(
            second,
            {},
        )

        records = store.list()

        assert len(records) == 2

        names = [
            record.tool_name
            for record in records
        ]

        assert set(names) == {
            "first_execution",
            "second_execution",
        }

    finally:
        store.close()


# =========================================================
# EMPTY RESULT
# =========================================================


def test_tool_returning_none_is_audited(
    tmp_path,
):
    store = make_store(
        tmp_path
    )

    try:
        gateway = ToolGateway(
            audit_store=store
        )

        tool = make_tool(
            name="none_result_tool",
            handler=lambda **kwargs: None,
        )

        result = gateway.execute(
            tool,
            {},
        )

        assert (
            result.executed
            is False
        )

        assert (
            result.success
            is False
        )

        record = store.recent(
            limit=1
        )[0]

        assert (
            record.success
            is False
        )

        assert (
            record.result
            is None
        )

    finally:
        store.close()